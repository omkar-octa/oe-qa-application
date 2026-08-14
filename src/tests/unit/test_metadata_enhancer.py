from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.documents import DocumentChunk
from utils.metadata_enhancer import MetadataEnhancer


def make_chunk(chunk_id: str, text: str, kind: str = "table") -> DocumentChunk:
    # Defaults to "table" because that is the only kind enhance_chunks
    # summarises; a "text" chunk is passed straight through.
    return DocumentChunk(
        chunk_id=chunk_id, file_name="paper.pdf", page_number=1, text=text, kind=kind
    )


def text_response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )


@pytest.mark.unit
def test_summarize_returns_model_text():
    client = MagicMock()
    client.messages.create.return_value = text_response("A short summary.")

    summary = MetadataEnhancer(client=client).summarize("Some raw chunk text.")

    assert summary == "A short summary."
    assert client.messages.create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Some raw chunk text."}
    ]


@pytest.mark.unit
def test_enhance_chunks_populates_summary_without_mutating_input():
    chunk = make_chunk("c0", "Raw table markdown.")
    client = MagicMock()
    client.messages.create.return_value = text_response("Describes the table's figures.")

    result = MetadataEnhancer(client=client).enhance_chunks([chunk])

    assert chunk.summary is None
    assert result[0].summary == "Describes the table's figures."
    assert result[0].chunk_id == "c0"


@pytest.mark.unit
def test_enhance_chunks_calls_model_once_per_chunk():
    client = MagicMock()
    client.messages.create.return_value = text_response("summary")
    chunks = [make_chunk("c0", "first"), make_chunk("c1", "second")]

    MetadataEnhancer(client=client).enhance_chunks(chunks)

    assert client.messages.create.call_count == 2


@pytest.mark.unit
def test_enhance_chunks_empty_list_returns_empty():
    assert MetadataEnhancer(client=MagicMock()).enhance_chunks([]) == []


@pytest.mark.unit
def test_enhance_chunks_passes_previous_chunk_as_context_within_same_file():
    first = make_chunk("c0", "Table 1 shows regional cost breakdowns.")
    second = make_chunk("c1", "| Region | Cost |\n| --- | --- |\n| UK | 100 |")
    client = MagicMock()
    client.messages.create.return_value = text_response("summary")

    MetadataEnhancer(client=client).enhance_chunks([first, second])

    second_call_content = client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert "Table 1 shows regional cost breakdowns." in second_call_content
    assert "Chunk to summarise:" in second_call_content


@pytest.mark.unit
def test_enhance_chunks_no_context_across_different_files():
    first = make_chunk("a#p1#c0", "From file a.")
    second = make_chunk("b#p1#c0", "From file b.")
    second = second.model_copy(update={"file_name": "b.pdf"})
    client = MagicMock()
    client.messages.create.return_value = text_response("summary")

    MetadataEnhancer(client=client).enhance_chunks([first, second])

    second_call_content = client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert second_call_content == "From file b."


@pytest.mark.unit
def test_enhance_chunks_summarises_tables_and_skips_other_kinds():
    chunks = [
        make_chunk("c0", "Body prose.", kind="text"),
        make_chunk("c1", "| Region | Cost |", kind="table"),
        make_chunk("c2", "Figure 1. A chart.\n\nIt shows costs rising.", kind="figure"),
    ]
    client = MagicMock()
    client.messages.create.return_value = text_response("Describes the table.")

    result = MetadataEnhancer(client=client).enhance_chunks(chunks)

    assert client.messages.create.call_count == 1
    assert [chunk.summary for chunk in result] == [None, "Describes the table.", None]


@pytest.mark.unit
def test_enhance_chunks_kinds_can_be_widened():
    chunks = [make_chunk("c0", "Body prose.", kind="text"), make_chunk("c1", "| A | B |")]
    client = MagicMock()
    client.messages.create.return_value = text_response("summary")

    result = MetadataEnhancer(client=client).enhance_chunks(chunks, kinds=("text", "table"))

    assert client.messages.create.call_count == 2
    assert all(chunk.summary == "summary" for chunk in result)


@pytest.mark.unit
def test_context_is_the_real_predecessor_not_the_previous_summarised_chunk():
    # Skipping the prose chunk must not make the second table's context the
    # first table; a table's introducing sentence is the chunk right before it.
    chunks = [
        make_chunk("c0", "| Region | Cost |", kind="table"),
        make_chunk("c1", "Table 2 shows emissions by sector.", kind="text"),
        make_chunk("c2", "| Sector | Emissions |", kind="table"),
    ]
    client = MagicMock()
    client.messages.create.return_value = text_response("summary")

    MetadataEnhancer(client=client).enhance_chunks(chunks)

    second_table_content = client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert "Table 2 shows emissions by sector." in second_table_content
    assert "| Region | Cost |" not in second_table_content


@pytest.mark.unit
def test_enhance_chunks_returns_skipped_chunks_unchanged():
    chunk = make_chunk("c0", "Body prose.", kind="text")
    client = MagicMock()

    result = MetadataEnhancer(client=client).enhance_chunks([chunk])

    client.messages.create.assert_not_called()
    assert result == [chunk]


@pytest.mark.unit
def test_summarize_discards_a_max_tokens_truncated_answer():
    # Adaptive thinking's own tokens count against max_tokens; on dense or
    # noisy content it can consume the budget before finishing the answer,
    # leaving a sentence that reads as complete but is actually cut off.
    # That must never be returned as if it were a real summary.
    client = MagicMock()
    client.messages.create.return_value = text_response(
        "Table 1 defines the three binary low-carbon technology outcome variables and re",
        stop_reason="max_tokens",
    )

    summary = MetadataEnhancer(client=client).summarize("Some raw chunk text.")

    assert summary == ""


@pytest.mark.unit
def test_enhance_chunks_tracks_chunks_whose_summary_was_discarded():
    chunk = make_chunk("c0", "| Region | Cost |\n| --- | --- |\n| UK | 100 |")
    client = MagicMock()
    client.messages.create.return_value = text_response("truncated", stop_reason="max_tokens")

    enhancer = MetadataEnhancer(client=client)
    result = enhancer.enhance_chunks([chunk])

    assert result[0].summary == ""
    assert enhancer.failed_chunks == ["c0"]


@pytest.mark.unit
def test_enhance_chunks_does_not_flag_a_skipped_chunk_as_failed():
    chunk = make_chunk("c0", "Body prose.", kind="text")
    client = MagicMock()

    enhancer = MetadataEnhancer(client=client)
    enhancer.enhance_chunks([chunk])

    assert enhancer.failed_chunks == []


@pytest.mark.unit
def test_summarize_includes_headings():
    client = MagicMock()
    client.messages.create.return_value = text_response("summary")

    MetadataEnhancer(client=client).summarize("Body text.", headings=["Results", "Cost breakdown"])

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Results > Cost breakdown" in content
