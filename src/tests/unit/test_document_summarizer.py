from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.documents import DocumentChunk
from utils.document_summarizer import MAX_HEADINGS, MAX_OPENING_CHARS, DocumentSummarizer


def make_chunk(
    file_name: str = "report.pdf",
    page_number: int = 1,
    text: str = "Body text.",
    headings=None,
    kind: str = "text",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{file_name}#p{page_number}#c0",
        file_name=file_name,
        page_number=page_number,
        text=text,
        headings=headings or [],
        kind=kind,
    )


def text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


@pytest.mark.unit
def test_summarize_returns_model_text():
    client = MagicMock()
    client.messages.create.return_value = text_response("A DESNZ impact assessment.")

    summary = DocumentSummarizer(client=client).summarize([make_chunk()])

    assert summary == "A DESNZ impact assessment."


@pytest.mark.unit
def test_summarize_with_no_chunks_makes_no_api_call():
    client = MagicMock()

    assert DocumentSummarizer(client=client).summarize([]) == ""
    client.messages.create.assert_not_called()


@pytest.mark.unit
def test_profile_carries_page_count_headings_and_content_mix():
    client = MagicMock()
    client.messages.create.return_value = text_response("profile")
    chunks = [
        make_chunk(page_number=1, text="Title page.", headings=["Introduction"]),
        make_chunk(page_number=12, text="| a | b |", headings=["Results"], kind="table"),
        make_chunk(page_number=12, text="Figure 3: map.", headings=["Results"], kind="figure"),
    ]

    DocumentSummarizer(client=client).summarize(chunks)

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "File name: report.pdf" in content
    assert "Pages: 12" in content
    assert "1 are tables and 1 are figures" in content
    assert "- Introduction" in content
    assert "- Results" in content


@pytest.mark.unit
def test_profile_dedupes_repeated_heading_paths_in_document_order():
    client = MagicMock()
    client.messages.create.return_value = text_response("profile")
    chunks = [
        make_chunk(page_number=1, headings=["Intro"]),
        make_chunk(page_number=2, headings=["Intro"]),
        make_chunk(page_number=3, headings=["Method", "Data"]),
    ]

    DocumentSummarizer(client=client).summarize(chunks)

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content.count("- Intro\n") == 1
    assert content.index("- Intro") < content.index("- Method > Data")


@pytest.mark.unit
def test_profile_truncates_the_opening_and_says_how_many_headings_were_dropped():
    client = MagicMock()
    client.messages.create.return_value = text_response("profile")
    chunks = [
        make_chunk(page_number=page, text="x" * 2000, headings=[f"Section {page}"])
        for page in range(1, MAX_HEADINGS + 12)
    ]

    DocumentSummarizer(client=client).summarize(chunks)

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content.count("x") <= MAX_OPENING_CHARS
    assert f"and {len(chunks) - MAX_HEADINGS} more" in content


@pytest.mark.unit
def test_summarize_documents_calls_the_model_once_per_file():
    client = MagicMock()
    client.messages.create.return_value = text_response("profile")
    chunks = [
        make_chunk(file_name="a.pdf", page_number=1),
        make_chunk(file_name="a.pdf", page_number=2),
        make_chunk(file_name="b.pdf", page_number=1),
    ]

    summaries = DocumentSummarizer(client=client).summarize_documents(chunks)

    assert client.messages.create.call_count == 2
    assert [doc.file_name for doc in summaries] == ["a.pdf", "b.pdf"]


@pytest.mark.unit
def test_summarize_documents_records_the_highest_page_number_as_page_count():
    client = MagicMock()
    client.messages.create.return_value = text_response("profile")
    chunks = [make_chunk(page_number=1), make_chunk(page_number=48), make_chunk(page_number=7)]

    summaries = DocumentSummarizer(client=client).summarize_documents(chunks)

    assert summaries[0].page_count == 48


@pytest.mark.unit
def test_summarize_documents_skips_a_file_whose_summary_comes_back_empty():
    """A refusal returns 200 with no content blocks. That document is still
    indexed, it just gets no profile, rather than a blank one reaching the
    answering model."""
    client = MagicMock()
    client.messages.create.side_effect = [
        SimpleNamespace(content=[]),
        text_response("a real profile"),
    ]
    chunks = [make_chunk(file_name="a.pdf"), make_chunk(file_name="b.pdf")]

    summaries = DocumentSummarizer(client=client).summarize_documents(chunks)

    assert [doc.file_name for doc in summaries] == ["b.pdf"]


@pytest.mark.unit
def test_summarize_documents_empty_list_returns_empty():
    assert DocumentSummarizer(client=MagicMock()).summarize_documents([]) == []
