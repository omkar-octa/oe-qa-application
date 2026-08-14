from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.documents import DocumentChunk, SearchResult
from utils.qa_agent import REFUSAL_MESSAGE, SEARCH_BUDGET_MESSAGE, QAAgent


class _FakeIndex:
    """Minimal search()-only stub: a query matches a chunk when any of its
    words is a case-insensitive substring of the chunk's text. Real ranking
    behaviour belongs to PostgresSearchIndex's own tests; these tests only
    need QAAgent's tool loop to see a hit or a miss on cue."""

    def __init__(self, chunks: list[DocumentChunk]):
        self._chunks = chunks

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        words = query.lower().split()
        hits = [
            SearchResult(chunk=chunk, score=1.0)
            for chunk in self._chunks
            if any(word in chunk.text.lower() for word in words)
        ]
        return hits[:top_k]


def make_index() -> _FakeIndex:
    return _FakeIndex(
        [
            DocumentChunk(
                chunk_id="report.pdf#p4#c0",
                file_name="report.pdf",
                page_number=4,
                text="Workplace charging reduces fleet running costs.",
            )
        ]
    )


def _usage(input_tokens: int = 100, output_tokens: int = 20) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def tool_use_response(query: str, tool_use_id: str = "tool_1") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use",
                id=tool_use_id,
                name="search_knowledge_base",
                input={"query": query},
            )
        ],
        usage=_usage(),
    )


def text_response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=_usage(),
    )


@pytest.mark.unit
def test_search_tool_offered_to_model():
    client = MagicMock()
    client.messages.create.side_effect = [
        tool_use_response("workplace charging costs"),
        text_response("Workplace charging lowers fleet costs (report.pdf, p. 4)."),
    ]
    agent = QAAgent(make_index(), client=client)

    result = agent.answer("What reduces fleet running costs?")

    first_call_kwargs = client.messages.create.call_args_list[0].kwargs
    assert first_call_kwargs["tools"][0]["name"] == "search_knowledge_base"
    assert result == "Workplace charging lowers fleet costs (report.pdf, p. 4)."


@pytest.mark.unit
def test_tool_result_contains_search_hits():
    client = MagicMock()
    client.messages.create.side_effect = [
        tool_use_response("fleet running costs"),
        text_response("An answer."),
    ]
    agent = QAAgent(make_index(), client=client)

    agent.answer("What reduces fleet running costs?")

    second_call_kwargs = client.messages.create.call_args_list[1].kwargs
    tool_result = second_call_kwargs["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tool_1"
    assert "report.pdf, page 4" in tool_result["content"]


@pytest.mark.unit
def test_model_can_search_more_than_once_with_refined_keywords():
    client = MagicMock()
    client.messages.create.side_effect = [
        tool_use_response("zebra quantum banjo", tool_use_id="tool_1"),
        tool_use_response("fleet running costs", tool_use_id="tool_2"),
        text_response("Workplace charging lowers fleet costs."),
    ]
    agent = QAAgent(make_index(), client=client)

    result = agent.answer("What reduces fleet running costs?")

    assert client.messages.create.call_count == 3
    assert result == "Workplace charging lowers fleet costs."


@pytest.mark.unit
def test_empty_search_result_reported_to_model_not_swallowed():
    client = MagicMock()
    client.messages.create.side_effect = [
        tool_use_response("zebra quantum banjo"),
        text_response("I could not find anything relevant."),
    ]
    agent = QAAgent(make_index(), client=client)

    result = agent.answer("Irrelevant question")

    second_call_kwargs = client.messages.create.call_args_list[1].kwargs
    tool_result = second_call_kwargs["messages"][-1]["content"][0]
    assert "No matching extracts" in tool_result["content"]
    assert result == "I could not find anything relevant."


@pytest.mark.unit
def test_answer_without_tool_use_returns_model_text():
    agent = QAAgent(make_index(), client=MagicMock())
    agent._client.messages.create.return_value = text_response("Charging at work.")
    assert agent.answer("What reduces costs? charging") == "Charging at work."


@pytest.mark.unit
def test_refusal_handled_gracefully():
    client = MagicMock()
    client.messages.create.return_value = text_response("", stop_reason="refusal")
    agent = QAAgent(make_index(), client=client)
    assert agent.answer("What reduces costs? charging") == REFUSAL_MESSAGE


@pytest.mark.unit
def test_gives_up_after_max_iterations():
    client = MagicMock()
    client.messages.create.side_effect = [
        tool_use_response(f"query {i}", tool_use_id=f"tool_{i}") for i in range(10)
    ]
    agent = QAAgent(make_index(), client=client)

    result = agent.answer("What reduces fleet running costs?")

    assert result == SEARCH_BUDGET_MESSAGE


class _ReadableIndex:
    """Index stub exposing the accessors the read_source tool needs."""

    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, top_k=5):
        return []

    def files(self):
        return list(dict.fromkeys(c.file_name for c in self._chunks))

    def document(self, file_name):
        return [c for c in self._chunks if c.file_name == file_name]

    def page(self, file_name, page_number):
        return [
            c for c in self._chunks if c.file_name == file_name and c.page_number == page_number
        ]


def _readable_chunks():
    from models.documents import DocumentChunk

    return [
        DocumentChunk(chunk_id="a#c0", file_name="a.pdf", page_number=1, text="First page text."),
        DocumentChunk(chunk_id="a#c1", file_name="a.pdf", page_number=2, text="Second page text."),
    ]


@pytest.mark.unit
def test_read_source_returns_whole_document_when_page_omitted():
    from utils.qa_agent import QAAgent

    agent = QAAgent(_ReadableIndex(_readable_chunks()), client=MagicMock())

    body = agent._read_source("a.pdf", None)

    assert "First page text." in body and "Second page text." in body
    assert "[p. 1]" in body and "[p. 2]" in body


@pytest.mark.unit
def test_read_source_returns_a_single_page():
    from utils.qa_agent import QAAgent

    agent = QAAgent(_ReadableIndex(_readable_chunks()), client=MagicMock())

    body = agent._read_source("a.pdf", 2)

    assert "Second page text." in body
    assert "First page text." not in body


@pytest.mark.unit
def test_read_source_lists_available_files_on_a_miss():
    """A wrong file name should steer the model, not dead-end it."""
    from utils.qa_agent import QAAgent

    agent = QAAgent(_ReadableIndex(_readable_chunks()), client=MagicMock())

    body = agent._read_source("typo.pdf", None)

    assert "No content found" in body
    assert "a.pdf" in body


@pytest.mark.unit
def test_read_source_degrades_when_the_retriever_is_search_only():
    """The retriever protocol only promises search(); a swapped-in backend
    without the accessors must not raise."""
    from utils.qa_agent import QAAgent

    search_only = MagicMock(spec=["search"])
    agent = QAAgent(search_only, client=MagicMock())

    body = agent._read_source("a.pdf", None)

    assert "does not support" in body


@pytest.mark.unit
def test_formatting_marks_the_match_and_labels_context():
    from models.documents import DocumentChunk, SearchResult
    from utils.qa_agent import QAAgent

    match = DocumentChunk(chunk_id="a#c1", file_name="a.pdf", page_number=6, text="the match")
    before = DocumentChunk(chunk_id="a#c0", file_name="a.pdf", page_number=5, text="table rows")
    result = SearchResult(chunk=match, score=1.0, context_before=[before])
    agent = QAAgent(MagicMock(), client=MagicMock())

    body = agent._format_results([result])

    assert ">>> MATCH: the match" in body
    assert "...context, p. 5: table rows" in body
    assert "Source: a.pdf, page 6" in body


@pytest.mark.unit
def test_context_chunk_already_shown_as_a_match_is_not_repeated_as_context():
    """A chunk already sent to the model as a full, cited match this
    answer() call shouldn't also be echoed as someone else's uncited
    context line -- that's the same text twice for no new information."""
    match_a = DocumentChunk(chunk_id="a#c0", file_name="a.pdf", page_number=5, text="chunk zero")
    match_b = DocumentChunk(chunk_id="a#c1", file_name="a.pdf", page_number=6, text="chunk one")
    agent = QAAgent(MagicMock(), client=MagicMock())
    # _run_tools records a match's own id in _seen_chunk_ids before ever
    # calling _format_results; seed that directly rather than going through
    # _run_tools, since _format_results's dedup behaviour is what's under test.
    agent._seen_chunk_ids.add(match_a.chunk_id)

    # chunk zero was already sent as its own cited match; it now shows up
    # again as chunk one's context_before in a later search.
    body = agent._format_results(
        [SearchResult(chunk=match_b, score=1.0, context_before=[match_a])]
    )

    assert "chunk zero" not in body
    assert ">>> MATCH: chunk one" in body


@pytest.mark.unit
def test_context_chunk_already_shown_as_context_is_not_repeated():
    """The same neighbour chunk surfacing as context for two different
    matches, whether in one search call or two, is shown only once."""
    match_a = DocumentChunk(chunk_id="a#c0", file_name="a.pdf", page_number=5, text="match a")
    match_b = DocumentChunk(chunk_id="a#c2", file_name="a.pdf", page_number=7, text="match b")
    shared_neighbour = DocumentChunk(
        chunk_id="a#c1", file_name="a.pdf", page_number=6, text="shared neighbour"
    )
    agent = QAAgent(MagicMock(), client=MagicMock())

    body = agent._format_results(
        [
            SearchResult(chunk=match_a, score=2.0, context_after=[shared_neighbour]),
            SearchResult(chunk=match_b, score=1.0, context_before=[shared_neighbour]),
        ]
    )

    assert body.count("shared neighbour") == 1
