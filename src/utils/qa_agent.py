import logging
from pathlib import Path

import anthropic

from models.config import settings
from models.documents import SearchResult

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "qa_system.prompt"

REFUSAL_MESSAGE = "The model declined to answer this question."
SEARCH_BUDGET_MESSAGE = "I could not settle on an answer within the allotted search attempts."
DUPLICATE_RESULTS_MESSAGE = (
    "All matching extracts for that query were already returned by an earlier "
    "search in this conversation; see those results above."
)

# Round trips of tool calls allowed per question, so a model stuck rephrasing
# the same query cannot loop forever. Higher than the search-only loop needed
# because reading a source costs an extra round trip on top of the search that
# found it.
MAX_TOOL_ITERATIONS = 8

SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the document knowledge base and return the top-matching extracts "
        "with their source file name and page number. Rewrite the question into "
        "the keywords most likely to appear in the source text -- not the question "
        "itself. If a search doesn't surface what you need, call this tool again "
        "with different or narrower keywords rather than giving up after one try."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to search for, not a full sentence.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return.",
            },
        },
        "required": ["query"],
    },
}

READ_SOURCE_TOOL = {
    "name": "read_source",
    "description": (
        "Read a source document in full, or one page of it, rather than the "
        "keyword-matched extracts search returns. Use this when a search result "
        "looks decisive but incomplete: the extract is cut off mid-sentence, it "
        "shows table rows without the header that names the columns, it refers to "
        "a figure or a footnote whose content is not in the extract, or the "
        "question needs several facts that are spread across one document. Give "
        "the file name exactly as the search results report it. Omit page to read "
        "the whole document, which is the right choice when the answer depends on "
        "comparing distant parts of the same paper."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string",
                "description": "File name exactly as reported by the search tool.",
            },
            "page": {
                "type": "integer",
                "description": "1-indexed page. Omit to read the whole document.",
            },
        },
        "required": ["file_name"],
    },
}


class QAAgent:
    """Answers questions from the knowledge base with an agentic retrieval loop:
    Claude chooses the search keywords, calls the search tool -- possibly more
    than once with refined terms -- and cites file/page in its final answer.

    The tool implementation is whatever `index.search(query, top_k)` returns
    (see PostgresSearchIndex); swapping in a different retriever means
    changing only what `index` is, not this loop."""

    def __init__(self, index, client: anthropic.Anthropic | None = None, model: str | None = None):
        self._index = index
        self._client = client or anthropic.Anthropic(api_key=settings.claude_api_key)
        self._model = model or settings.claude_model
        self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        # Cumulative Claude usage for the most recent answer() call. Output
        # tokens include any thinking tokens the model spent, since the API
        # reports them as one combined figure rather than a separate field.
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Every chunk id already sent to the model this answer() call,
        # whether as a ranked match or as another match's neighbouring
        # context, so nothing already shown once -- as a duplicate match or
        # as redundant context -- is repeated for the rest of this question.
        self._seen_chunk_ids: set[str] = set()

    def answer(self, question: str, top_k: int | None = None) -> str:
        default_top_k = top_k or settings.top_k
        messages: list[dict] = [{"role": "user", "content": question}]
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._seen_chunk_ids = set()
        logger.debug("question: %r", question)

        for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
            logger.debug(
                "round %d/%d: calling %s", iteration, MAX_TOOL_ITERATIONS, self._model
            )
            response = self._client.messages.create(
                model=self._model,
                max_tokens=settings.claude_max_tokens,
                system=self._system_prompt,
                tools=[SEARCH_TOOL, READ_SOURCE_TOOL],
                messages=messages,
            )
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            logger.debug(
                "round %d: stop_reason=%s input_tokens=%d output_tokens=%d",
                iteration,
                response.stop_reason,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            if response.stop_reason == "refusal":
                return REFUSAL_MESSAGE
            if response.stop_reason != "tool_use":
                return self._final_text(response)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": self._run_tools(response, default_top_k)})

        return SEARCH_BUDGET_MESSAGE

    def _run_tools(self, response, default_top_k: int) -> list[dict]:
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == READ_SOURCE_TOOL["name"]:
                file_name, page = block.input.get("file_name", ""), block.input.get("page")
                logger.debug("read_source: file_name=%r page=%s", file_name, page)
                content = self._read_source(file_name, page)
            else:
                query = block.input.get("query", "")
                query_top_k = block.input.get("top_k") or default_top_k
                logger.debug("search_knowledge_base: query=%r top_k=%d", query, query_top_k)
                results = self._index.search(query, query_top_k)
                if not results:
                    content = self._no_results_message(query)
                else:
                    fresh = [r for r in results if r.chunk.chunk_id not in self._seen_chunk_ids]
                    self._seen_chunk_ids.update(r.chunk.chunk_id for r in fresh)
                    logger.debug(
                        "search_knowledge_base: %d result(s), %d new after dedup, for query=%r",
                        len(results),
                        len(fresh),
                        query,
                    )
                    content = (
                        self._format_results(fresh) if fresh else DUPLICATE_RESULTS_MESSAGE
                    )
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content}
            )
        return tool_results

    def _read_source(self, file_name: str, page: int | None) -> str:
        """Whole-document or whole-page text for the escalation tool.

        Requires an index exposing document()/page()/files(); a retriever that
        only implements search() degrades to a message rather than an error, so
        the agent falls back to searching instead of failing the question."""
        if not all(hasattr(self._index, name) for name in ("document", "page", "files")):
            return "This knowledge base does not support reading whole sources; use search instead."

        chunks = (
            self._index.page(file_name, page)
            if page is not None
            else self._index.document(file_name)
        )
        if not chunks:
            available = ", ".join(self._index.files())
            return (
                f"No content found for {file_name}"
                + (f" page {page}." if page is not None else ".")
                + f" Indexed files are: {available}"
            )

        scope = f"page {page}" if page is not None else f"all {len(chunks)} extracts"
        body = "\n\n".join(f"[p. {chunk.page_number}] {chunk.text}" for chunk in chunks)
        return f"Source: {file_name} ({scope})\n\n{body}"

    @staticmethod
    def _final_text(response) -> str:
        return "\n".join(block.text for block in response.content if block.type == "text")

    def _no_results_message(self, query: str) -> str:
        """Empty chunk search, plus whatever the document lane can offer.

        A backend without search_documents degrades to the bare message, the
        same check-then-degrade shape read_source uses for document()/page()."""
        base = "No matching extracts found for that query."
        if not hasattr(self._index, "search_documents"):
            return base

        documents = self._index.search_documents(query)
        if not documents:
            return base

        listed = "\n".join(
            f"- {doc.file_name} ({doc.page_count} pages): {doc.summary}" for doc in documents
        )
        return (
            f"{base} These documents are about the closest subject match, judged "
            f"on their profiles rather than on any extract. Nothing here is "
            f"quotable or citable; use read_source on one of them, or search "
            f"again with different keywords.\n\n{listed}"
        )

    def _format_results(self, results: list[SearchResult]) -> str:
        """Render matches with their surrounding chunks marked as context.

        The match is delimited rather than merged into the neighbours so the
        model can tell what actually matched the query from what is only there
        to complete a sentence or restore a table header, and so it cites the
        page the match came from rather than a neighbour's page.

        Document profiles are printed once each in a preamble rather than on
        every result, so five hits in one file do not repeat the same paragraph
        five times, and so the ranked list itself stays in score order.

        A context chunk already sent to the model this answer() call -- as an
        earlier match, or as another result's context -- is dropped rather
        than repeated: the model has already read that text, so resending it
        is pure token overhead with nothing new for the model to learn from
        it. Only context lines are dropped this way; the match itself is
        never dropped here, since _run_tools has already filtered `results`
        down to matches not seen before calling this."""
        if not results:
            return "No matching extracts found for that query."

        profiles = {}
        for result in results:
            if result.doc_summary and result.chunk.file_name not in profiles:
                profiles[result.chunk.file_name] = result.doc_summary

        blocks = []
        if profiles:
            listed = "\n".join(f"- {name}: {summary}" for name, summary in profiles.items())
            blocks.append(
                "The extracts below come from these documents. The descriptions "
                "are background for judging fit and are not quotable or citable:\n"
                f"{listed}"
            )
        for i, result in enumerate(results, start=1):
            chunk = result.chunk
            parts = [f"[{i}] Source: {chunk.file_name}, page {chunk.page_number}"]
            for before in result.context_before:
                if before.chunk_id in self._seen_chunk_ids:
                    continue
                parts.append(f"...context, p. {before.page_number}: {before.text}")
                self._seen_chunk_ids.add(before.chunk_id)
            parts.append(f">>> MATCH: {chunk.text}")
            for after in result.context_after:
                if after.chunk_id in self._seen_chunk_ids:
                    continue
                parts.append(f"...context, p. {after.page_number}: {after.text}")
                self._seen_chunk_ids.add(after.chunk_id)
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)
