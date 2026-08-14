from collections.abc import Collection
from pathlib import Path

import anthropic

from models.config import settings
from models.documents import DocumentChunk

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "chunk_summary.prompt"

# A summary is a sentence or two, but adaptive thinking's own tokens count
# against this same budget, and on dense or noisy content (garbled OCR text,
# in the real case that surfaced this) thinking alone can consume most or all
# of a tight budget before any answer is written -- see summarize()'s
# stop_reason check below. 500 leaves real headroom for that.
SUMMARY_MAX_TOKENS = 500

# Which chunk kinds are worth a summary. Tables only, because a table is the
# one kind whose summary anything actually reads: index_entry_builder uses it
# as the table's embed_text, falling back to truncated Markdown when absent.
# Text chunks embed their own prose, and a figure chunk's text is already an
# LLM-written description, so summarising either buys nothing and costs a call
# per chunk. On the bundled fixtures that is 36 calls rather than 384.
DEFAULT_SUMMARISED_KINDS = ("table",)


class MetadataEnhancer:
    """Adds an LLM-written summary to a DocumentChunk, one Claude call per
    chunk. Meant for chunks whose raw text is a poor embedding target on its
    own -- tables especially, where the exact figures matter for keyword
    search but a sentence describing them is what should actually be
    embedded.

    Deliberately separate from the extractors: this costs a real API call
    per chunk, so callers decide when to run it rather than it happening as
    a side effect of extraction."""

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client or anthropic.Anthropic(api_key=settings.claude_api_key)
        self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        # chunk_ids from the most recent enhance_chunks() call whose summary
        # was discarded rather than stored, because there was real text to
        # summarise but no complete answer came back. Not the same as a
        # chunk that legitimately has nothing to say (a refusal on truly
        # unreadable content, say) -- this is for the case that surfaced it:
        # a truncated answer that reads as a complete sentence but isn't.
        self.failed_chunks: list[str] = []

    def summarize(self, text: str, *, context: str = "", headings: list[str] | None = None) -> str:
        parts = []
        if headings:
            parts.append("Section: " + " > ".join(headings))
        if context:
            parts.append("Preceding chunk, for context only -- do not summarise this part:\n" + context)
        content = "\n\n".join([*parts, f"Chunk to summarise:\n{text}"]) if parts else text

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=SUMMARY_MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        # Adaptive thinking's own tokens count against max_tokens, and on
        # dense or noisy content it can consume the whole budget before any
        # answer is written, or cut the answer off mid-sentence. Either way
        # the visible text is not a real summary, and a truncated sentence
        # that reads as complete is worse to store than nothing at all.
        if response.stop_reason == "max_tokens":
            return ""

        return "".join(block.text for block in response.content if block.type == "text").strip()

    def enhance_chunks(
        self,
        chunks: list[DocumentChunk],
        kinds: Collection[str] = DEFAULT_SUMMARISED_KINDS,
    ) -> list[DocumentChunk]:
        """Returns new DocumentChunks with `summary` populated on those whose
        `kind` is in `kinds`. Chunks of any other kind are passed through
        untouched, and the input chunks are left unmodified either way. Pass
        `kinds=("text", "table", "figure")` to summarise everything.

        Each summarised chunk is given the immediately preceding chunk's text
        as context, when it belongs to the same file: a table's introducing
        sentence often lands in the chunk before it rather than its own.
        `previous` deliberately tracks the real predecessor from the full
        list, not the last chunk that was summarised, so filtering by kind
        cannot hand a table the context of some earlier, unrelated table."""
        result = []
        failed_chunks = []
        previous: DocumentChunk | None = None
        for chunk in chunks:
            if chunk.kind in kinds:
                context = (
                    previous.text if previous and previous.file_name == chunk.file_name else ""
                )
                summary = self.summarize(chunk.text, context=context, headings=chunk.headings)
                if not summary and chunk.text.strip():
                    failed_chunks.append(chunk.chunk_id)
                result.append(chunk.model_copy(update={"summary": summary}))
            else:
                result.append(chunk)
            previous = chunk
        self.failed_chunks = failed_chunks
        return result
