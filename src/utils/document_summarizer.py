"""Document-level summaries: one Claude call per source file, not per chunk.

Complements utils/metadata_enhancer.py rather than replacing it. A chunk
summary is an embedding target for text that embeds badly on its own; a
document summary is routing information, shown alongside every result from
its file so the answering model can tell whether it is in the right document
and whether the extract is worth escalating to read_source.

Deliberately separate from the extractors and behind an explicit flag
(`ingest --summarise-documents`), for the same reason MetadataEnhancer is:
it costs real API calls, so a caller asks for it rather than getting it as a
side effect of extraction.
"""

from pathlib import Path

import anthropic

from models.config import settings
from models.documents import DocumentChunk, DocumentSummary

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "document_summary.prompt"

# Three or four sentences of scope and inventory. Larger than the per-chunk
# budget because a profile names sections and coverage, not one idea.
SUMMARY_MAX_TOKENS = 500

# The model is shown the front of the document, not all of it: the opening
# pages carry the title, author and scope statement, and the heading list
# carries the structure. Sending 345 chunks would cost more than the summary
# is worth and would tempt the model into summarising findings.
MAX_OPENING_CHARS = 6000
MAX_HEADINGS = 60


class DocumentSummarizer:
    """Writes a DocumentSummary for each source file in a chunk list.

    Input is the chunks themselves rather than the PDF, so this runs after
    extraction and needs no parser, and works identically for the fast and
    vision paths.
    """

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client or anthropic.Anthropic(api_key=settings.claude_api_key)
        self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def summarize(self, chunks: list[DocumentChunk]) -> str:
        """One file's chunks in, the profile text out. Returns "" for no chunks."""
        if not chunks:
            return ""

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=SUMMARY_MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": self._build_profile(chunks)}],
        )
        # A refusal returns HTTP 200 with no content blocks, so this degrades to
        # an empty summary; the document is still indexed, it just has no profile.
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def summarize_documents(self, chunks: list[DocumentChunk]) -> list[DocumentSummary]:
        """Groups a mixed chunk list by file and summarises each file once.

        File order follows first appearance in `chunks`, matching the order
        `ingest` extracted them in. Files whose summary comes back empty are
        skipped rather than stored blank, so a refusal on one document does
        not put an empty profile in front of the answering model.
        """
        by_file: dict[str, list[DocumentChunk]] = {}
        for chunk in chunks:
            by_file.setdefault(chunk.file_name, []).append(chunk)

        summaries = []
        for file_name, file_chunks in by_file.items():
            summary = self.summarize(file_chunks)
            if not summary:
                continue
            summaries.append(
                DocumentSummary(
                    file_name=file_name,
                    summary=summary,
                    page_count=max(chunk.page_number for chunk in file_chunks),
                )
            )
        return summaries

    @staticmethod
    def _build_profile(chunks: list[DocumentChunk]) -> str:
        """Front matter, structure and content mix, in one prompt-sized string."""
        ordered = sorted(chunks, key=lambda chunk: chunk.page_number)

        headings: list[str] = []
        for chunk in ordered:
            crumb = " > ".join(chunk.headings)
            if crumb and crumb not in headings:
                headings.append(crumb)

        opening = ""
        for chunk in ordered:
            if len(opening) >= MAX_OPENING_CHARS:
                break
            opening += chunk.text + "\n\n"

        kinds = {kind: 0 for kind in ("table", "figure")}
        for chunk in ordered:
            if chunk.kind in kinds:
                kinds[chunk.kind] += 1

        parts = [
            f"File name: {ordered[0].file_name}",
            f"Pages: {ordered[-1].page_number}",
            f"Extracted chunks: {len(ordered)}, of which "
            f"{kinds['table']} are tables and {kinds['figure']} are figures",
        ]
        if headings:
            listed = headings[:MAX_HEADINGS]
            body = "\n".join(f"- {heading}" for heading in listed)
            if len(headings) > MAX_HEADINGS:
                body += f"\n- ... and {len(headings) - MAX_HEADINGS} more"
            parts.append(f"Section headings, in order:\n{body}")
        parts.append(f"Opening pages:\n{opening[:MAX_OPENING_CHARS].strip()}")

        return "\n\n".join(parts)
