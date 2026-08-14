"""Extraction by looking at the page instead of parsing the PDF.

The pdf_inspector/Docling path reads the PDF's own structure, which is only
as good as the producing tool made it: running headers arrive as section
headings, two-column reading order can interleave, and table geometry is
inferred. This extractor renders each page to an image (utils.page_renderer)
and has Claude read it, so what lands in a chunk is what a person would see.

The trade is cost and latency: one model call per page rather than none, so
this is a deliberate choice by the caller, not a fallback the ingest picks up
on its own. The model is settings.claude_vision_model rather than the
claude_model the rest of the repo uses, because the only thing that matters
here is reading small type off a render, and that is bounded by the model's
image resolution ceiling rather than by its reasoning.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from models.config import settings
from models.documents import DocumentChunk
from utils.page_renderer import PageRenderer, RenderedPage
from utils.pdf_extractor import PdfExtractor

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "page_transcription.prompt"

# A dense two-column journal page transcribes to well under this; the
# headroom is for full-page tables, which are the longest thing a page can be.
PAGE_MAX_TOKENS = 8000

# Transcription is recall, not reasoning: the page is in front of the model.
# Thinking stays on (rather than disabled) because with thinking off the
# model is prone to leaking stray tags into the visible response, and a
# transcript is exactly where that would be invisible until it hit the index.
# settings.claude_vision_model is on adaptive thinking by default and accepts
# the full effort ladder, so no thinking parameter is passed and "low" is
# valid; a model outside that ladder would reject this.
PAGE_EFFORT = "low"


class VisionExtractor:
    """Transcribes PDF pages with Claude's vision, one model call per page.

    Produces the same DocumentChunk objects as the other extractors, so it is
    a drop-in alternative in the ingest: chunk_id, file name, and 1-indexed
    page number all carry the same meaning, and the Markdown headings Claude
    emits become chunk headings exactly as they do for pdf_inspector output.
    """

    def __init__(
        self,
        image_dir: Path | str,
        client: anthropic.Anthropic | None = None,
        renderer: PageRenderer | None = None,
        max_chars: int = 3000,
        max_workers: int = 4,
    ):
        self._client = client or anthropic.Anthropic(api_key=settings.claude_api_key)
        self._renderer = renderer or PageRenderer(image_dir)
        self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        self.max_chars = max_chars
        self.max_workers = max_workers
        # 1-indexed pages of the most recent extract_chunks call that produced
        # no text. Callers should surface these: a silently short document is
        # worse than a loud failure.
        self.failed_pages: list[int] = []

    def extract_chunks(
        self, pdf_path: Path | str, pages: list[int] | None = None
    ) -> list[DocumentChunk]:
        """Render and transcribe the given 1-indexed pages (all when None).

        Pages are transcribed concurrently because each is an independent
        request, but the result is re-sorted by page so chunk order and
        chunk_id numbering match a sequential run.
        """
        file_name = Path(pdf_path).name
        rendered = self._renderer.render(pdf_path, pages)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            markdowns = list(pool.map(self._transcribe_or_skip, rendered))

        self.failed_pages = [
            page.page_number for page, text in zip(rendered, markdowns) if text is None
        ]
        markdowns = ["" if text is None else text for text in markdowns]

        chunks: list[DocumentChunk] = []
        for page, markdown in zip(rendered, markdowns):
            if not markdown.strip():
                continue
            chunks.extend(
                PdfExtractor._page_to_chunks(
                    file_name, page.page_number, markdown, self.max_chars
                )
            )

        # _page_to_chunks numbers chunks within a page, so ids collide across
        # pages; renumber over the document the way DocumentExtractor does.
        return [
            chunk.model_copy(update={"chunk_id": f"{file_name}#p{chunk.page_number}#c{index}"})
            for index, chunk in enumerate(chunks)
        ]

    def _transcribe_or_skip(self, page: RenderedPage) -> str | None:
        """Isolate per-page failures to that page.

        A single page can fail for reasons that say nothing about the rest of
        the document: a transient API error, or output blocked by a content
        filter. Letting that exception escape the pool loses every page of the
        document, including the ones that transcribed fine, so failures are
        recorded and reported instead."""
        try:
            return self.transcribe_page(page)
        except Exception as exc:
            print(f"  page {page.page_number} FAILED: {exc}")
            return None

    def transcribe_page(self, page: RenderedPage) -> str:
        """Markdown transcription of one rendered page."""
        image_data = base64.standard_b64encode(page.read_bytes()).decode("utf-8")

        response = self._client.messages.create(
            model=settings.claude_vision_model,
            max_tokens=PAGE_MAX_TOKENS,
            system=self._system_prompt,
            output_config={"effort": PAGE_EFFORT},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": page.media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": "Transcribe this page."},
                    ],
                }
            ],
        )

        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text").strip()
