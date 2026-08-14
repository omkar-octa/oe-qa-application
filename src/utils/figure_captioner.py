"""Turns the figures on a page into their own searchable chunks.

A value that exists only as a label inside a chart or a map is unreachable by
every other path in this pipeline. pdf_inspector and Docling skip picture
items entirely, and while the vision extractor's transcription prompt does ask
for a figure description, that description lands as loose prose inside a
3000-character page transcript, sharing a chunk with whatever body text
happened to sit next to it. Nothing cites it, and keyword search scores it
against the whole page rather than the figure.

This produces one chunk per figure instead, carrying the caption and a factual
description of what the figure shows, so a figure ranks and cites like any
other chunk.

It also recovers text that the layout-detection stage of extraction sometimes
drops: Docling's layout model can box a whole page of real body text as one
giant misclassified "picture" region (observed on a real fixture, see
docs/roadmap.md), and neither extractor's picture-skipping logic ever sees
what was inside it. The model is asked to report any such prose as its own
item (classification "text"), which _to_chunk turns into an ordinary
kind="text" chunk rather than a figure caption, so it merges into search like
any other extracted text.

Detection is done by the model on the whole page image rather than by cropping,
which keeps this independent of both extractors: the page PNGs come from
PageRenderer, which already caches them, so this works the same behind
--extractor fast and --extractor vision.

Costs one model call per page, so it runs by default on ingest's
likely_figure_pages() candidates (pages with a plausible reason to hold an
image) rather than every page; --no-caption-figures turns it off entirely for
a cost-sensitive run.
"""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pdf_inspector
import pypdfium2 as pdfium

from models.config import settings
from models.documents import DocumentChunk
from utils.page_renderer import PageRenderer, RenderedPage

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "figure_description.prompt"

# Structure-tree roles that mark a figure or its caption in a *tagged* PDF.
# Untagged PDFs -- several of the bundled fixtures among them -- return none
# of these, which is why likely_figure_pages() unions this with the pypdfium2
# check below rather than relying on either signal alone.
_FIGURE_STRUCTURE_ROLES = frozenset({"Figure", "Caption"})

# Shallow on purpose: a figure's image object is typically at most one Form
# XObject deep (a border or grouping wrapper), and bounding the walk keeps
# this a cheap pre-filter rather than a second layout pass.
_IMAGE_OBJECT_SCAN_DEPTH = 2


def likely_figure_pages(pdf_path: Path | str) -> list[int]:
    """1-indexed pages worth sending to describe_page, cheaply, before paying
    for a Claude vision call on every page of every document.

    Unions two independent, local signals -- there is nothing closer to real
    figure detection available without running Docling's layout model over
    every page, which is the cost this whole two-tier pipeline exists to
    avoid paying except where OCR is actually needed:

    - the PDF's own accessibility structure tree, when tagged: a "Figure" or
      "Caption" role at a page is about as direct a signal as exists, via
      pdf_inspector.extract_structure_elements(). Untagged PDFs return an
      empty list here, silently, rather than an error.
    - embedded raster image objects, via pypdfium2's page object scan. This
      catches untagged PDFs the structure tree says nothing about, but it
      only sees embedded bitmaps: a chart drawn as vector paths (common for
      matplotlib/matlab exports) has no image object and is invisible here.

    Because each signal has a real blind spot, this unions rather than
    intersects them, and still leans on describe_page's own furniture
    classification (logo/signature) to reject false positives such as a
    masthead image that happens to share a page with real content.
    """
    tagged_pages = {
        element.page
        for element in pdf_inspector.extract_structure_elements(str(pdf_path))
        if element.role in _FIGURE_STRUCTURE_ROLES
    }

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        image_pages: set[int] = set()
        for page_number in range(1, len(document) + 1):
            page = document[page_number - 1]
            try:
                has_image = any(
                    obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE
                    for obj in page.get_objects(max_depth=_IMAGE_OBJECT_SCAN_DEPTH)
                )
            finally:
                page.close()
            if has_image:
                image_pages.add(page_number)
    finally:
        document.close()

    return sorted(tagged_pages | image_pages)

# The worst case is a page of multi-panel figures where every panel carries a
# label for every region, which is what the prompt asks for and what makes the
# output useful: Fig. 5 of the Calvillo fixture is six panels x eleven regions
# x three sectors and runs past 4000 tokens on its own. Structured output means
# hitting this ceiling truncates the JSON rather than the prose, so the page is
# lost outright rather than quietly shortened (see _parse_response) -- hence
# the generous headroom. Only tokens actually generated are billed.
PAGE_MAX_TOKENS = 16000

# Logos and signatures are page furniture, not content. Mirrors
# FigureElement.is_furniture in models/elements.py rather than inventing a
# second rule for the same idea.
_FURNITURE = frozenset({"logo", "signature"})

# "text" is not a figure at all -- it marks recovered body text a layout model
# misclassified as a picture. _to_chunk gives it kind="text" instead of the
# kind="figure" every other classification gets, see there.
_CLASSIFICATIONS = ["chart", "diagram", "map", "photo", "logo", "signature", "text", "other"]

_RECOVERED_TEXT = "text"

# Constrains the response shape, which matters for more than parsing: the
# cheap models this runs on have no adaptive thinking, and a schema-bound
# response cannot leak stray tags into the text the way a free-form one can.
_FIGURE_SCHEMA = {
    "type": "object",
    "properties": {
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "description": {"type": "string"},
                    "classification": {"type": "string", "enum": _CLASSIFICATIONS},
                },
                "required": ["caption", "description", "classification"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["figures"],
    "additionalProperties": False,
}


class FigureCaptioner:
    """Describes each figure on a page and emits it as a DocumentChunk.

    Produces `kind="figure"` chunks with the same chunk_id, file name and
    1-indexed page number conventions as every other extractor, so they merge
    into a document's chunk list and are indexed and cited unchanged.
    """

    def __init__(
        self,
        image_dir: Path | str,
        client: anthropic.Anthropic | None = None,
        renderer: PageRenderer | None = None,
        max_workers: int = 4,
    ):
        self._client = client or anthropic.Anthropic(api_key=settings.claude_api_key)
        self._renderer = renderer or PageRenderer(image_dir)
        self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        self.max_workers = max_workers
        # 1-indexed pages of the most recent caption_document call that raised.
        # A page with genuinely no figures is not a failure and is not listed
        # here; callers should surface this list, since a page that failed and
        # a page with no figures are indistinguishable in the output otherwise.
        self.failed_pages: list[int] = []

    def caption_document(
        self, pdf_path: Path | str, pages: list[int] | None = None
    ) -> list[DocumentChunk]:
        """Figure chunks for the given 1-indexed pages (all pages when None).

        Pages are described concurrently because each is an independent
        request, but the result is re-sorted by page so chunk order matches a
        sequential run and matches the document's reading order.
        """
        file_name = Path(pdf_path).name
        rendered = self._renderer.render(pdf_path, pages)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            per_page = list(pool.map(self._describe_or_skip, rendered))

        self.failed_pages = [
            page.page_number for page, figures in zip(rendered, per_page) if figures is None
        ]

        chunks: list[DocumentChunk] = []
        for page, figures in zip(rendered, per_page):
            for index, figure in enumerate(figures or []):
                chunks.append(self._to_chunk(file_name, page.page_number, index, figure))
        return chunks

    def _to_chunk(
        self, file_name: str, page_number: int, index: int, figure: dict
    ) -> DocumentChunk:
        caption = figure["caption"].strip()
        description = figure["description"].strip()

        if figure["classification"] == _RECOVERED_TEXT:
            # Recovered prose, not a figure: description already holds the
            # verbatim transcription, so it becomes the chunk's text directly
            # rather than being wrapped into a caption+description pair. This
            # is what lets it merge into search and citations exactly like any
            # other extracted text chunk instead of reading as a figure.
            return DocumentChunk(
                chunk_id=f"{file_name}#p{page_number}#f{index}",
                file_name=file_name,
                page_number=page_number,
                text=description,
                headings=[caption] if caption else [],
                kind="text",
            )

        return DocumentChunk(
            # The #f discriminator keeps these from colliding with the #c text
            # chunks the extractors number over the same pages.
            chunk_id=f"{file_name}#p{page_number}#f{index}",
            file_name=file_name,
            page_number=page_number,
            # Caption and description both live in text because text is the
            # field retrieval actually searches: index_entry_builder copies it
            # straight into keyword_text and embed_text, and QAAgent shows the
            # model the same field. A description parked in `summary` would
            # be invisible to search.
            text=f"{caption}\n\n{description}",
            # Carried through as heading_path for display and citation
            # context. PostgresSearchIndex does not fold heading_path into
            # keyword_text or embed_text, so this does not currently weight
            # ranking the way a repeated section title would.
            headings=[caption] if caption else [],
            kind="figure",
        )

    def _describe_or_skip(self, page: RenderedPage) -> list[dict] | None:
        """Isolate per-page failures to that page.

        A single page can fail for reasons that say nothing about the rest of
        the document: a transient API error, or a response that did not parse.
        Letting that exception escape the pool loses every page's figures,
        including the ones that described fine. Returns None on failure, which
        is distinct from the empty list a page with no figures returns.
        """
        try:
            return self.describe_page(page)
        except Exception as exc:
            print(f"  page {page.page_number} figure captioning FAILED: {exc}")
            return None

    def describe_page(self, page: RenderedPage) -> list[dict]:
        """Every non-furniture figure on one rendered page."""
        image_data = base64.standard_b64encode(page.read_bytes()).decode("utf-8")

        # No output_config.effort and no thinking parameter, so that
        # claude_figure_model can be pointed at a cheaper model without the
        # call shape becoming invalid: models below the current default reject
        # effort outright and have no adaptive thinking to configure. Do not
        # copy the call shape from vision_extractor.py, which does pass effort.
        response = self._client.messages.create(
            model=settings.claude_figure_model,
            max_tokens=PAGE_MAX_TOKENS,
            system=self._system_prompt,
            output_config={"format": {"type": "json_schema", "schema": _FIGURE_SCHEMA}},
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
                        {"type": "text", "text": "Describe the figures on this page."},
                    ],
                }
            ],
        )
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response) -> list[dict]:
        # A refusal is HTTP 200 with empty content, so this has to be checked
        # before reading content at all. Degrade this page to no figures
        # rather than failing the document.
        if response.stop_reason == "refusal":
            return []

        # Under structured output the response is valid JSON or nothing, so a
        # truncated response is unparseable rather than subtly short. Raising
        # routes it through _describe_or_skip and onto failed_pages.
        if response.stop_reason == "max_tokens":
            raise ValueError(f"figure JSON truncated at max_tokens ({PAGE_MAX_TOKENS})")

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            return []

        figures = json.loads(text).get("figures", [])
        return [
            figure
            for figure in figures
            if figure.get("classification") not in _FURNITURE
            and figure.get("description", "").strip()
        ]
