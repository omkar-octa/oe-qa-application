"""Rasterises PDF pages to PNG images for the vision extraction path.

pdf_inspector and Docling both work from the PDF's internal structure, so
what they recover is only as good as that structure: running headers get
mistaken for section headings, multi-column reading order can interleave,
and a table's visual layout is inferred rather than seen. Rendering the page
and letting a vision model read it sidesteps all of that at the cost of one
model call per page (see utils.vision_extractor).

pypdfium2 is used rather than a system poppler install: it ships as a wheel
with no external binary, which matters on Windows where pdftoppm is rarely
on PATH.
"""

from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

# Claude reads images up to 2576px on the long edge at full fidelity and
# downsamples anything larger, so rendering beyond this only costs time.
MAX_LONG_EDGE_PX = 2576

# Below roughly this width, 8-9pt body text in a two-column journal article
# starts to lose strokes and transcription accuracy drops sharply.
MIN_LONG_EDGE_PX = 1400


@dataclass(frozen=True)
class RenderedPage:
    """One rasterised page. page_number is 1-indexed, matching DocumentChunk."""

    page_number: int
    image_path: Path
    media_type: str = "image/png"

    def read_bytes(self) -> bytes:
        return self.image_path.read_bytes()


class PageRenderer:
    """Renders PDF pages to PNG files on disk.

    Images are written to disk rather than held in memory so a long document
    doesn't have to be fully rasterised before extraction starts, and so a
    failed or interrupted run can be re-inspected page by page.
    """

    def __init__(self, output_dir: Path | str, long_edge_px: int = MAX_LONG_EDGE_PX):
        self.output_dir = Path(output_dir)
        self.long_edge_px = max(MIN_LONG_EDGE_PX, min(long_edge_px, MAX_LONG_EDGE_PX))

    def page_count(self, pdf_path: Path | str) -> int:
        document = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(document)
        finally:
            document.close()

    def render(
        self, pdf_path: Path | str, pages: list[int] | None = None
    ) -> list[RenderedPage]:
        """Render the given 1-indexed pages (all pages when None).

        Existing files are reused: rasterising is deterministic and the slow
        part of a re-run, so a repeated ingest of the same corpus skips it.
        """
        pdf_path = Path(pdf_path)
        page_dir = self.output_dir / pdf_path.stem
        page_dir.mkdir(parents=True, exist_ok=True)

        document = pdfium.PdfDocument(str(pdf_path))
        try:
            wanted = pages if pages is not None else range(1, len(document) + 1)
            rendered: list[RenderedPage] = []
            for page_number in wanted:
                image_path = page_dir / f"p{page_number:04d}.png"
                if not image_path.exists():
                    self._render_page(document, page_number, image_path)
                rendered.append(RenderedPage(page_number=page_number, image_path=image_path))
            return rendered
        finally:
            document.close()

    def _render_page(self, document, page_number: int, image_path: Path) -> None:
        page = document[page_number - 1]
        try:
            # pypdfium2 renders at 72 DPI * scale; scale from the page's own
            # long edge so a wide landscape figure page and a portrait text
            # page both land at the same target resolution.
            long_edge_pt = max(page.get_width(), page.get_height())
            scale = self.long_edge_px / long_edge_pt
            page.render(scale=scale).to_pil().save(image_path)
        finally:
            page.close()
