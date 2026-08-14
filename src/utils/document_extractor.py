import os
from pathlib import Path

# RapidOCR's torch backend tries to JIT-compile via torch.compile/inductor,
# which needs an MSVC C++ compiler. Many Windows machines don't have one on
# PATH; running eager instead avoids a hard failure with no functional loss
# beyond a slightly slower first OCR call. Must be set before docling (and
# therefore torch) is imported below.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, PictureItem, SectionHeaderItem, TableItem, TextItem

from models.documents import DocumentChunk
from utils.pdf_extractor import PdfExtractor

# A crop this small relative to its page is furniture (a publisher logo, an
# ORCID icon, a small badge), not a content figure. Verified against a real
# fixture: a masthead-heavy page returned seven such crops, all between
# 10x10 and 70x73px on a ~1200x1600px render, none of them a real figure,
# while a genuine chart on another page of the same document filled well
# over half the page. 2% leaves a wide margin between the two.
_MIN_PICTURE_AREA_FRACTION = 0.02


def _pages_to_ranges(pages: list[int]) -> list[tuple[int, int]]:
    """Collapse a list of 1-indexed page numbers into contiguous (start, end)
    ranges, so Docling converts each run of consecutive pages in one pass
    rather than the whole document."""
    ranges: list[tuple[int, int]] = []
    for page in sorted(pages):
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], page)
        else:
            ranges.append((page, page))
    return ranges


def extract_figure_images(
    pdf_path: Path, pages: list[int], output_dir: Path | str
) -> dict[int, list[Path]]:
    """Crops every real figure Docling's layout model finds on the given
    1-indexed pages into PNG files on disk, via PictureItem.get_image().

    Meant to run only on utils.figure_captioner.likely_figure_pages()'s
    candidates, not every page: the layout model runs on every page this is
    given, which is the cost the two-tier pipeline exists to avoid paying
    where it doesn't have to.

    A dedicated converter, not DocumentExtractor's: this one turns on
    generate_page_images (a prerequisite for get_image() to have anything to
    crop from) and turns off do_ocr/do_table_structure, since PictureItem
    detection comes from the layout model, which runs regardless of either
    -- there is no reason to pay for OCR just to locate a picture region.

    Crops smaller than _MIN_PICTURE_AREA_FRACTION of their page are dropped
    as furniture rather than saved; see that constant for the evidence.

    Returns 1-indexed page number -> saved crop paths, in Docling's own
    reading order. Every requested page gets an entry, an empty list when
    the layout model (after the size filter) found nothing on it, so a
    caller can tell "checked, nothing there" from "never checked".
    """
    if not pages:
        return {}

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = False
    options.generate_page_images = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    target_dir = Path(output_dir) / Path(pdf_path).stem / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)

    crops: dict[int, list[Path]] = {page: [] for page in pages}
    for page_range in _pages_to_ranges(pages):
        document = converter.convert(pdf_path, page_range=page_range).document
        page_areas = {
            page_no: page.image.pil_image.size[0] * page.image.pil_image.size[1]
            for page_no, page in document.pages.items()
            if page.image
        }
        for item, _level in document.iterate_items():
            if not isinstance(item, PictureItem) or not item.prov:
                continue
            page_number = item.prov[0].page_no
            image = item.get_image(document)
            if image is None:
                continue
            page_area = page_areas.get(page_number)
            if page_area and (image.width * image.height) < _MIN_PICTURE_AREA_FRACTION * page_area:
                continue

            page_crops = crops.setdefault(page_number, [])
            image_path = target_dir / f"p{page_number:04d}-fig{len(page_crops):02d}.png"
            image.save(image_path)
            page_crops.append(image_path)

    return crops


class DocumentExtractor:
    """Wraps the Docling PDF pipeline: OCR for scanned pages, table structure
    recovery, and layout-aware reading order. Produces DocumentChunk objects
    carrying file name and page number provenance.

    This is the slow fallback alongside utils.pdf_extractor.PdfExtractor: use
    it only for the pages PdfExtractor reports as needing OCR, via
    extract_chunks_for_pages(), rather than reprocessing whole documents that
    are mostly born-digital.
    """

    def __init__(self, do_ocr: bool = True, do_table_structure: bool = True):
        options = PdfPipelineOptions()
        options.do_ocr = do_ocr
        options.do_table_structure = do_table_structure
        options.table_structure_options.do_cell_matching = True
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None = None) -> DoclingDocument:
        kwargs = {"page_range": page_range} if page_range else {}
        return self._converter.convert(pdf_path, **kwargs).document

    def extract_chunks(self, pdf_path: Path, max_chars: int = 3000) -> list[DocumentChunk]:
        document = self.extract(pdf_path)
        return self._document_to_chunks(document, Path(pdf_path).name, max_chars)

    def extract_chunks_for_pages(
        self, pdf_path: Path, pages: list[int], max_chars: int = 3000
    ) -> list[DocumentChunk]:
        """Convert and chunk only the given 1-indexed pages. Docling's
        page_range preserves absolute page numbers in the result, so chunks
        from different ranges can be merged and sorted by page directly."""
        file_name = Path(pdf_path).name
        chunks: list[DocumentChunk] = []
        for page_range in _pages_to_ranges(pages):
            document = self.extract(pdf_path, page_range=page_range)
            chunks.extend(self._document_to_chunks(document, file_name, max_chars))

        # Chunk IDs are assigned per conversion call, so ranges beyond the
        # first collide (c0, c1, ...); renumber over the merged list.
        return [
            chunk.model_copy(
                update={"chunk_id": f"{file_name}#p{chunk.page_number}#c{index}"}
            )
            for index, chunk in enumerate(chunks)
        ]

    @staticmethod
    def _document_to_chunks(
        document: DoclingDocument, file_name: str, max_chars: int
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        buffer: list[str] = []
        buffer_page = 0
        last_page = 0
        current_headings: list[str] = []

        def page_of(item) -> int:
            nonlocal last_page
            if item.prov:
                last_page = item.prov[0].page_no
            return last_page

        def flush() -> None:
            nonlocal buffer, buffer_page
            text = "\n".join(buffer).strip()
            if text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{file_name}#p{buffer_page}#c{len(chunks)}",
                        file_name=file_name,
                        page_number=buffer_page,
                        text=text,
                        headings=list(current_headings),
                        kind="text",
                    )
                )
            buffer = []

        for item, _level in document.iterate_items():
            if isinstance(item, SectionHeaderItem):
                flush()
                current_headings = [item.text.strip()]
            elif isinstance(item, TableItem):
                flush()
                page = page_of(item)
                table_text = item.export_to_markdown(document).strip()
                if table_text:
                    # Docling hands back the whole table however wide it is, so
                    # emitting it unsplit broke max_chars, which is a hard bound
                    # everywhere else. Split with the same rule the fast path
                    # uses, repeating the header on each part rather than
                    # leaving a run of numbers with no column names.
                    for part in PdfExtractor._split_table(
                        table_text.splitlines(), max_chars
                    ):
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{file_name}#p{page}#c{len(chunks)}",
                                file_name=file_name,
                                page_number=page,
                                text="\n".join(part).strip(),
                                headings=list(current_headings),
                                kind="table",
                            )
                        )
            elif isinstance(item, TextItem):
                text = item.text.strip()
                if not text:
                    continue
                page = page_of(item)
                # A page change or a full buffer starts a new chunk, keeping
                # page provenance accurate for citations.
                if buffer and (page != buffer_page or sum(map(len, buffer)) + len(text) > max_chars):
                    flush()
                if not buffer:
                    buffer_page = page
                buffer.append(text)

        flush()
        return chunks
