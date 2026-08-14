from pathlib import Path

import pytest

from utils.pdf_extractor import PdfExtractor

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MIXED_DOCUMENT = "1-s2.0-S0140988325000672-main.pdf"


@pytest.mark.integration
def test_extract_chunks_from_fixture_pdf():
    pdf_path = min(FIXTURES.glob("*.pdf"), key=lambda p: p.stat().st_size)

    chunks = PdfExtractor().extract_chunks(pdf_path)

    assert chunks
    assert all(chunk.file_name == pdf_path.name for chunk in chunks)
    assert all(chunk.page_number >= 1 for chunk in chunks)
    assert any(len(chunk.text) > 200 for chunk in chunks)


@pytest.mark.integration
def test_max_chars_is_a_hard_bound_on_real_documents():
    extractor = PdfExtractor()

    for pdf_path in sorted(FIXTURES.glob("*.pdf")):
        chunks = extractor.extract_chunks(pdf_path, max_chars=1000)

        assert all(len(chunk.text) <= 1000 for chunk in chunks), pdf_path.name


@pytest.mark.integration
def test_mixed_document_reports_its_scanned_pages_needing_ocr():
    """Locks in the mixed-document case found during the PyMuPDF -> pdf_inspector
    migration: this fixture has pages 1-7 with no usable embedded text and
    pages 8-16 that are fine. A document-wide check would either skip the
    whole file or silently drop these pages; per-page classification catches
    them so main.py can route just these pages to Docling."""
    pdf_path = FIXTURES / MIXED_DOCUMENT

    ocr_pages = PdfExtractor().pages_needing_ocr(pdf_path)
    chunks = PdfExtractor().extract_chunks(pdf_path)

    assert ocr_pages
    assert all(chunk.page_number not in ocr_pages for chunk in chunks)


@pytest.mark.integration
def test_clean_fixtures_report_no_pages_needing_ocr():
    extractor = PdfExtractor()

    for pdf_path in sorted(FIXTURES.glob("*.pdf")):
        if pdf_path.name == MIXED_DOCUMENT:
            continue
        assert extractor.pages_needing_ocr(pdf_path) == [], pdf_path.name
