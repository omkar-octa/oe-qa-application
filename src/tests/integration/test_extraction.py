from pathlib import Path

import pytest

from utils.document_extractor import DocumentExtractor, extract_figure_images

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# Page 1 carries several small badges (a publisher mark, an ORCID icon) that
# Docling's layout model still detects as picture regions; page 9 carries one
# large genuine chart. Together they exercise both sides of the furniture
# size filter in extract_figure_images.
_FIGURE_FIXTURE = FIXTURES / "1-s2.0-S0301421525000862-main.pdf"


@pytest.mark.integration
def test_extract_chunks_from_fixture_pdf():
    pdf_path = min(FIXTURES.glob("*.pdf"), key=lambda p: p.stat().st_size)

    chunks = DocumentExtractor().extract_chunks(pdf_path)

    assert chunks
    assert all(chunk.file_name == pdf_path.name for chunk in chunks)
    assert all(chunk.page_number >= 1 for chunk in chunks)
    assert any(len(chunk.text) > 200 for chunk in chunks)


@pytest.mark.integration
def test_extract_chunks_for_pages_restricts_to_the_given_pages():
    pdf_path = FIXTURES / "1-s2.0-S0140988325000672-main.pdf"

    chunks = DocumentExtractor().extract_chunks_for_pages(pdf_path, [1, 2, 3])

    assert chunks
    assert {chunk.page_number for chunk in chunks} <= {1, 2, 3}
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


@pytest.mark.integration
def test_extract_figure_images_saves_a_real_figure_crop(tmp_path):
    crops = extract_figure_images(_FIGURE_FIXTURE, [9], tmp_path)

    assert crops[9]
    assert all(path.exists() for path in crops[9])


@pytest.mark.integration
def test_extract_figure_images_drops_furniture_sized_crops(tmp_path):
    # A masthead-heavy page's small badges must not survive the area filter,
    # even though Docling's layout model detects them as picture regions.
    crops = extract_figure_images(_FIGURE_FIXTURE, [1], tmp_path)

    assert crops[1] == []


@pytest.mark.integration
def test_extract_figure_images_reports_every_requested_page(tmp_path):
    crops = extract_figure_images(_FIGURE_FIXTURE, [1, 9], tmp_path)

    assert set(crops.keys()) == {1, 9}
