from types import SimpleNamespace
from unittest.mock import patch

import pytest

from utils.pdf_extractor import PdfExtractor


def page(page_index: int, markdown: str, needs_ocr: bool = False):
    return SimpleNamespace(page=page_index, markdown=markdown, needs_ocr=needs_ocr, ocr_reason=None)


def result(pages: list, pages_needing_ocr: list[int] | None = None):
    return SimpleNamespace(
        pages=pages,
        pages_needing_ocr=pages_needing_ocr or [],
        pages_with_tables=[],
        pages_with_columns=[],
        ocr_reasons_by_page=[],
        is_complex=False,
    )


def mocked(return_value):
    return patch("utils.pdf_extractor.pdf_inspector.extract_pages_markdown", return_value=return_value)


@pytest.mark.unit
def test_extract_chunks_uses_one_indexed_page_numbers():
    with mocked(result([page(0, "Body text on the first page.")])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].file_name == "doc.pdf"


@pytest.mark.unit
def test_extract_chunks_skips_pages_needing_ocr():
    with mocked(
        result(
            [
                page(0, "Readable page.", needs_ocr=False),
                page(1, "", needs_ocr=True),
            ]
        )
    ):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert len(chunks) == 1
    assert chunks[0].page_number == 1


@pytest.mark.unit
def test_extract_chunks_skips_pages_with_blank_markdown():
    with mocked(result([page(0, "   \n  ")])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert chunks == []


@pytest.mark.unit
def test_heading_lines_become_headings_metadata():
    markdown = "# Introduction\nSome body text.\n## Background\nMore body text."
    with mocked(result([page(0, markdown)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    headings = [chunk.headings for chunk in chunks]
    assert ["Introduction"] in headings
    assert ["Background"] in headings


@pytest.mark.unit
def test_heading_text_included_in_chunk_body():
    with mocked(result([page(0, "# Results\nFindings go here.")])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert "# Results" in chunks[0].text
    assert "Findings go here." in chunks[0].text


@pytest.mark.unit
def test_max_chars_splits_a_long_page_into_multiple_chunks():
    body = "Body line.\n" * 200
    with mocked(result([page(0, body)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf", max_chars=200)

    assert len(chunks) > 1
    assert all(chunk.page_number == 1 for chunk in chunks)


@pytest.mark.unit
def test_chunk_ids_are_unique_within_a_document():
    with mocked(result([page(0, "# A\nOne."), page(1, "# B\nTwo.")])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


@pytest.mark.unit
def test_pages_needing_ocr_returns_the_result_field():
    with mocked(result([page(0, "Text.")], pages_needing_ocr=[3, 4, 5])):
        assert PdfExtractor().pages_needing_ocr("doc.pdf") == [3, 4, 5]


@pytest.mark.unit
def test_markdown_table_row_produces_a_table_chunk():
    markdown = "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |"
    with mocked(result([page(0, markdown)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert len(chunks) == 1
    assert chunks[0].kind == "table"


@pytest.mark.unit
def test_plain_prose_produces_a_text_chunk():
    with mocked(result([page(0, "Some ordinary body text.")])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert len(chunks) == 1
    assert chunks[0].kind == "text"


@pytest.mark.unit
def test_a_single_line_longer_than_max_chars_is_still_split():
    # A wide markdown table row can be one unbroken line longer than
    # max_chars; the chunker must split it, not just the buffer between lines.
    long_line = "|" + " word" * 100 + "|"
    with mocked(result([page(0, long_line)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf", max_chars=200)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks).split() == long_line.split()


@pytest.mark.unit
def test_table_is_never_split_from_its_header():
    """A run of numbers with no column names cannot be attributed to a
    variable, so a table and its header must land in the same chunk."""
    body = "\n".join(f"| row {i} | {i}.0 | {i}.5 |" for i in range(120))
    markdown = f"| Var | Est | SE |\n| --- | --- | --- |\n{body}"
    with mocked(result([page(0, markdown)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf", max_chars=800)

    assert len(chunks) > 1, "fixture must be long enough to force a split"
    for chunk in chunks:
        assert "| Var | Est | SE |" in chunk.text
        assert chunk.kind == "table"
        assert len(chunk.text) <= 800


@pytest.mark.unit
def test_table_caption_stays_with_its_table():
    markdown = (
        "Some preceding prose that belongs to the section.\n"
        "Table 5\n"
        "Decomposition of inequality.\n"
        "| Factor | Share |\n| --- | --- |\n| Education | 14.71 % |"
    )
    with mocked(result([page(0, markdown)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    table = [c for c in chunks if c.kind == "table"]
    assert len(table) == 1
    assert "Table 5" in table[0].text
    assert "Decomposition of inequality." in table[0].text
    assert "preceding prose" not in table[0].text


@pytest.mark.unit
def test_prose_before_a_table_is_not_swallowed_as_a_caption():
    markdown = (
        "The results are reported below and discussed in the next section.\n"
        "| Factor | Share |\n| --- | --- |\n| Education | 14.71 % |"
    )
    with mocked(result([page(0, markdown)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    table = [c for c in chunks if c.kind == "table"]
    assert "discussed in the next section" not in table[0].text


@pytest.mark.unit
def test_table_does_not_share_a_chunk_with_surrounding_prose():
    """index_entry_builder embeds a table's summary but keyword-searches its
    Markdown; that split is incoherent if prose rides along in the chunk."""
    markdown = (
        "Introductory sentence.\n"
        "| Factor | Share |\n| --- | --- |\n| Education | 14.71 % |\n"
        "Trailing discussion of the result."
    )
    with mocked(result([page(0, markdown)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    table = [c for c in chunks if c.kind == "table"][0]
    assert "Introductory sentence" not in table.text
    assert "Trailing discussion" not in table.text


@pytest.mark.unit
def test_kind_counts_rows_rather_than_testing_the_first_line():
    """A running header lands ahead of the caption on these fixtures, so a
    first-line test called whole tables prose and header-less fragments tables."""
    markdown = (
        "*A. Burlinson et al. Energy Economics 143 108244*\n"
        "Table 3\n"
        "| Factor | Share |\n| --- | --- |\n| Education | 14.71 % |"
    )
    with mocked(result([page(0, markdown)])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert [c.kind for c in chunks if "| Education |" in c.text] == ["table"]


@pytest.mark.unit
def test_single_pipe_line_in_prose_is_not_a_table():
    with mocked(result([page(0, "Prose mentioning a | pipe character inline.")])):
        chunks = PdfExtractor().extract_chunks("doc.pdf")

    assert chunks[0].kind == "text"
