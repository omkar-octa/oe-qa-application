import hashlib
from pathlib import Path

import pytest

from models.documents import DocumentChunk
from models.elements import ElementType
from utils.index_entry_builder import build_index_entries


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    page_number: int = 1,
    kind: str = "text",
    summary: str | None = None,
    headings: list[str] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        file_name="paper.pdf",
        page_number=page_number,
        text=text,
        kind=kind,
        summary=summary,
        headings=headings or [],
    )


def make_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4 fake pdf bytes") -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(content)
    return path


@pytest.mark.unit
def test_text_chunk_has_identical_embed_keyword_display_text(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunk = make_chunk("c0", "Some prose about heat pumps.")

    entries = build_index_entries([chunk], pdf_path)

    assert entries[0].embed_text == entries[0].keyword_text == entries[0].display_text == chunk.text


@pytest.mark.unit
def test_table_chunk_with_summary_uses_summary_for_embed_text_only(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunk = make_chunk(
        "c0",
        "| Region | Cost |\n| --- | --- |\n| UK | 100 |",
        kind="table",
        summary="A table of regional costs.",
    )

    entries = build_index_entries([chunk], pdf_path)
    entry = entries[0]

    assert entry.embed_text == chunk.summary
    assert entry.keyword_text == chunk.text
    assert entry.display_text == chunk.text
    assert entry.embed_text != entry.keyword_text
    assert entry.embed_text != entry.display_text


@pytest.mark.unit
def test_table_chunk_without_summary_falls_back_to_truncated_text(tmp_path):
    pdf_path = make_pdf(tmp_path)
    long_markdown = "| A | B |\n" + ("| x | y |\n" * 200)
    chunk = make_chunk("c0", long_markdown, kind="table", summary=None)

    entries = build_index_entries([chunk], pdf_path)
    entry = entries[0]

    assert entry.embed_text == chunk.text[:500]
    assert entry.keyword_text == chunk.text
    assert entry.display_text == chunk.text


@pytest.mark.unit
def test_doc_id_matches_independently_computed_file_hash(tmp_path):
    known_bytes = b"known pdf content for hashing"
    pdf_path = tmp_path / "known.pdf"
    pdf_path.write_bytes(known_bytes)
    expected_doc_id = hashlib.sha256(known_bytes).hexdigest()[:12]
    chunks = [make_chunk("c0", "first"), make_chunk("c1", "second", page_number=2)]

    entries = build_index_entries(chunks, pdf_path)

    assert entries[0].doc_id == expected_doc_id
    assert entries[1].doc_id == expected_doc_id


@pytest.mark.unit
def test_doc_id_identical_across_chunks_from_same_call(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunks = [
        make_chunk("c0", "first", page_number=1),
        make_chunk("c1", "second", page_number=1),
        make_chunk("c2", "third", page_number=3),
    ]

    entries = build_index_entries(chunks, pdf_path)

    doc_ids = {entry.doc_id for entry in entries}
    assert len(doc_ids) == 1


@pytest.mark.unit
def test_chunk_ids_are_unique_across_the_batch(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunks = [
        make_chunk("c0", "first", page_number=1),
        make_chunk("c1", "second", page_number=1),
        make_chunk("c2", "third", page_number=2),
    ]

    entries = build_index_entries(chunks, pdf_path)

    chunk_ids = [entry.chunk_id for entry in entries]
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.unit
def test_page_start_and_end_track_each_chunks_own_page_not_the_batch(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunks = [
        make_chunk("c0", "page one text", page_number=1),
        make_chunk("c1", "page five text", page_number=5),
        make_chunk("c2", "page three text", page_number=3),
    ]

    entries = build_index_entries(chunks, pdf_path)

    assert (entries[0].page_start, entries[0].page_end) == (1, 1)
    assert (entries[1].page_start, entries[1].page_end) == (5, 5)
    assert (entries[2].page_start, entries[2].page_end) == (3, 3)


@pytest.mark.unit
def test_element_types_reflect_chunk_kind(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunks = [
        make_chunk("c0", "prose", kind="text"),
        make_chunk("c1", "| a |", kind="table", summary="a table"),
        make_chunk("c2", "Fig. 1.\n\nA chart.", kind="figure"),
    ]

    entries = build_index_entries(chunks, pdf_path)

    assert entries[0].element_types == [ElementType.TEXT]
    assert entries[1].element_types == [ElementType.TABLE]
    assert entries[2].element_types == [ElementType.FIGURE]


@pytest.mark.unit
def test_figure_chunk_has_identical_embed_keyword_display_text(tmp_path):
    # A figure chunk's text is already caption plus description, which is both
    # the searchable and the readable form, so unlike a table the three fields
    # do not diverge.
    pdf_path = make_pdf(tmp_path)
    chunk = make_chunk("c0", "Fig. 5. Regional split.\n\nChoropleth map.", kind="figure")

    entries = build_index_entries([chunk], pdf_path)

    assert entries[0].embed_text == entries[0].keyword_text == entries[0].display_text == chunk.text


@pytest.mark.unit
def test_figure_chunk_records_its_page_image_when_image_dir_given(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunk = make_chunk("c0", "Fig. 5.\n\nA map.", kind="figure", page_number=9)

    entries = build_index_entries([chunk], pdf_path, tmp_path / "pages")

    assert entries[0].asset_paths == [str(tmp_path / "pages" / "paper" / "p0009.png")]


@pytest.mark.unit
def test_figure_chunk_asset_paths_empty_without_image_dir(tmp_path):
    # main.index()'s caller may never have rendered anything, so omitting
    # image_dir must not fabricate a path to a file that does not exist.
    pdf_path = make_pdf(tmp_path)
    chunk = make_chunk("c0", "Fig. 5.\n\nA map.", kind="figure", page_number=9)

    entries = build_index_entries([chunk], pdf_path)

    assert entries[0].asset_paths == []


@pytest.mark.unit
def test_non_figure_chunks_get_no_asset_paths_even_with_image_dir(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunks = [
        make_chunk("c0", "prose", kind="text"),
        make_chunk("c1", "| a |", kind="table", summary="a table"),
    ]

    entries = build_index_entries(chunks, pdf_path, tmp_path / "pages")

    assert entries[0].asset_paths == []
    assert entries[1].asset_paths == []


@pytest.mark.unit
def test_heading_path_carried_through_as_is(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunk = make_chunk("c0", "text", headings=["Results", "Cost breakdown"])

    entries = build_index_entries([chunk], pdf_path)

    assert entries[0].heading_path == ["Results", "Cost breakdown"]


@pytest.mark.unit
def test_element_ids_and_asset_paths_are_empty(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunk = make_chunk("c0", "text")

    entries = build_index_entries([chunk], pdf_path)

    assert entries[0].element_ids == []
    assert entries[0].asset_paths == []


@pytest.mark.unit
def test_embedding_and_embedding_model_left_none(tmp_path):
    pdf_path = make_pdf(tmp_path)
    chunk = make_chunk("c0", "text")

    entries = build_index_entries([chunk], pdf_path)

    assert entries[0].embedding is None
    assert entries[0].embedding_model is None
