import pytest
from pydantic import ValidationError

from models.documents import DocumentChunk, SearchIndex


def make_chunk(**overrides) -> DocumentChunk:
    defaults = {
        "chunk_id": "paper.pdf#p1#c0",
        "file_name": "paper.pdf",
        "page_number": 1,
        "text": "Some content.",
    }
    defaults.update(overrides)
    return DocumentChunk(**defaults)


@pytest.mark.unit
def test_chunk_requires_core_fields():
    with pytest.raises(ValidationError):
        DocumentChunk(chunk_id="x", file_name="paper.pdf")


@pytest.mark.unit
def test_chunk_defaults():
    chunk = make_chunk()
    assert chunk.headings == []
    assert chunk.embedding is None


@pytest.mark.unit
def test_chunk_kind_defaults_to_text_and_accepts_table():
    assert make_chunk().kind == "text"
    assert make_chunk(kind="table").kind == "table"


@pytest.mark.unit
def test_search_index_json_round_trip():
    index = SearchIndex(chunks=[make_chunk(), make_chunk(chunk_id="paper.pdf#p2#c1", page_number=2)])
    restored = SearchIndex.model_validate_json(index.model_dump_json())
    assert restored == index


@pytest.mark.unit
def test_search_index_written_before_profiles_existed_still_loads():
    """`documents` was added after `chunks`; an index file from before that
    change has no such key at all, not an empty list for it."""
    index = SearchIndex.model_validate_json(
        '{"version": 1, "created_at": "2026-01-01T00:00:00Z", "chunks": []}'
    )
    assert index.documents == []
