from unittest.mock import MagicMock

import pytest

from models.elements import IndexEntry
from utils.postgres_store import upsert_document


def make_entry(chunk_id: str = "doc1-p001-c00", embedding=(0.1, 0.2)) -> IndexEntry:
    return IndexEntry(
        chunk_id=chunk_id,
        doc_id="doc1",
        file_name="paper.pdf",
        file_type="pdf",
        page_start=1,
        page_end=1,
        embed_text="body",
        keyword_text="body",
        display_text="body",
        embedding=list(embedding) if embedding is not None else None,
    )


def make_conn_and_cursor():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    return conn, cursor


@pytest.mark.unit
def test_upsert_document_deletes_before_inserting():
    conn, cursor = make_conn_and_cursor()

    upsert_document(conn, "doc1", "paper.pdf", "pdf", [make_entry()])

    calls = cursor.execute.call_args_list
    assert "DELETE FROM index_entries" in calls[0].args[0]
    assert "INSERT INTO documents" in calls[1].args[0]
    cursor.executemany.assert_called_once()


@pytest.mark.unit
def test_upsert_document_raises_on_missing_embedding():
    conn, cursor = make_conn_and_cursor()

    with pytest.raises(ValueError):
        upsert_document(conn, "doc1", "paper.pdf", "pdf", [make_entry(embedding=None)])

    # The bug is caught before any statement is issued, not partway through.
    cursor.execute.assert_not_called()


@pytest.mark.unit
def test_upsert_document_bulk_inserts_all_entries():
    conn, cursor = make_conn_and_cursor()
    entries = [make_entry("doc1-p001-c00"), make_entry("doc1-p001-c01")]

    upsert_document(conn, "doc1", "paper.pdf", "pdf", entries)

    rows = cursor.executemany.call_args.args[1]
    assert [row["chunk_id"] for row in rows] == ["doc1-p001-c00", "doc1-p001-c01"]


@pytest.mark.unit
def test_upsert_document_with_no_entries_skips_bulk_insert_but_still_upserts_document():
    conn, cursor = make_conn_and_cursor()

    upsert_document(conn, "doc1", "paper.pdf", "pdf", [])

    cursor.executemany.assert_not_called()
    calls = cursor.execute.call_args_list
    assert "DELETE FROM index_entries" in calls[0].args[0]
    assert "INSERT INTO documents" in calls[1].args[0]


@pytest.mark.unit
def test_upsert_document_deletes_scoped_to_this_doc_id():
    conn, cursor = make_conn_and_cursor()

    upsert_document(conn, "doc1", "paper.pdf", "pdf", [make_entry()])

    delete_params = cursor.execute.call_args_list[0].args[1]
    assert delete_params == {"doc_id": "doc1"}
