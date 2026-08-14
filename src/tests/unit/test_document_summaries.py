"""Document profiles end to end through the pieces that carry them: the JSON
index, the Postgres store and read path, and what the QA agent shows the model.

The per-file generator has its own file (test_document_summarizer.py); this one
is about the profile travelling from an index to the model's tool result
without ever being ranked against a chunk or turned into a citation."""

from unittest.mock import MagicMock

import pytest

from models.documents import DocumentChunk, DocumentSummary, SearchResult
from models.elements import IndexEntry
from utils.postgres_search_index import PostgresSearchIndex
from utils.postgres_store import upsert_document
from utils.qa_agent import QAAgent

PROFILE = "DESNZ impact assessment covering England and Wales, 2024-2035."


def make_chunk(chunk_id: str, text: str, file_name: str = "paper.pdf", page: int = 1):
    return DocumentChunk(chunk_id=chunk_id, file_name=file_name, page_number=page, text=text)


def make_entry(chunk_id: str = "doc1-p001-c00") -> IndexEntry:
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
        embedding=[0.1, 0.2],
    )


def make_conn_and_cursor():
    conn = MagicMock()
    return conn, conn.cursor.return_value.__enter__.return_value


# =========================================================================
# Postgres store
# =========================================================================


@pytest.mark.unit
def test_upsert_document_writes_the_profile_and_its_vector():
    conn, cursor = make_conn_and_cursor()

    upsert_document(
        conn,
        "doc1",
        "paper.pdf",
        "pdf",
        [make_entry()],
        page_count=48,
        doc_summary=PROFILE,
        summary_embedding=[0.3, 0.4],
        summary_embedding_model="text-embedding-3-large",
    )

    params = cursor.execute.call_args_list[1].args[1]
    assert params["doc_summary"] == PROFILE
    assert params["page_count"] == 48
    assert params["summary_embedding"] is not None


@pytest.mark.unit
def test_upsert_document_without_a_profile_passes_nulls():
    conn, cursor = make_conn_and_cursor()

    upsert_document(conn, "doc1", "paper.pdf", "pdf", [make_entry()])

    params = cursor.execute.call_args_list[1].args[1]
    assert params["doc_summary"] is None
    assert params["summary_embedding"] is None


@pytest.mark.unit
def test_reindexing_without_a_profile_does_not_blank_an_existing_one():
    """`index` re-run after a plain `ingest` passes doc_summary=None, so the
    upsert has to coalesce rather than assign."""
    from utils.postgres_store import _UPSERT_DOCUMENT_SQL

    assert "coalesce(excluded.doc_summary, documents.doc_summary)" in _UPSERT_DOCUMENT_SQL
    assert (
        "coalesce(excluded.summary_embedding, documents.summary_embedding)"
        in _UPSERT_DOCUMENT_SQL
    )


@pytest.mark.unit
def test_upsert_document_raises_on_a_profile_with_no_vector():
    conn, cursor = make_conn_and_cursor()

    with pytest.raises(ValueError):
        upsert_document(
            conn, "doc1", "paper.pdf", "pdf", [make_entry()], doc_summary=PROFILE
        )

    cursor.execute.assert_not_called()


@pytest.mark.unit
def test_schema_adds_profile_columns_to_an_already_created_documents_table():
    """CREATE TABLE IF NOT EXISTS is a no-op on an existing database, so the
    columns only reach one through the ALTERs."""
    from utils.postgres_store import _ALTER_DOCUMENTS_TABLE_SQL

    altered = " ".join(_ALTER_DOCUMENTS_TABLE_SQL)
    for column in ("page_count", "doc_summary", "summary_embedding", "summary_embedding_model"):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in altered


# =========================================================================
# Postgres read path
# =========================================================================


def make_row(**overrides) -> dict:
    row = {
        "chunk_id": "doc1-p003-c00",
        "doc_id": "doc1",
        "file_name": "paper.pdf",
        "file_type": "pdf",
        "granularity": "element",
        "page_start": 3,
        "page_end": 3,
        "element_ids": [],
        "element_types": ["text"],
        "heading_path": ["Results"],
        "embed_text": "body",
        "keyword_text": "body",
        "display_text": "body",
        "asset_paths": [],
        "n_chars": 4,
        "embedding_model": "text-embedding-3-large",
        "embedding": None,
        "doc_summary": PROFILE,
        "fused_score": 0.03,
    }
    row.update(overrides)
    return row


def make_pg_index(rows):
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = rows
    return PostgresSearchIndex(embedder, conn=conn), cursor


@pytest.mark.unit
def test_postgres_search_joins_the_profile_onto_results():
    index, cursor = make_pg_index([make_row()])

    results = index.search("gdp contraction")

    assert results[0].doc_summary == PROFILE
    # First call: the RRF search itself. search() issues a second call after
    # this one to resolve neighbour context (see test_postgres_search_index.py).
    assert "LEFT JOIN documents USING (doc_id)" in cursor.execute.call_args_list[0].args[0]


@pytest.mark.unit
def test_postgres_search_tolerates_a_document_with_no_profile():
    index, _ = make_pg_index([make_row(doc_summary=None)])

    assert index.search("gdp contraction")[0].doc_summary is None


@pytest.mark.unit
def test_search_documents_queries_the_documents_table_not_index_entries():
    index, cursor = make_pg_index(
        [{"file_name": "paper.pdf", "doc_summary": PROFILE, "page_count": 48, "score": 0.7}]
    )

    documents = index.search_documents("heat pump grants", top_k=2)

    sql = cursor.execute.call_args.args[0]
    assert "FROM documents" in sql
    assert "index_entries" not in sql
    assert documents == [
        DocumentSummary(file_name="paper.pdf", summary=PROFILE, page_count=48)
    ]


@pytest.mark.unit
def test_search_documents_ignores_rows_with_no_profile_or_no_vector():
    index, cursor = make_pg_index([])

    index.search_documents("heat pump grants")

    sql = cursor.execute.call_args.args[0]
    assert "summary_embedding IS NOT NULL" in sql
    assert "doc_summary IS NOT NULL" in sql


@pytest.mark.unit
def test_search_documents_on_a_blank_query_makes_no_database_call():
    index, cursor = make_pg_index([])

    assert index.search_documents("   ") == []
    cursor.execute.assert_not_called()


# =========================================================================
# What the model sees
# =========================================================================


@pytest.mark.unit
def test_profile_is_printed_once_per_document_not_once_per_hit():
    results = [
        SearchResult(chunk=make_chunk("c0", "first", page=3), score=2.0, doc_summary=PROFILE),
        SearchResult(chunk=make_chunk("c1", "second", page=9), score=1.0, doc_summary=PROFILE),
    ]

    body = QAAgent(MagicMock(), client=MagicMock())._format_results(results)

    assert body.count(PROFILE) == 1


@pytest.mark.unit
def test_profile_preamble_tells_the_model_it_is_not_citable():
    result = SearchResult(chunk=make_chunk("c0", "first"), score=1.0, doc_summary=PROFILE)

    body = QAAgent(MagicMock(), client=MagicMock())._format_results([result])

    assert "not quotable or citable" in body
    # The ranked list still follows, in score order, unchanged.
    assert body.index(PROFILE) < body.index(">>> MATCH: first")


@pytest.mark.unit
def test_formatting_is_unchanged_when_nothing_was_summarised():
    result = SearchResult(chunk=make_chunk("c0", "first", page=6), score=1.0)

    body = QAAgent(MagicMock(), client=MagicMock())._format_results([result])

    assert body.startswith("[1] Source: paper.pdf, page 6")


@pytest.mark.unit
def test_an_empty_search_falls_back_to_the_document_lane():
    index = MagicMock()
    index.search.return_value = []
    index.search_documents.return_value = [
        DocumentSummary(file_name="paper.pdf", summary=PROFILE, page_count=48)
    ]
    agent = QAAgent(index, client=MagicMock())

    body = agent._no_results_message("heat pump grants")

    assert "No matching extracts found" in body
    assert "paper.pdf (48 pages)" in body
    assert PROFILE in body
    index.search_documents.assert_called_once_with("heat pump grants")


@pytest.mark.unit
def test_an_empty_search_degrades_on_a_backend_without_a_document_lane():
    """The retriever protocol only promises search(); a backend without
    search_documents (checked with hasattr, same as read_source's accessors)
    must degrade to the bare message rather than raise."""
    index = MagicMock(spec=["search"])
    agent = QAAgent(index, client=MagicMock())

    assert agent._no_results_message("heat pump") == "No matching extracts found for that query."


@pytest.mark.unit
def test_an_empty_document_lane_degrades_to_the_bare_message():
    index = MagicMock()
    index.search_documents.return_value = []
    agent = QAAgent(index, client=MagicMock())

    assert agent._no_results_message("heat pump") == "No matching extracts found for that query."
