"""Integration test against a real Postgres + pgvector container.

Requires `docker compose up -d` from the repository root (the compose file
lives at the repo root, one level above src/). See test_postgres_connection.py
for the same connectivity pattern without the schema/upsert/search layer.
"""

import pytest

from models.elements import Granularity, IndexEntry
from utils.postgres_search_index import PostgresSearchIndex
from utils.postgres_store import connect, ensure_schema, upsert_document

_FAKE_DIM = 512  # matches settings.embedding_dimensions


def make_entry(
    chunk_id: str, doc_id: str, keyword_text: str, display_text: str, page: int = 1
) -> IndexEntry:
    return IndexEntry(
        chunk_id=chunk_id,
        doc_id=doc_id,
        file_name="integration.pdf",
        file_type="pdf",
        granularity=Granularity.ELEMENT,
        page_start=page,
        page_end=page,
        embed_text=display_text,
        keyword_text=keyword_text,
        display_text=display_text,
        n_chars=len(display_text),
        embedding=[0.1] * _FAKE_DIM,
        embedding_model="fake-test-model",
    )


class FakeEmbedder:
    """Returns a fixed vector regardless of input text; small and deterministic,
    which is all a query embedding needs to be for this test."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _FAKE_DIM for _ in texts]


@pytest.mark.integration
def test_ensure_schema_upsert_and_search_round_trip():
    conn = connect()
    try:
        ensure_schema(conn)

        doc_id = "it-doc-0001"
        entries = [
            make_entry(f"{doc_id}-p001-c00", doc_id, "reciprocal rank fusion hybrid search", "Hybrid search fuses keyword and vector ranks."),
            make_entry(f"{doc_id}-p001-c01", doc_id, "unrelated filler text about weather", "Unrelated filler text about weather."),
        ]

        upsert_document(conn, doc_id, "integration.pdf", "pdf", entries)
        conn.commit()

        index = PostgresSearchIndex(FakeEmbedder(), conn=conn)
        results = index.search("reciprocal rank fusion hybrid search", top_k=5)

        assert results
        result_chunk_ids = [r.chunk.chunk_id for r in results]
        assert entries[0].chunk_id in result_chunk_ids

        top = next(r for r in results if r.chunk.chunk_id == entries[0].chunk_id)
        assert top.chunk.file_name == "integration.pdf"
        assert top.chunk.page_number == 1
        assert top.chunk.text == "Hybrid search fuses keyword and vector ranks."
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM index_entries WHERE doc_id = %(doc_id)s", {"doc_id": "it-doc-0001"})
            cur.execute("DELETE FROM documents WHERE doc_id = %(doc_id)s", {"doc_id": "it-doc-0001"})
        conn.commit()
        conn.close()


@pytest.mark.integration
def test_document_profile_round_trip_and_survives_a_reindex():
    """The document lane against real DDL.

    Worth an integration test rather than a mock: the profile columns reach an
    already-created `documents` table only through ensure_schema's ALTERs, and
    a mocked connection cannot show that those ran, that `vector(N)` was a
    known type when they did, or that the coalescing upsert really preserves a
    profile across a re-index that has none.
    """
    conn = connect()
    doc_id = "it-doc-0002"
    profile = "Integration fixture describing hybrid search over two fake chunks."
    try:
        ensure_schema(conn)

        entries = [
            make_entry(f"{doc_id}-p001-c00", doc_id, "reciprocal rank fusion", "Hybrid search fuses ranks.")
        ]
        upsert_document(
            conn,
            doc_id,
            "integration.pdf",
            "pdf",
            entries,
            page_count=12,
            doc_summary=profile,
            summary_embedding=[0.1] * _FAKE_DIM,
            summary_embedding_model="fake-test-model",
        )
        conn.commit()

        index = PostgresSearchIndex(FakeEmbedder(), conn=conn)

        # Carried onto ordinary results by the join, for the "right document?" call.
        results = index.search("reciprocal rank fusion", top_k=5)
        assert results
        assert results[0].doc_summary == profile

        # And reachable on its own, from its own table and its own HNSW index.
        documents = index.search_documents("reciprocal rank fusion", top_k=5)
        assert any(doc.file_name == "integration.pdf" and doc.summary == profile for doc in documents)
        assert next(doc for doc in documents if doc.summary == profile).page_count == 12

        # A plain `index` run passes no profile; the existing one must survive.
        upsert_document(conn, doc_id, "integration.pdf", "pdf", entries)
        conn.commit()
        assert index.search("reciprocal rank fusion", top_k=5)[0].doc_summary == profile
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM index_entries WHERE doc_id = %(doc_id)s", {"doc_id": doc_id})
            cur.execute("DELETE FROM documents WHERE doc_id = %(doc_id)s", {"doc_id": doc_id})
        conn.commit()
        conn.close()


@pytest.mark.integration
def test_document_page_and_files_against_real_ddl():
    """The read_source accessors, against a real schema rather than a mocked
    cursor -- worth it mainly to prove the column names in the hand-written
    SQL (page_start, display_text, heading_path, element_types) actually match
    what ensure_schema() creates."""
    conn = connect()
    doc_id = "it-doc-0003"
    try:
        ensure_schema(conn)

        entries = [
            make_entry(f"{doc_id}-p001-c00", doc_id, "page one keywords", "Page one text.", page=1),
            make_entry(f"{doc_id}-p002-c00", doc_id, "page two keywords", "Page two text.", page=2),
        ]
        upsert_document(conn, doc_id, "integration.pdf", "pdf", entries)
        conn.commit()

        index = PostgresSearchIndex(FakeEmbedder(), conn=conn)

        whole_document = index.document("integration.pdf")
        assert [c.text for c in whole_document] == ["Page one text.", "Page two text."]
        assert [c.page_number for c in whole_document] == [1, 2]

        page_two = index.page("integration.pdf", 2)
        assert [c.text for c in page_two] == ["Page two text."]

        assert index.document("no-such-file.pdf") == []
        assert "integration.pdf" in index.files()
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM index_entries WHERE doc_id = %(doc_id)s", {"doc_id": doc_id})
            cur.execute("DELETE FROM documents WHERE doc_id = %(doc_id)s", {"doc_id": doc_id})
        conn.commit()
        conn.close()
