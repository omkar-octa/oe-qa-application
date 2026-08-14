"""Postgres-backed storage for IndexEntry rows: schema management and upsert.

Two tables, one join: `documents` (one row per source file) and
`index_entries` (one row per IndexEntry, see models/elements.py for why the
schema is split into three text fields per entry).

No migration tooling. Schema changes are idempotent DDL run from Python
every time the caller wants to be sure the schema is current; this is a
deliberate choice for this project's size, not an oversight.
"""

from __future__ import annotations

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from models.config import settings
from models.elements import IndexEntry

# =========================================================================
# DDL
# =========================================================================

_CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

# doc_summary and its vector live here, on the one-row-per-file table, rather
# than as granularity='document' rows in index_entries. Keeping them in a
# separate table is what stops a profile competing with chunks for a slot in
# the ranked results: it takes a deliberate second query to reach them, not a
# `granularity = 'element'` filter that every future query has to remember.
# It also means a profile never needs a page range, so it can never be dressed
# up as a citation.
#
# The vector dimension is inlined from settings.embedding_dimensions (an
# f-string, not a bound parameter: pgvector's type declaration syntax needs
# the number in the DDL text itself, the same way _RRF_K is inlined into
# postgres_search_index.py's query below). Changing the setting changes the
# column on the next ensure_schema() call only for a fresh database -- see
# ensure_schema()'s docstring for what an existing one needs instead.
_CREATE_DOCUMENTS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS documents (
    doc_id text PRIMARY KEY,
    file_name text NOT NULL,
    file_type text NOT NULL,
    page_count int,
    doc_summary text,
    summary_embedding vector({settings.embedding_dimensions}),
    summary_embedding_model text,
    ingested_at timestamptz NOT NULL DEFAULT now()
)
"""

# CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
# the columns above reach an already-created database only through these.
# Both statements are needed, and both must stay: the CREATE is what a fresh
# database uses, these are what every other one uses.
_ALTER_DOCUMENTS_TABLE_SQL = [
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count int",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_summary text",
    f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_embedding vector({settings.embedding_dimensions})",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_embedding_model text",
]

# embedding vector(N): dimension matches settings.embedding_dimensions, the
# length OpenAI's embeddings API is asked to return (see utils/embedder.py).
_CREATE_INDEX_ENTRIES_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS index_entries (
    chunk_id text PRIMARY KEY,
    doc_id text NOT NULL REFERENCES documents(doc_id),
    file_name text NOT NULL,
    file_type text NOT NULL,
    granularity text NOT NULL,
    page_start int NOT NULL,
    page_end int NOT NULL,
    element_ids text[] NOT NULL DEFAULT '{{}}',
    element_types text[] NOT NULL DEFAULT '{{}}',
    heading_path text[] NOT NULL DEFAULT '{{}}',
    embed_text text NOT NULL,
    keyword_text text NOT NULL,
    display_text text NOT NULL,
    asset_paths text[] NOT NULL DEFAULT '{{}}',
    n_chars int NOT NULL DEFAULT 0,
    embedding_model text,
    embedding vector({settings.embedding_dimensions}),
    keyword_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', keyword_text)) STORED
)
"""

_CREATE_KEYWORD_TSV_GIN_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS index_entries_keyword_tsv_gin_idx
    ON index_entries USING gin (keyword_tsv)
"""

# OpenAI's embeddings are documented as L2-normalised, so cosine and
# inner-product ranking agree; cosine (vector_cosine_ops) is the simpler one
# to reason about.
_CREATE_EMBEDDING_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS index_entries_embedding_hnsw_idx
    ON index_entries USING hnsw (embedding vector_cosine_ops)
"""

# The document lane's own index. Separate from the entry index by construction,
# since it is on a different table.
_CREATE_SUMMARY_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS documents_summary_embedding_hnsw_idx
    ON documents USING hnsw (summary_embedding vector_cosine_ops)
"""

# =========================================================================
# Upsert
# =========================================================================

_DELETE_STALE_ENTRIES_SQL = "DELETE FROM index_entries WHERE doc_id = %(doc_id)s"

# coalesce on the summary columns, not plain assignment: re-running `index`
# without having re-summarised passes doc_summary=None, and that must leave an
# existing profile alone rather than blanking it. file_name and file_type are
# always known by the caller, so those overwrite unconditionally.
_UPSERT_DOCUMENT_SQL = """
INSERT INTO documents (
    doc_id, file_name, file_type, page_count,
    doc_summary, summary_embedding, summary_embedding_model
)
VALUES (
    %(doc_id)s, %(file_name)s, %(file_type)s, %(page_count)s,
    %(doc_summary)s, %(summary_embedding)s, %(summary_embedding_model)s
)
ON CONFLICT (doc_id) DO UPDATE SET
    file_name = excluded.file_name,
    file_type = excluded.file_type,
    page_count = coalesce(excluded.page_count, documents.page_count),
    doc_summary = coalesce(excluded.doc_summary, documents.doc_summary),
    summary_embedding = coalesce(excluded.summary_embedding, documents.summary_embedding),
    summary_embedding_model = coalesce(
        excluded.summary_embedding_model, documents.summary_embedding_model
    )
"""

_INSERT_ENTRY_SQL = """
INSERT INTO index_entries (
    chunk_id, doc_id, file_name, file_type, granularity, page_start, page_end,
    element_ids, element_types, heading_path, embed_text, keyword_text, display_text,
    asset_paths, n_chars, embedding_model, embedding
) VALUES (
    %(chunk_id)s, %(doc_id)s, %(file_name)s, %(file_type)s, %(granularity)s,
    %(page_start)s, %(page_end)s, %(element_ids)s, %(element_types)s, %(heading_path)s,
    %(embed_text)s, %(keyword_text)s, %(display_text)s, %(asset_paths)s, %(n_chars)s,
    %(embedding_model)s, %(embedding)s
)
"""


def connect() -> psycopg.Connection:
    """Open a connection using models.config.settings, ready for vector columns.

    register_vector() requires the vector type to already exist in this
    database. On a brand new database that error is swallowed here:
    ensure_schema() creates the extension and registers the type itself, so
    the first-ever run still works, it just does the registration a step
    later than this function.
    """
    conn = psycopg.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
        connect_timeout=5,
    )
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        pass
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    """Idempotent DDL: safe to call every time. Creates the vector extension,
    both tables, any columns added to `documents` since it was first created,
    the keyword GIN index and both embedding HNSW indexes if they do not
    already exist.

    CREATE TABLE IF NOT EXISTS does not change an existing table's column
    types, so changing settings.embedding_dimensions only takes effect on a
    fresh database. An existing one with vector columns at the old dimension
    needs those columns dropped and recreated by hand (safe only if they hold
    no data worth keeping, since the vectors themselves aren't portable
    across a dimension change anyway)."""
    with conn.transaction():
        conn.execute(_CREATE_EXTENSION_SQL)
        conn.execute(_CREATE_DOCUMENTS_TABLE_SQL)
        # After CREATE EXTENSION, so `vector(N)` is a known type here.
        for statement in _ALTER_DOCUMENTS_TABLE_SQL:
            conn.execute(statement)
        conn.execute(_CREATE_INDEX_ENTRIES_TABLE_SQL)
        conn.execute(_CREATE_KEYWORD_TSV_GIN_INDEX_SQL)
        conn.execute(_CREATE_EMBEDDING_HNSW_INDEX_SQL)
        conn.execute(_CREATE_SUMMARY_HNSW_INDEX_SQL)
    # The extension is guaranteed to exist past this point, so registration
    # can't fail here the way it optimistically might in connect().
    register_vector(conn)


def _entry_to_row(entry: IndexEntry) -> dict:
    return {
        "chunk_id": entry.chunk_id,
        "doc_id": entry.doc_id,
        "file_name": entry.file_name,
        "file_type": entry.file_type,
        "granularity": entry.granularity.value,
        "page_start": entry.page_start,
        "page_end": entry.page_end,
        "element_ids": entry.element_ids,
        "element_types": [element_type.value for element_type in entry.element_types],
        "heading_path": entry.heading_path,
        "embed_text": entry.embed_text,
        "keyword_text": entry.keyword_text,
        "display_text": entry.display_text,
        "asset_paths": entry.asset_paths,
        "n_chars": entry.n_chars,
        "embedding_model": entry.embedding_model,
        "embedding": Vector(entry.embedding),
    }


def upsert_document(
    conn: psycopg.Connection,
    doc_id: str,
    file_name: str,
    file_type: str,
    entries: list[IndexEntry],
    *,
    page_count: int | None = None,
    doc_summary: str | None = None,
    summary_embedding: list[float] | None = None,
    summary_embedding_model: str | None = None,
) -> None:
    """Replace one document's index entries in a single transaction.

    Stale rows from a previous ingest of the same file are deleted first,
    since chunk_id's sequence number is positional and re-extraction can
    shift it; upserting by primary key alone would leave orphans behind.

    Every entry must already have `embedding` populated. Forgetting to embed
    before calling this is a caller bug, not something to paper over with a
    null vector.

    The document-level arguments are keyword-only and all optional: the
    profile is written by an opt-in ingest flag, so most calls have none of
    them, and passing none leaves whatever the document row already had. A
    profile is only stored with its vector, since a profile the document lane
    cannot search is dead weight in the row.
    """
    for entry in entries:
        if entry.embedding is None:
            raise ValueError(
                f"IndexEntry {entry.chunk_id!r} has embedding=None; embed entries "
                "before calling upsert_document, this function does not embed on "
                "the caller's behalf."
            )

    if doc_summary is not None and summary_embedding is None:
        raise ValueError(
            f"doc_summary given for doc_id {doc_id!r} with summary_embedding=None; "
            "embed the summary before calling upsert_document, the same way entry "
            "embeddings are the caller's job."
        )

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(_DELETE_STALE_ENTRIES_SQL, {"doc_id": doc_id})
            cur.execute(
                _UPSERT_DOCUMENT_SQL,
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "page_count": page_count,
                    "doc_summary": doc_summary,
                    "summary_embedding": (
                        Vector(summary_embedding) if summary_embedding is not None else None
                    ),
                    "summary_embedding_model": summary_embedding_model,
                },
            )
            if entries:
                cur.executemany(_INSERT_ENTRY_SQL, [_entry_to_row(entry) for entry in entries])
