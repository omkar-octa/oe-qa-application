# Postgres

The storage design for this repo's knowledge base: the two tables, why they're shaped
the way they are, how a document's rows stay consistent across repeated `index` runs,
and how to stand up and connect to the database itself. For the ranking query that reads
this schema (the RRF fusion, the keyword and vector branches, context expansion) see
[search.md](search.md#3-hybrid-search-the-rrf-query); for the design decisions behind
Postgres being the only backend see [architecture.md](architecture.md); for a one-line
description of each module see [features.md](features.md). This doc is the schema
reference: what each column is for, how identity and idempotency are enforced, and what
changes safely versus what needs a fresh database.

`utils/postgres_store.py` owns the schema and all writes. `utils/postgres_search_index.py`
owns all reads. Nothing else touches the database directly.

## 1. Standing up a database

`docker-compose.yml` at the repo root runs `pgvector/pgvector:pg16`, exposed on the
standard `5432` port with default credentials (`postgres`/`postgres`, database
`embeddings`) matching `models/config.py`'s defaults:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: embeddings
    ports:
      - "5432:5432"
    volumes:
      - pg_embeddings_data:/var/lib/postgresql/data
```

`docker compose up -d` is enough for local development; override `pg_host`, `pg_port`,
`pg_user`, `pg_password`, `pg_database` via `.env` or the real environment for anything
else. Embedding still needs `OPENAI_API_KEY` set (`docs/search.md#7-embeddings`); the
database itself needs nothing beyond the `vector` extension, which `ensure_schema()`
creates itself.

There is no separate migration step or CLI command to run first. `python main.py index`
calls `connect()` then `ensure_schema(conn)` before touching any document, every run, so
the schema is always brought current as a side effect of indexing rather than something
you run once and forget.

## 2. The two tables

One row per source file, one row per chunk, joined on `doc_id`.

### `documents`

| Column | Type | Notes |
| --- | --- | --- |
| `doc_id` | `text` | Primary key. First 12 hex characters of the source file's sha256 (section 3). |
| `file_name` | `text` | `NOT NULL`. Overwritten unconditionally on re-index. |
| `file_type` | `text` | `NOT NULL`. Overwritten unconditionally on re-index. |
| `page_count` | `int` | Nullable. `coalesce`d on upsert (section 4), so a run that doesn't recompute it doesn't blank it. |
| `doc_summary` | `text` | Nullable. LLM-written document profile from `ingest --summarise-documents`; most corpora never populate this. |
| `summary_embedding` | `vector(N)` | Nullable. Only ever set alongside `doc_summary`. |
| `summary_embedding_model` | `text` | Nullable. Which model produced `summary_embedding`. |
| `ingested_at` | `timestamptz` | `NOT NULL DEFAULT now()`. Set once on first insert; upsert never touches it. |

### `index_entries`

| Column | Type | Notes |
| --- | --- | --- |
| `chunk_id` | `text` | Primary key. `{doc_id}-p{page:03d}-c{seq:02d}` (section 3). |
| `doc_id` | `text` | `NOT NULL REFERENCES documents(doc_id)`. |
| `file_name` | `text` | `NOT NULL`. Denormalised from `documents` so search results and `read_source` lookups never need the join. |
| `file_type` | `text` | `NOT NULL`. |
| `granularity` | `text` | `NOT NULL`. One of `Granularity`'s values (`document`, `section`, `element`); chunk rows are always `element`. |
| `page_start` / `page_end` | `int` | `NOT NULL`. 1-indexed; a chunk never spans pages, so these are equal for every row today. |
| `element_ids` | `text[]` | Join back to the extraction layer (`ParsedDocument.elements`). Empty for anything that isn't element-level. |
| `element_types` | `text[]` | Tags used to recover `kind` (`table`/`text`/`figure`) on read, since the column isn't stored directly. |
| `heading_path` | `text[]` | Section breadcrumb, used for display and prepended to `embed_text` for prose chunks. |
| `embed_text` | `text` | `NOT NULL`. What gets embedded (section 5 of `search.md`). |
| `keyword_text` | `text` | `NOT NULL`. What Postgres FTS indexes. |
| `display_text` | `text` | `NOT NULL`. What the LLM actually sees after retrieval. |
| `asset_paths` | `text[]` | Table CSVs, figure crops, page images on disk. |
| `n_chars` | `int` | `NOT NULL DEFAULT 0`. |
| `embedding_model` | `text` | Nullable. Which model produced `embedding`. |
| `embedding` | `vector(N)` | Nullable in the column definition, but `upsert_document()` raises before insert if any entry's embedding is `None` (section 4). |
| `keyword_tsv` | `tsvector` | `GENERATED ALWAYS AS (to_tsvector('english', keyword_text)) STORED`. Postgres maintains this; nothing ever writes to it directly. |

`N` (`vector(N)`'s dimension) is `settings.embedding_dimensions`, currently `512`,
inlined into the `CREATE TABLE`/`ALTER TABLE` text as an f-string rather than passed as a
bound parameter, since pgvector's type syntax needs the number written into the type
itself.

Why `embed_text`, `keyword_text` and `display_text` are three separate columns instead of
one, and why tables in particular need them to diverge, is `search.md`'s territory
(`docs/search.md#9-why-a-tables-three-text-fields-diverge`), not repeated here.

## 3. Identity: `doc_id` and `chunk_id`

Both ids are derived, not assigned:

- `doc_id` is the first 12 hex characters of `sha256(file_bytes)`
  (`ParsedDocument.hash_file()` / `make_doc_id()` in `models/elements.py`). Two runs over
  byte-identical files always produce the same `doc_id`; editing the file's content
  produces a different one.
- `chunk_id` is `{doc_id}-p{page:03d}-c{seq:02d}` (e.g. `a3f2c19b40e1-p012-c04`),
  assigned by `utils/index_entry_builder.py` while walking one document's chunks in
  order. The `-c{seq}` suffix is positional within that single build, not a stable id
  carried across runs.

Content identity, not filename, is what re-indexing keys on. Re-running `ingest` +
`index` on the same bytes reproduces the same `doc_id` and the same set of `chunk_id`s,
so the upsert in section 4 replaces the row set cleanly. Editing the source PDF and
re-indexing it produces a *new* `doc_id`: the old document's rows are not found or
touched by that run, so they are left behind under the old `doc_id` rather than being
replaced. There is no code path that reconciles two `doc_id`s sharing a `file_name`;
retiring a changed file's old rows is a manual `DELETE FROM documents WHERE doc_id = ...`
(cascading is not set up, so `index_entries` needs its own `DELETE ... WHERE doc_id`
first).

## 4. Re-indexing without duplicating: `upsert_document()`

`upsert_document()` (`utils/postgres_store.py`) is the only write path, and it replaces a
document's rows rather than incrementally patching them, in one transaction:

```sql
DELETE FROM index_entries WHERE doc_id = %(doc_id)s;

INSERT INTO documents (doc_id, file_name, file_type, page_count,
                        doc_summary, summary_embedding, summary_embedding_model)
VALUES (...)
ON CONFLICT (doc_id) DO UPDATE SET
    file_name = excluded.file_name,
    file_type = excluded.file_type,
    page_count = coalesce(excluded.page_count, documents.page_count),
    doc_summary = coalesce(excluded.doc_summary, documents.doc_summary),
    summary_embedding = coalesce(excluded.summary_embedding, documents.summary_embedding),
    summary_embedding_model = coalesce(excluded.summary_embedding_model,
                                        documents.summary_embedding_model);

-- then INSERT every fresh index_entries row for doc_id
```

Two deliberate choices here:

- **Delete-then-insert for `index_entries`, not an upsert by primary key.** A chunk's
  `-c{seq}` suffix is positional (section 3), so re-extraction can change how many
  chunks a document produces or how they're numbered. Upserting by `chunk_id` alone
  would leave old rows with now-stale sequence numbers as orphans instead of replacing
  them; deleting every row under that `doc_id` first guarantees the new insert is the
  complete, correct set with nothing left behind.
- **`coalesce` on the `documents` summary columns, not a plain overwrite.** `index`
  usually runs without a document profile (`doc_summary`/`summary_embedding` both
  `None`, since profiling is a separate opt-in `ingest --summarise-documents` step). A
  plain `SET doc_summary = excluded.doc_summary` would blank out a profile a previous run
  already wrote. `file_name`/`file_type` skip the `coalesce` and overwrite unconditionally
  because the caller always knows both.

Two related invariants `upsert_document()` enforces before it will touch the database:

- Every `IndexEntry` passed in must already have `embedding` populated; it raises
  rather than writing a null vector, because embedding is the caller's job
  (`main.py index`), not something to silently skip.
- If `doc_summary` is given, `summary_embedding` must be too, for the same reason.

The net effect: running `index` twice over an unchanged file is idempotent at the row
level, same `doc_id`, same final `chunk_id` set, same content, not merely idempotent at
the "doesn't error" level.

## 5. Indexes

Three, all created by `ensure_schema()` with `CREATE INDEX IF NOT EXISTS`:

| Index | Table.column | Kind | Backs |
| --- | --- | --- | --- |
| `index_entries_keyword_tsv_gin_idx` | `index_entries.keyword_tsv` | GIN | The keyword branch of hybrid search. |
| `index_entries_embedding_hnsw_idx` | `index_entries.embedding` | HNSW, `vector_cosine_ops` | The vector branch of hybrid search. |
| `documents_summary_embedding_hnsw_idx` | `documents.summary_embedding` | HNSW, `vector_cosine_ops` | `search_documents()`, the document lane. |

The two HNSW indexes are on separate tables by construction, not by a `granularity`
filter: a document profile is never a candidate row in a chunk search, so there is no
shared index for a query to accidentally rank it against. `vector_cosine_ops` specifically
because OpenAI's embeddings are documented as L2-normalised, making cosine and
inner-product ranking agree; cosine is just the more familiar of the two equivalent
choices.

## 6. Extension and connection setup

- `CREATE EXTENSION IF NOT EXISTS vector` runs first in `ensure_schema()`, inside the
  same transaction as the table/column/index DDL, since the `ALTER TABLE` statements that
  add `vector(N)` columns need the type to already exist.
- `register_vector(conn)` has to run on every connection before a `vector` column will
  accept a plain Python list rather than needing manual casting. `connect()` calls it
  eagerly and swallows the failure (`psycopg.ProgrammingError`) on a brand-new database
  where the extension doesn't exist yet; `ensure_schema()` calls it again at the end,
  once the extension is guaranteed to exist.
- Both tables' DDL is `CREATE TABLE IF NOT EXISTS`, which is a no-op against a table that
  already exists. Columns added to `documents` after it was first created
  (`page_count`, `doc_summary`, `summary_embedding`, `summary_embedding_model`) each also
  get an `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, since only that reaches a database
  that already has the table. Any future column follows the same two-statement pattern:
  one entry in the `CREATE TABLE` for a fresh database, one `ALTER` for every database
  that already has the table.

## 7. Changing `embedding_dimensions`

`vector(N)` is generated into the DDL text as an f-string at `ensure_schema()` call time,
not stored as a runtime-configurable value in the database. Changing
`settings.embedding_dimensions` only changes what a *fresh* database's columns are
created as; `CREATE TABLE IF NOT EXISTS` does nothing to an existing table's column
type. An existing database that needs a new dimension has to have its `embedding` and
`summary_embedding` columns dropped and recreated by hand, there is no automatic
migration, and this is safe only because the old vectors aren't portable across a
dimension change anyway (they'd need re-embedding regardless of whether the column
survived).

## 8. What's deliberately not here

- The actual ranking SQL (RRF fusion, the keyword and vector CTEs, context expansion,
  the document lane) is `docs/search.md`'s territory, since it's a read-side concern
  layered on top of this schema, not part of the schema itself.
- Why Postgres is the only backend, why document profiles are a separate table rather
  than a `granularity='document'` row, and the BM25-local-JSON backend this replaced are
  `docs/architecture.md`'s territory.
