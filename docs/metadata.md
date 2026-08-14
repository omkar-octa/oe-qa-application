# Metadata

## Project

| | |
| --- | --- |
| Name | octo_template |
| Purpose | Question answering over a PDF knowledge base, with file and page citations |
| Owner | sam-everson@phoenixs.co.uk |
| Entry point | `src/main.py` (`ingest`, `ask`); `src/api.py` (`POST /ask`, via uvicorn) |
| Verified on | Python 3.14.0, Windows 11 |
| Version control | Git, branch `master`. Most of the work described here is still uncommitted against the initial commit |

## Dependencies

Declared in `requirements.txt`. Versions are what is currently installed in `.venv`.

| Package | Version | Used for |
| --- | --- | --- |
| pydantic | 2.13.4 | Data models in `models/documents.py` |
| pydantic-settings | 2.15.0 | `Settings` in `models/config.py`, loaded from `src/.env` |
| pytest | 9.1.1 | Test runner |
| fastapi | 0.141.1 | `api.py`, `POST /ask` |
| uvicorn[standard] | 0.52.2 | ASGI server for `api.py` |
| httpx2 | 2.10.0 | Test-only: `fastapi.testclient.TestClient`'s transport in `test_api.py` |
| docling | 2.119.0 | OCR, table structure and reading order for scanned pages |
| pdf-inspector | 1.14.1 | Per-page classification and Markdown extraction (fast path) |
| pypdfium2 | 5.13.0 | PDF page rasterisation in `utils/page_renderer.py` (vision extraction path) |
| anthropic | 0.121.0 | Claude API client for `QAAgent`, `VisionExtractor`, `FigureCaptioner`, `MetadataEnhancer`, `DocumentSummarizer` |
| openai | 3.0.0 | `text-embedding-3-large` dense embeddings in `utils/embedder.py` |
| psycopg[binary] | 3.3.4 | Postgres client in `utils/postgres_store.py` |
| pgvector | 0.5.0 | `Vector` type and `register_vector` in `utils/postgres_store.py` |

Notable transitive dependencies: `torch` 2.13.0 and `docling-ibm-models` arrive via
docling, and are the reason the first OCR run is slow and large.

## Configuration

All settings live in `src/.env`, read by `models/config.py`. `src/.env.example`
documents the full set. Only `CLAUDE_API_KEY` has no default.

| Setting | Default | Notes |
| --- | --- | --- |
| `CLAUDE_API_KEY` | none, required | Secret. `.env` is gitignored, `.env.example` is not |
| `CLAUDE_MODEL` | `claude-sonnet-5` | Answering and the per-table/per-document summaries |
| `CLAUDE_VISION_MODEL` | `claude-sonnet-5` | Page transcription. Needs the high-resolution vision tier, or a 2576px render is downsampled and 8-9pt text loses strokes |
| `CLAUDE_FIGURE_MODEL` | `claude-sonnet-5` | Figure captioning. Also needs the high-resolution tier: below it, values are read but cannot be attached to the region or series they belong to (see `docs/features.md`) |
| `CLAUDE_MAX_TOKENS` | `4096` | |
| `INDEX_PATH` | `data/index.json` | Relative to `src/` |
| `TOP_K` | `5` | Default results per search, overridable per question with `--top-k` |
| `ATTACH_SEARCH_CONTEXT` | `false` | Whether `PostgresSearchIndex.search()` attaches one neighbouring chunk each side of every match; real token cost on every match regardless of need, so off unless asked for. Overridable per run with `ask --attach-context` (see `docs/search.md`) |
| `OPENAI_API_KEY` | none, required for `index`/`ask` | Read the same way `CLAUDE_API_KEY` is; falls back to the `openai` SDK's own env lookup if unset here |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI's embeddings API |
| `EMBEDDING_DIMENSIONS` | `512` | Truncated via the API's own `dimensions` parameter; also the Postgres `vector(N)` column width |
| `PG_HOST` | `localhost` | |
| `PG_PORT` | `5432` | |
| `PG_USER` | `postgres` | Matches `docker-compose.yml`; development credentials only |
| `PG_PASSWORD` | `postgres` | As above, not fit for anything beyond local use |
| `PG_DATABASE` | `embeddings` | |

## External services and models

| Service | Purpose | Notes |
| --- | --- | --- |
| Claude API | Answer generation and search keyword rewriting (`QAAgent`); optionally page transcription (`VisionExtractor`, `--extractor vision`); figure captions and misclassified-text recovery (`FigureCaptioner`, on by default, `--no-caption-figures` to opt out); per-table summaries (`MetadataEnhancer`, `--enhance-metadata`) and per-document profiles (`DocumentSummarizer`, `--summarise-documents`) | Metered per question/page/table/file as applicable. No telemetry is sent anywhere else |
| Hugging Face | Model downloads | Docling layout, table and OCR models on first OCR run (several hundred MB). Cached locally afterwards |
| OpenAI API | Dense embeddings (`text-embedding-3-large`) for `main.py index` and `ask` | Metered per embedding call. Replaced a locally-run BGE-M3 setup that measured at ~180 chars/second on ordinary laptop CPUs, see docs/architecture.md |
| Postgres and pgvector | The only vector store and search backend, behind `main.py index` and `ask` | Local container only, `docker compose up -d` |

Both downloads need network access on first use. Docling's can be pre-seeded by pointing
`DOCLING_ARTIFACTS_PATH` at an existing cache.

## Data

- `src/tests/fixtures/`: eight scientific PDFs on energy and decarbonisation topics, used
  as both the test corpus and the default `ingest` source. Committed to the repo.
- `src/data/index.json`: the generated index from the default `--extractor fast` path,
  currently 428 chunks across the eight fixtures: 384 text/table chunks (36 of them
  `kind="table"`) plus ~44 `kind="figure"`/`kind="text"` chunks from figure captioning and
  misclassified-text recovery, which run by default now (`--no-caption-figures` opts out).
  The figure/text count moves slightly run to run (these are live model calls, not a
  deterministic parse), so treat it as an approximate range rather than a fixed number.
  This copy was also built with `--enhance-metadata --summarise-documents`, so its table
  chunks carry LLM summaries and it carries all 8 document profiles; a plain
  `python main.py ingest` without those two flags will have the same ~422-428 chunks but
  no summaries or profiles. Gitignored via `src/data/` and fully regenerable.
- `src/tests/fixtures/index.json`: a committed copy of the same fully-enriched index
  (currently 428 chunks, 8 document profiles), kept in sync manually so `main.py index`
  can run straight from a fresh clone. Regenerate `src/data/index.json` as above and copy
  it over after any change to extraction, captioning, table summarisation or document
  profiling.
- `src/data/index_vision.json`: the generated index from `--extractor vision` (currently
  348 chunks -- close to but not identical to the fast-path count, since a page read as
  an image and a page parsed from PDF structure don't always split into the same number
  of chunks). Gitignored and regenerable the same way.
- `src/data/pages/`: rendered page PNGs written by `utils/page_renderer.py` for the
  vision extraction path, one subdirectory per source PDF. Reused across runs since
  rendering is deterministic; gitignored.
Chunk counts per fixture (fast-path index), for reference when a change alters
extraction. "Extracted" is text/table chunks only and is what to check after an
extraction change; "with captioning" is the default `ingest` output and moves whenever
the figure-captioning/text-recovery model call does, which is expected to be less stable
run to run than extraction itself:

| Fixture | Extracted | With captioning (default) |
| --- | --- | --- |
| 1-s2.0-S0140988325000672-main.pdf | 63 | 65 |
| 1-s2.0-S0301421525000862-main.pdf | 52 | 69 |
| 1-s2.0-S1755008425000705-main.pdf | 35 | 41 |
| 1-s2.0-S2214629624004663-main.pdf | 58 | 64 |
| 1-s2.0-S2214629626003270-main.pdf | 44 | 50 |
| A-policy-relevant-research-agenda-...-UK.pdf | 33 | 33 |
| Firm-level-optimisation-strategies-...-charging.pdf | 33 | 37 |
| s41560-025-01898-3.pdf | 66 | 69 |

Counts rose from 345 when tables stopped sharing chunks with the prose around
them: a table now always starts and ends a chunk, so a page holding one splits
into three where it used to split into one.

## Tests

193 tests, split by marker in `src/tests/pytest.ini`. The suite is changing quickly, so
treat these counts as a snapshot to re-check rather than a fixed number.

| Selection | Count | Requirements |
| --- | --- | --- |
| `pytest src/tests -m "not integration"` | 179 | None. Claude and OpenAI calls are all mocked |
| `pytest src/tests -m integration` | 14 | Real PDFs. Four of them (`test_postgres_connection.py`, `test_postgres_store_integration.py`) need the Postgres container running |

Those two selections cover everything, but `-m unit` collects only 178: `test_basic.py`'s
placeholder carries no marker, so it runs under `-m "not integration"` and is skipped by
`-m unit`. Mark it or delete it if the marker split is ever enforced.

Integration tests are slow: the extraction tests run Docling over real fixtures, which
takes minutes on first run while models download.

Known warning: pytest reports `Unknown config option:
asyncio_default_fixture_loop_scope`, because `pytest.ini` sets it but pytest-asyncio is
not installed. Harmless, and resolved either by installing the plugin or dropping the
line.

## Deployment

None as a hosted service. `api.py` runs a real HTTP server (`uvicorn api:app`), but it's
for local/dev use -- there is no container image or CI pipeline for the application
itself, and the only container is the local Postgres used for vector store work.
