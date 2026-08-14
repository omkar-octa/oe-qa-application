# octo_template

A question and answer system over a PDF knowledge base. PDFs are extracted page by
page, embedded and indexed into Postgres with pgvector for hybrid (keyword plus dense
vector) search, and queried through a Claude-powered agent that cites every claim by
source file name and page number.

Extraction is two-tier, so the slow OCR pipeline only runs where it is actually
needed: [pdf_inspector](https://github.com/firecrawl/pdf-inspector) reads the pages
that already carry a text layer, and Docling handles the pages it reports as scanned.

See [docs/flowchart.md](docs/flowchart.md) for how the pieces fit together,
[docs/features.md](docs/features.md) for what each component does,
[docs/architecture.md](docs/architecture.md) for the design decisions behind it,
[docs/search.md](docs/search.md) for how retrieval itself works (query rewriting, the
tools, RRF), [docs/postgres.md](docs/postgres.md) for the database schema and how
re-indexing avoids duplicating rows, and [docs/roadmap.md](docs/roadmap.md) for what is
built versus planned.

## Requirements

- Python 3.14 (currently verified on 3.14.0, Windows)
- Docker, for the Postgres and pgvector container `ask` and `index` both need

## Dependencies

What each external piece is actually doing:

- **[pdf_inspector](https://github.com/firecrawl/pdf-inspector)**: classifies each page as
  born-digital or scanned and extracts text and tables from the pages that already carry a
  text layer. The free, fast tier of extraction (`utils/pdf_extractor.py`), runs locally.
- **Docling**: OCR and layout-model fallback for the pages `pdf_inspector` reports as
  scanned, and also crops figure images and reads whole pages under `--extractor vision`.
  Runs locally; downloads several hundred MB of models from Hugging Face on first use.
- **Claude (Anthropic API)**: the compute engine throughout, not only the answering step.
  It answers questions (`QAAgent`, `claude_model`), transcribes pages under
  `--extractor vision`, captions figures and recovers misclassified body text
  (`claude_figure_model`), writes table summaries (`--enhance-metadata`) and document
  profiles (`--summarise-documents`, both `claude_model`), and grades eval answers
  (`scripts/run_evals.py`). Needs `CLAUDE_API_KEY`.
- **OpenAI API**: embeds every chunk (`text-embedding-3-large`, truncated to 512
  dimensions) for the dense side of hybrid search, a remote call rather than a local
  model, see
  [docs/architecture.md](docs/architecture.md#why-embeddings-are-a-remote-api-call-not-a-local-model).
  Needs `OPENAI_API_KEY`.
- **Postgres and pgvector**: the only search backend, storing chunks and running hybrid
  (keyword plus dense vector) retrieval fused with Reciprocal Rank Fusion. Runs locally via
  `docker compose up -d` (`docker-compose.yml`).

## Setup

```
pip install -r requirements.txt
```

Copy `src/.env.example` to `src/.env` and fill in `CLAUDE_API_KEY` and
`OPENAI_API_KEY` (embeddings are a remote OpenAI call, not a local model -- see
[docs/architecture.md](docs/architecture.md#why-embeddings-are-a-remote-api-call-not-a-local-model)).
Every other setting has a working default.

First-run downloads to be aware of: the first ingestion run that hits a scanned page
downloads Docling's layout, table and OCR models (several hundred MB) from Hugging Face.
Ingestion of the eight bundled fixtures needs this in full for only the one mixed
document; the rest have a text layer already.

## Running

All commands are run from `src/`, because `.env` and the default index path resolve
relative to that directory.

Build the index from the PDFs in `tests/fixtures`:

```
python main.py ingest
```

Options: `--source <dir>` to index a different folder of PDFs, `--output <path>` to change
where the index is written (default `data/index.json`), and `--image-dir <dir>` for
rendered page images (default `data/pages`). Ingesting the bundled fixtures produces 384
text/table chunks across the eight PDFs, plus figure and recovered-text chunks from
captioning (see below) unless it's turned off.

Note the plain command above already costs Claude API calls: figure captioning (below) runs
by default, one call per page with a plausible reason to hold an image. Pass
`--no-caption-figures` for a free, offline, fully repeatable run.

| Option | Cost |
| --- | --- |
| `--extractor vision` | one call per page, instead of parsing the PDF structure; off by default |
| `--no-caption-figures` | turns off the one call per candidate page (cheaply pre-filtered, not every page) that adds a chunk per figure, saves real cropped figure images via Docling, and recovers body text a layout model misclassified as a picture; **on by default**, works with either extractor |
| `--enhance-metadata` | one call per **table** chunk, 36 for the bundled fixtures; off by default |
| `--summarise-documents` | one call per file, 8 for the bundled fixtures; off by default |

`--summarise-documents` writes a short profile of each whole document: what it covers, how
it is structured, and where its numbers are. The profile is shown alongside every search
result from its file so the model can tell whether it is in the right document and whether
to pull the whole thing. Profiles are never ranked against extracts and are never citable.

`tests/fixtures/index.json` is a committed, fully-enriched extraction index for the eight
bundled PDFs (captioning plus `--enhance-metadata --summarise-documents`, 428 chunks), so
`index` can be run straight from a fresh clone without an `ingest` pass first:

```
python main.py index --index tests/fixtures/index.json
```

`ingest` and `index` are independent steps; `index` only requires that the index file at
`--index` already exists, not that `ingest` ran in the same session. Regenerate this fixture
(same command as above, with `--output tests/fixtures/index.json`) after changing extraction,
figure-captioning or metadata-enhancement code, since it reflects whatever that code produced
at the time it was last written, not the code as it stands now.

## Postgres and pgvector

`ask` and `index` both need the container running. Start it:

```
docker compose up -d
```

That starts `pgvector/pgvector:pg16` on port 5432 with database `embeddings`. Check it
from `src/`:

```
python scripts/check_postgres.py
```

Then embed the extraction index from `ingest` and load it into Postgres:

```
python main.py index
```

This is a deliberate second step rather than part of `ingest`, because extraction is free
and offline while this needs the database and the embedding model. It creates the schema if
absent, embeds every chunk, and replaces each document's rows in one transaction, so it is
safe to re-run. Options: `--index <path>`, `--source <dir>` (where the PDFs live, needed to
hash each file for its `doc_id`), `--image-dir <dir>`.

If the index was built with `--summarise-documents`, this also embeds each document profile
into the `documents` table, where it gets its own vector index and never competes with
extracts for a slot in the results. Re-running without profiles leaves any already stored
untouched, so a plain `ingest` followed by `index` does not lose them.

Embedding is a remote call to OpenAI's API (`text-embedding-3-large`, truncated to 512
dimensions), not a locally-run model: the whole 419-chunk, 8-file bundled corpus embeds in
around 11 seconds. This replaced an earlier local BGE-M3 setup, which measured at roughly
180 characters/second on ordinary laptop CPUs -- fine for a single query, but turning a
bulk `index` run into the better part of an hour. See
[docs/architecture.md](docs/architecture.md#why-embeddings-are-a-remote-api-call-not-a-local-model)
for the measurement.

Now ask a question:

```
python main.py ask "What does the research say about industrial decarbonisation in the UK?"
```

Options: `--top-k <n>`, `--debug` (logs each search query, tool call, and LLM round trip
to stderr), and `--attach-context` (attaches one neighbouring chunk each side of every
search match, for a completed sentence or table header, at the cost of extra tokens on
every match whether or not it needed it -- off by default; set
`ATTACH_SEARCH_CONTEXT=true` in `.env` to change the default instead of passing the flag
every time). Retrieval is hybrid keyword-plus-vector search fused with Reciprocal Rank
Fusion (`PostgresSearchIndex`, see [docs/search.md](docs/search.md)); there is no other
backend to choose. With context expansion off (the default), the agent leans on its
`read_source` escalation tool (whole-document and whole-page reads) to recover a cut-off
extract instead of getting the completing text for free.

## API

`api.py` wraps the same `QAAgent` behind an HTTP endpoint, building the embedder and
search index once at startup instead of once per question:

```
uvicorn api:app --reload
```

```
POST /ask
{"question": "What does the research say about industrial decarbonisation in the UK?"}
```

Returns `{"answer": "..."}`. A blank `question` is rejected with `422` before it
reaches Claude. A ready-made Postman collection and local environment live in
[postman/](postman/) -- import both, select the "octo_template local" environment, and
run the requests against the server started above.

## Testing

Unit tests mock the Claude API, the embedding model and the Postgres connection, so
they need no key, no network and no container (171 tests):

```
pytest src/tests -m "not integration"
```

Integration tests run real PDFs through both extractors, and are slow on first run
because of the Docling model downloads (14 tests):

```
pytest src/tests -m integration
```

Five of the integration tests exercise the Postgres and pgvector container and fail
unless `docker compose up -d` is running.

## Evaluation

`tests/fixtures/eval_questions.json` is a hand-maintained question bank (80 questions,
tagged `single`, `multi-hop`, `compute`, `table`, `figure`, `footnote`, `cross-doc` and
`trap`) over the bundled fixtures, each with its own expected answer and citation, plus a
short-name-to-real-file-name table used to abbreviate citations (e.g. "Burlinson p1").

`scripts/run_evals.py` reads that JSON and runs it end to end against a live `QAAgent`,
grading each answer against the question's own reference answer with Claude since these
are open-ended prose answers that exact-match scoring can't score:

```
python scripts/run_evals.py
```

Needs Postgres running with an index already loaded (`python main.py index`) and costs
two Claude calls per question: one to answer, one to grade. Questions run concurrently,
`--workers` (default 5). Each question is graded pass, partial or fail; results are
written to `data/eval_results.json` and summarised by tag and overall on stdout, with every
non-passing question listed by ID and reason. Narrow a run with `--tag figure`, `--id C5`,
or `--limit 10` (all repeatable except `--limit`).

## Layout

```
src/
  main.py                     CLI: ingest, index and ask
  api.py                      FastAPI service: POST /ask
  models/                     pydantic settings and shared data models
  utils/                      extraction, indexing, embedding, QA agent
  prompts/qa_system.prompt    QA agent system prompt
  scripts/                    standalone checks and exploratory notebooks
  tests/                      unit, integration, PDF fixtures, a pre-built fixture index
                              and the eval question bank
docs/                         features, architecture, roadmap, metadata
postman/                      Postman collection and environment for api.py
```

The layout follows the convention in
[.claude/skills/python-project-structure](.claude/skills/python-project-structure/SKILL.md).
