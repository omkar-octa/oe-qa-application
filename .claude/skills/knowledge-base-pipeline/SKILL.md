---
name: knowledge-base-pipeline
description: Working rules for this repo's PDF knowledge base pipeline - the fast and vision extraction paths, page numbering conventions, the DocumentChunk and IndexEntry schemas, the retriever protocol behind QAAgent, the Postgres/pgvector store, the OpenAI-based embedder, and the environment traps that break Docling and the CLI on Windows. Tagged pdf, extraction, ocr, retrieval, docling, pgvector, claude, openai.
when_to_use: Use when changing extraction, chunking, indexing, retrieval, the QA agent or the Postgres store in this repo; when adding a new extractor or search backend; when a page number, citation or chunk count looks wrong; or when running the ingest/ask CLI, the API, or the tests.
---

# Knowledge base pipeline

How to work on this repo's ingestion and question answering without breaking the things
that are easy to break. Reference documentation lives in `docs/features.md` (what each
component does), `docs/architecture.md` (why), and `docs/flowchart.md` (the flows as
diagrams). This skill is the procedural part.

## Running things

- **All CLI commands run from `src/`.** `Settings` uses `env_file=".env"` and
  `index_path` defaults to `data/index.json`, both relative to the working directory.
  Running `python src/main.py` from the repository root fails to find the API key.
- The pipeline is three ordered subcommands: `ingest` (extract to `data/index.json`),
  `index` (embed and upsert into Postgres), `ask` (queries Postgres; there is no other
  backend to select). `index.json` is regenerable and gitignored, so rebuild rather than
  hand-editing it.
- `uvicorn api:app --reload` from `src/` serves `POST /ask` against the same `QAAgent`.
- Tests run from the repository root: `pytest src/tests -m "not integration"` for the
  fast, fully mocked set, `-m integration` for the slow one. Mark new tests `unit` or
  `integration`; only `test_basic.py`'s placeholder is unmarked, so `-m unit` collects one
  fewer test than `-m "not integration"`.
- Five integration tests need `docker compose up -d` for Postgres and pgvector. Failures
  there mean the container is down, not that the code is broken.

### What costs money

Several paths hit the Claude API per unit of content rather than per question:

| Command | Cost |
| --- | --- |
| `ingest --enhance-metadata` | one call per table chunk (36 for the bundled fixtures), not per chunk -- see below |
| `ingest --extractor vision` | one call per page, every page, scanned or not |
| `ingest` figure captioning | one call per *candidate* page (`likely_figure_pages()` pre-filter, ~42% of pages on the fixtures), on either extractor |
| `scripts/test_tool_use.py` | real tool-use loops against the live API |

`--enhance-metadata` and `--extractor vision` must never become a default, an example
command, or an unmocked test -- they trade cost for accuracy, and that trade is the
caller's to make. Figure captioning is the one exception: it runs on `ingest` **by
default** now (`--no-caption-figures` opts out), because it also recovers text a page's
layout-detection stage can silently drop, not just chart values -- see
`docs/architecture.md#why-figure-captioning-is-the-one-exception`. Unmocked unit tests
still must not trigger it: pass `caption_figures=False` (or use `--no-caption-figures` on
the CLI) in any test not specifically exercising captioning.

`python main.py index` and `ask` cost no Claude calls, but both need Postgres up and
`OPENAI_API_KEY` set (embeddings are an OpenAI API call, not a local model -- see
`## Embeddings` below for why that changed).

## Extraction: two paths, not one

`--extractor fast` (the default) is the two-tier path, and the split is not negotiable:

- `PdfExtractor` (pdf_inspector) reads pages that already have a text layer. No OCR, no
  ML models.
- `DocumentExtractor` (Docling) is the slow fallback, called **only** for the pages
  `PdfExtractor.pages_needing_ocr()` reports, via `extract_chunks_for_pages()`.

`--extractor vision` is a separate first stage, not a fallback the fast path escalates to.
`VisionExtractor` renders every page with `PageRenderer` and has Claude transcribe it, so
scanned and born-digital pages cost the same. It reuses `PdfExtractor._page_to_chunks`, so
both paths must keep producing identically shaped `DocumentChunk` objects.

Rules:

- **Never classify text-layer presence per document.** A document-wide average is what
  caused pages 1 to 7 of `1-s2.0-S0140988325000672-main.pdf` to be silently dropped by an
  earlier PyMuPDF implementation. Classification is per page. `test_mixed_document_reports_
  its_scanned_pages_needing_ocr` pins that fixture's expected page list; if it fails, take
  it seriously rather than updating the expectation.
- Build `DocumentExtractor` lazily. Constructing it loads Docling's models, so `ingest`
  only does so when some PDF actually needs OCR, and `ask` never imports it at all. Keep
  the extractor imports inside `main.py`'s subcommand functions.
- Rendered pages in `data/pages/` are a deterministic cache and are reused across runs.
  Do not invalidate them casually; re-rendering is the slow part of a vision re-run.
- Do not reintroduce PyMuPDF. It was deliberately removed. `pypdfium2` is the renderer.

## Page numbers and chunk IDs

Page numbers are citations, so mistakes here are user-visible and quiet.

- `DocumentChunk.page_number` is **1-indexed**; `0` means unknown.
- pdf_inspector is inconsistent with itself: `PageMarkdown.page` is **0-indexed**, while
  `PagesExtractionResult.pages_needing_ocr` is **1-indexed**. Convert the former with
  `+ 1`, pass the latter through. Check its shipped `.pyi` stub rather than guessing.
- Docling's `page_range=(start, end)` preserves absolute page numbers, which is why
  partial conversions can be merged by sorting on `page_number`. Do not add offsets.
- Chunk IDs are assigned per conversion call, so multiple `page_range` passes collide at
  `c0`. Renumber over the merged list, as `extract_chunks_for_pages` does.
- Two ID schemes coexist. `DocumentChunk.chunk_id` is `file_name`-based;
  `IndexEntry.chunk_id` is `{doc_id}-p{page:03d}-c{seq:02d}`, where `doc_id` is the file's
  sha256 prefix. Do not mix them. The `IndexEntry` form is the one an LLM has to emit
  verbatim for citations, and the one the database keys on.

## Chunking rules

- A chunk never spans two pages. Flush on a page change and on a section heading.
- `max_chars` (default 3000) is a **hard bound**, not a target. A single line longer than
  the limit, typically a wide Markdown table row, must still be split. There are tests for
  this at both the unit and integration level; `test_max_chars_is_a_hard_bound_on_real_
  documents` runs against real fixtures.
- `kind` is one of `"text"`, `"table"` or `"figure"`. Anything deciding how to embed,
  summarise or display a chunk branches on `kind`, never on sniffing the text.
- Tables become their own chunks (`kind="table"`). Both PDF extractors skip picture items
  entirely, so figure chunks come only from `FigureCaptioner`, which runs by default on
  `ingest` (see "Keeping expensive work opt-in" below for the one exception this is), and
  they use an `#f` discriminator in the chunk ID so they cannot collide with the `#c` text
  chunks numbered over the same pages. `FigureCaptioner` can also emit `kind="text"` chunks
  when it recognises a picture-classified region is actually body text a layout model
  dropped (`classification: "text"` in its schema) -- those merge as ordinary text, not as
  figures, and `main.py::_drop_duplicate_recovered_text` guards against them duplicating
  text the primary extractor already got right.
- Never run `likely_figure_pages()` or `extract_figure_images()` over every page as a
  "simplification" -- the whole point of the first is to bound how many pages the second
  (Docling's layout model) and `FigureCaptioner` (a Claude call) ever touch. If detection
  fails, fall back to captioning every page (as `_caption_figures` does), never to cropping
  every page: that's the one cost this pre-filter exists to avoid paying unconditionally.
- `extract_figure_images()` builds its own converter (`do_ocr=False`,
  `do_table_structure=False`, `generate_page_images=True`), not `DocumentExtractor`'s.
  Picture detection comes from the layout model regardless of either OCR option, so don't
  route this through `DocumentExtractor` just to reuse a converter -- its default one is
  configured for a different job and doesn't set `generate_page_images`, which
  `PictureItem.get_image()` needs something to crop from.
- A crop's real size is what tells furniture (a logo, an ORCID icon) apart from a genuine
  figure -- not its classification, since Docling's layout model doesn't classify at all.
  `_MIN_PICTURE_AREA_FRACTION` (2%) is verified against a real fixture page, not guessed;
  don't loosen it without re-checking against `test_extract_figure_images_drops_
  furniture_sized_crops`.
- Figure crops from `extract_figure_images()` are not yet bound to a specific
  `FigureCaptioner`-described figure, and must not be assumed to be: a page can hold
  several of each, found in orders that need not align. Treat them as a standalone
  artifact on disk until that binding is deliberately solved.

## Changing retrieval

`QAAgent` requires exactly one method:

```python
search(query: str, top_k: int) -> list[SearchResult]
```

Implement that and swap what `index` is in `main.py` and `api.py`. Do not modify the agent
loop to accommodate a new backend. Related invariants:

- `read_source` additionally wants `document()`, `page()` and `files()`. It checks with
  `hasattr` first and degrades to a plain message if they are absent, so a new backend is
  allowed to omit them, but it loses the escalation tool. Follow that same
  check-then-degrade pattern when adding further optional accessors.
- Ranking happens on the matched chunk only. `context_before` and `context_after` on
  `SearchResult` are attached after scoring and must never influence ranking or displace a
  genuine match. On `PostgresSearchIndex` this attachment is opt-in
  (`attach_context`/`settings.attach_search_context`, off by default): it costs real
  tokens on every match whether or not that match needed it, so don't make it
  unconditional again without re-reading `docs/search.md`'s "Unconditional per match, and
  off by default because of it".
- `SearchIndex` (the extraction-time JSON, `data/index.json`) persists content only, never
  search statistics -- it feeds `main.py index`'s embed-and-upsert step, not question
  answering directly; nothing reads it at `ask`/`api.py` time.
- Document profiles are **never** ranked against chunks. Do not tokenise
  `SearchIndex.documents` for keyword ranking, and do not store one as a
  `granularity='document'` row in `index_entries` even though `Granularity.DOCUMENT`
  exists: a profile mentions every topic its document touches, so it matches most queries
  weakly and displaces the chunk holding the actual figure. They ride on
  `SearchResult.doc_summary` for display, and are searchable only through the separate
  `search_documents()` lane. See
  [architecture.md](../../../docs/architecture.md#why-document-profiles-are-not-chunks).

## The Postgres store

`utils/postgres_store.py` owns schema and writes; `utils/postgres_search_index.py` owns
reads. This is the only retrieval backend -- `main.py`'s `index` and `ask`, and `api.py`,
all build `PostgresSearchIndex` directly. An earlier JSON-backed `LexicalSearchIndex`
(BM25 keyword search, selectable via `--backend`) existed and was deleted once this
backend covered everything it did.

- `ensure_schema()` is idempotent DDL and is meant to be called every time. There is no
  migration tool by choice, so schema changes go in that DDL and must stay re-runnable.
- A new column needs **two** statements, not one: an entry in the `CREATE TABLE ... IF NOT
  EXISTS` for a fresh database, and an `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for every
  database that already has the table, where the `CREATE` is a silent no-op. Any ALTER
  naming a `vector` type must run after `CREATE EXTENSION` in the same transaction.
- The `documents` row is upserted with `coalesce(excluded.col, documents.col)` on the
  summary columns, so a re-index that carries no document profile does not blank one that
  is already stored. Keep that shape when adding optional per-document columns.
- Call `register_vector()` on every connection, or vector columns will not accept Python
  lists. `connect()` does it and deliberately swallows the failure on a database where the
  extension does not exist yet, because `ensure_schema()` creates it and re-registers.
- `upsert_document()` deletes the document's existing rows before inserting. Keep it that
  way: `chunk_id`'s sequence number is positional, so re-extraction shifts it and a
  primary-key upsert would leave orphaned rows behind.
- Embed before you upsert. `upsert_document()` raises on `embedding=None` rather than
  writing a null vector, and that check should stay.
- `keyword_tsv` is a **generated** column. Postgres maintains it; never write to it, and
  keep `keyword_text` as the single source of truth.
- The vector column is `vector(N)`, `N` inlined from `settings.embedding_dimensions`
  (currently 512) at DDL time via an f-string, the same way `_RRF_K` is inlined into
  `postgres_search_index.py`'s query. `ensure_schema()`'s `CREATE TABLE IF NOT EXISTS`
  does not change an existing table's column type, so changing the setting only takes
  effect on a fresh database; an existing one needs the vector columns dropped and
  recreated by hand.
- Keyword search is Postgres FTS, ranked with `ts_rank_cd` (cover density), fused with
  the vector branch by rank (RRF), not by score -- see `docs/search.md` section 3 for the
  full formula. Note that `to_tsvector` keeps the sign in numeric tokens: `-0.59%` indexes
  as `-0.59`, so a query for `0.59` will not match it.
- `PostgresSearchIndex` fuses the keyword and vector branches by **rank** (RRF, `_RRF_K =
  60`), not by score. Do not "simplify" it into a weighted sum of `ts_rank_cd` and cosine
  distance: those scales are incompatible, which is the whole reason for fusing by rank.
- It returns `DocumentChunk`, not `IndexEntry`, because that is what `QAAgent` consumes.
  `display_text` becomes `chunk.text` and `kind` is recovered from `element_types`.
- Its embedder is duck-typed on `embed_texts`. Keep it that way so the module never imports
  `Embedder` concretely and a test double can be injected instead.
- It implements `document()`/`page()`/`files()` (exact `file_name` lookups against
  `index_entries`), so `read_source` works against it. It still attaches no
  neighbouring-chunk context, so swapping it in silently disables the match-context
  window. Close that gap before making it the default rather than after.

## Embeddings

- `Embedder` (`utils/embedder.py`) calls OpenAI's API (`text-embedding-3-large`,
  truncated to `settings.embedding_dimensions` via the API's own `dimensions`
  parameter). There is no local model, no daemon, no resident process, and no reason
  to build one back in: a remote call has no load cost to amortise. Every caller --
  `main.py index`, `PostgresSearchIndex`, tests -- just constructs an `Embedder`
  directly.
- Do not reintroduce a locally-run embedding model without re-checking the numbers
  first. This repo already tried one (BGE-M3, via FlagEmbedding): measured at ~180
  characters/second on an ordinary laptop CPU, it took a bulk `index` run of this
  repo's own corpus from 11 seconds (OpenAI) to an estimated 63 minutes. See
  docs/architecture.md for the measurement before assuming a local model would be
  fine.
- Needs `OPENAI_API_KEY` in `src/.env` or the real OS environment. Mock the client in
  unit tests (`Embedder(client=...)`), the same pattern used for `anthropic.Anthropic`
  elsewhere in this repo; no unit test should make a real network call.
- Only the dense vector is requested. OpenAI's embeddings API has no sparse/lexical
  output the way BGE-M3 did, so hybrid search's keyword side stays Postgres FTS
  (`ts_rank_cd`) rather than a model-provided sparse vector.
- `DocumentChunk.summary` is the LLM gloss from `MetadataEnhancer` and is `None` unless
  `--enhance-metadata` ran. Anything reading it must tolerate `None`.
- `enhance_chunks()` defaults to summarising `kind="table"` only
  (`DEFAULT_SUMMARISED_KINDS`), because `index_entry_builder.py` is the only reader of
  `summary` and only reads it for tables. Do not widen the default to cover text or figure
  chunks without also giving something a reason to read their `summary`; that is what
  turns the flag back into 384 calls that produce a value nothing consumes. Pass
  `kinds=(...)` to summarise other kinds for a specific caller instead.
- For tables, `summary` is the better embedding input and the Markdown is the better
  keyword input. `IndexEntry` encodes exactly that split across `embed_text`,
  `keyword_text` and `display_text`; `index_entry_builder.py` is where the choice is made.

## Keeping expensive work opt-in

Extraction on the fast path is free, offline and repeatable. Anything that costs API
calls, needs network, or is non-deterministic stays behind an explicit flag, the way
`MetadataEnhancer` sits behind `--enhance-metadata` and vision sits behind
`--extractor vision`. Do not make an extractor call the Claude API as a side effect of the
default path. Methods that add derived data return copies via `model_copy(update={...})`
and leave their inputs unmodified; `Embedder.embed_chunks` and
`MetadataEnhancer.enhance_chunks` both follow this, and new ones should.

Figure captioning is the one deliberate exception to "opt-in by default": it runs on
`ingest` unless `--no-caption-figures` is passed. See
`docs/architecture.md#why-figure-captioning-is-the-one-exception` before assuming a new
expensive step should follow this precedent rather than the opt-in default -- the
justification there is specific (it closes a confirmed silent-data-loss bug, not just an
accuracy/cost trade), not a general licence to default new Claude calls on.

## Environment traps

- **Set `TORCHDYNAMO_DISABLE=1` before importing Docling.** `document_extractor.py` does
  this at module top. Any new module or script that imports Docling, including notebooks in
  `scripts/`, needs it too, or RapidOCR fails looking for an MSVC compiler (`cl is not
  found`).
- **Windows console encoding.** `main.py` reconfigures stdout to UTF-8 with replacement.
  Any new entry point printing model output needs the same, or it crashes on a successful
  answer.
- `tests/conftest.py` sets a placeholder `CLAUDE_API_KEY` so importing `models.config`
  works with no `.env`. Unit tests must not depend on a real key or on `src/.env` existing.

## Claude API usage

- Three models, three settings, and the call shapes are **not** interchangeable. Pick by job
  rather than copying whichever call is nearest:

| Setting | Job | Call shape |
| --- | --- | --- |
| `claude_model` (`claude-sonnet-5`) | answering, chunk and document summaries | adaptive thinking on by default |
| `claude_vision_model` (`claude-sonnet-5`) | page transcription | needs the high-resolution tier, uses `output_config.effort`, thinking left on |
| `claude_figure_model` (`claude-sonnet-5`) | figure captioning | no `effort`, no `thinking`, structured output via `output_config.format` |

- Both vision settings need the high-resolution tier, which is why they are not pointed at a
  cheaper model. Below it the render is downsampled to ~1568px: page transcription loses
  strokes on 8-9pt body text, and figure captioning still reads the values but can no longer
  attach them to the region or series they belong to, which is the whole point of the chunk.
  `claude_figure_model` deliberately omits `effort` and `thinking` anyway, so that swapping
  it down for a cost-sensitive run stays a one-line change rather than a call-shape change.

- Never pass `temperature` or `top_p`; these models reject them. Pass no `thinking`
  parameter where adaptive thinking is available.
- Do not disable thinking on the vision path. With thinking off the model leaks stray tags
  into visible output, which here would land silently in the index. On the figure path the
  JSON schema is what prevents the same leak.
- Check `stop_reason == "refusal"` before reading response content; a refusal returns HTTP
  200 with empty content. Prefer degrading that unit of work (one page transcribes empty)
  over failing the whole document.
- Keep the tool loop bounded. `MAX_TOOL_ITERATIONS` (currently 8) exists so a model that
  keeps rephrasing the same query cannot loop indefinitely; it allows for `read_source`
  escalations on top of the search that motivated them.
- Inject the client for tests: `QAAgent(index, client=...)`, and override FastAPI's
  `get_agent()` dependency in `api.py` tests. No unit test should need an API key or
  network.

## Keeping docs honest

When behaviour, dependencies or commands change, update the affected file:
`docs/features.md` for what components do, `docs/architecture.md` for design decisions and
traps, `docs/flowchart.md` for the diagrams, `docs/metadata.md` for dependencies, settings,
chunk counts and test counts, `docs/roadmap.md` for build status, and `README.md` for setup
or commands. `docs/metadata.md` states concrete numbers, so re-check them rather than
leaving stale figures: test counts move whenever tests are added, and chunk counts move
whenever extraction changes.
