# Features

What each component does. For the reasoning behind the design and the traps it works
around, see [architecture.md](architecture.md). For the same flows drawn as diagrams,
see [flowchart.md](flowchart.md). For how retrieval itself works in depth (query
rewriting, the tools, RRF, the schema), see [search.md](search.md). For build status,
see [roadmap.md](roadmap.md).

## Architecture

```
System 1a: extraction                 System 1b: indexing              System 2: QA
PdfExtractor (pdf_inspector) ---.
                                 >--- DocumentChunk[] ---.
DocumentExtractor (Docling) ----'                        >--- index.json
  only for pages needing OCR                                    |
                                     main.py index, Embedder (OpenAI API)
                                                  v
                                    Postgres + pgvector, PostgresSearchIndex ---> QAAgent
                                       ask / api.py (hybrid search, RRF)         (claude-sonnet-5)
```

Optional per-file stages, all costing Claude calls: `FigureCaptioner` (a chunk per figure,
plus recovering any body text a layout model misclassified as a picture -- on by default,
`--no-caption-figures` opts out), `MetadataEnhancer` (a summary per table chunk, off by
default) and `DocumentSummarizer` (a profile per document, never ranked and never cited,
off by default).

## Ingestion

`--extractor fast` (the default) is two-tier, so the slow OCR pipeline only runs on the
pages that actually need it:

- `utils/pdf_extractor.py`: `PdfExtractor` is the fast path, built on
  [pdf_inspector](https://github.com/firecrawl/pdf-inspector) (Rust, no OCR and no ML
  models). It classifies **per page** rather than per document, so a PDF mixing
  born-digital and scanned pages is handled correctly: `extract_chunks()` returns
  chunks only for the pages that have usable text, and `pages_needing_ocr()` reports
  the rest as 1-indexed page numbers for the caller to route elsewhere. Headings come
  from pdf_inspector's native Markdown output (`#` and `##` lines), which is more
  reliable than a font-size heuristic. `max_chars` is a hard bound on chunk length,
  including single unbroken lines such as wide table rows. A table is kept whole: it
  gets its own chunk, never shares one with surrounding prose, keeps the `Table N`
  caption that introduces it, and if it is too wide for `max_chars` it is divided into
  parts that each repeat the caption and header row. Each chunk is tagged `kind="table"`
  when it holds two or more Markdown table rows, `"text"` otherwise.
- `utils/document_extractor.py`: `DocumentExtractor` wraps Docling with OCR (Docling's
  bundled RapidOCR engine) and table structure recovery enabled, so scanned pages and
  tables in multi-column scientific PDFs are extracted in reading order.
  `extract_chunks_for_pages()` restricts a conversion to specific 1-indexed pages,
  grouped into contiguous runs via Docling's `page_range`, so `ingest` only pays for
  the OCR and layout models on the pages `PdfExtractor` flagged. Tables are serialised
  to Markdown as their own chunks (`kind="table"`), split by the same header-repeating
  rule `PdfExtractor` uses so Docling's output honours `max_chars` too; pictures are
  skipped.
- One of the eight fixtures is a real mixed document:
  `1-s2.0-S0140988325000672-main.pdf` has no text layer on pages 1 to 7 and a clean one
  on pages 8 to 16, so its chunks come from both extractors, merged and sorted by page.
- `data/index.json` (`models.documents.SearchIndex`) is a pure content store: chunks plus
  any document profiles, no search statistics. `main.py index` reads it to embed and
  upsert into Postgres; nothing reads it at question-answering time any more, since
  `ask`/`api.py` query Postgres directly.

## Vision extraction

`--extractor vision` is an alternative first stage, not a fallback the fast path picks
up automatically -- it reads a picture of the page instead of the PDF's internal
structure, so scanned and born-digital pages cost the same: one Claude vision call each.

- `utils/page_renderer.py`: `PageRenderer` rasterises PDF pages to PNG with
  [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) (a wheel, no system Poppler
  install needed), scaling each page so its long edge lands at `MAX_LONG_EDGE_PX`
  (2576px, Claude's own limit before it downsamples) with a floor of `MIN_LONG_EDGE_PX`
  (1400px, below which 8-9pt two-column body text starts losing strokes). Images are
  written to `data/pages/<stem>/pNNNN.png` and reused on a repeated ingest, since
  rendering is deterministic and the slow part of a re-run.
- `utils/vision_extractor.py`: `VisionExtractor` renders every page of a PDF, transcribes
  each concurrently (`ThreadPoolExecutor`, 4 workers by default) with a single Claude
  call per page (`effort="low"`, thinking left on rather than disabled -- with thinking
  off the model is prone to leaking stray tags into visible output, which here would be
  invisible until it hit the index), and feeds the resulting Markdown through the same
  chunker `PdfExtractor` uses (`PdfExtractor._page_to_chunks`) so both paths produce
  identical `DocumentChunk` shapes. A `refusal` stop reason yields an empty transcript
  for that page rather than failing the whole document. The system prompt lives in
  `prompts/page_transcription.prompt`.
- Wired into the CLI as `python main.py ingest --extractor vision [--image-dir <dir>]`.

## Question answering

- `utils/qa_agent.py`: `QAAgent` runs an agentic retrieval loop rather than a single
  fixed search, with two tools. `search_knowledge_base` lets Claude decide the keywords
  itself, rewriting the question into search terms rather than searching on the question
  verbatim, and it may call the tool more than once with refined keywords before
  answering. Each call returns the top matching chunks as numbered extracts with file and
  page provenance; when `PostgresSearchIndex`'s context expansion is turned on (off by
  default, see below), the matched text is delimited with `>>> MATCH:` and its
  neighbouring chunks labelled `...context` on either side. Claude
  cites each claim as `(file.pdf, p. 4)` against the match, not the context, using only
  what the tools returned. `read_source` is a second, escalation tool for when a result
  looks decisive but incomplete -- a whole document or a single page, for footnotes,
  figures, or facts spread across a paper -- that degrades to a plain message rather than
  erroring if the underlying index doesn't support it (see the retriever protocol in
  [architecture.md](architecture.md#the-retriever-protocol)). The loop is capped at
  `MAX_TOOL_ITERATIONS` round trips per question, and a `refusal` stop reason is handled
  before the response content is read. When document profiles exist, each result set is
  prefaced by one line per distinct source document, printed once however many hits that
  document contributed and marked as not quotable or citable, with the ranked list
  following unchanged in score order. When a search returns nothing at all, a backend that
  offers `search_documents()` supplies the closest-matching documents by profile so the
  model has a `read_source` target instead of only a dead end; one that doesn't degrades
  to the plain message, the same `hasattr` pattern `read_source` uses. `_seen_chunk_ids`
  tracks every chunk ID already sent to the model within the current `answer()` call and
  is reset at the start of the next one, so a rephrased `search_knowledge_base` query that
  re-surfaces a chunk already in the conversation sends it once rather than padding the
  context with the same evidence again; numbered extracts (`[1]`, `[2]`, ...) are
  renumbered from 1 over the surviving fresh chunks each call, local to that tool result
  rather than a running count. If a search's chunks have all already been seen, Claude
  gets `DUPLICATE_RESULTS_MESSAGE` pointing it back at the earlier results, distinct from
  the "no matching extracts, try different keywords" message a genuinely empty search
  returns. The system prompt lives in `prompts/qa_system.prompt` and is domain-agnostic.

## Embeddings and vector storage

Wired into the CLI as the only backend: `python main.py index` embeds and loads,
`python main.py ask` (and `api.py`) query it directly.

- `utils/embedder.py`: `Embedder` generates dense embeddings via OpenAI's API
  (`text-embedding-3-large`, truncated to `settings.embedding_dimensions` = 512 via the
  API's own `dimensions` parameter). Replaced an earlier locally-run BGE-M3 setup
  (`FlagEmbedding`, now removed from `requirements.txt`): measured at ~180 chars/second
  on an ordinary laptop CPU, it turned a bulk embed of this repo's own corpus into a
  ~63-minute job. A remote call has no model-load cost to amortise, so this is what
  every caller uses directly -- no resident daemon, no separate in-process variant. See
  [architecture.md](architecture.md#why-embeddings-are-a-remote-api-call-not-a-local-model).
  `embed_chunks()` returns copies with `embedding` populated and leaves the inputs
  untouched. Needs `OPENAI_API_KEY` in `src/.env` or the real OS environment.
- `docker-compose.yml`: a local `pgvector/pgvector:pg16` container on port 5432,
  database `embeddings`, with a named volume for persistence.
- `scripts/check_postgres.py`: a standalone reachability check that also confirms the
  `vector` extension can be created.
- `utils/postgres_store.py`: the write side of the vector store. `connect()` opens a
  connection with pgvector's types registered, `ensure_schema()` runs idempotent DDL every
  time rather than using migration tooling, and `upsert_document()` replaces one
  document's entries in a single transaction. Two tables, `documents` and `index_entries`,
  joined on `doc_id`. `index_entries` carries an `embedding vector(N)` column, `N` inlined
  from `settings.embedding_dimensions` at DDL time (currently 512), with an HNSW cosine
  index (OpenAI's embeddings are documented as L2-normalised, so cosine and inner product
  agree) and a **generated** `keyword_tsv tsvector` column with a GIN index, so the keyword
  side is maintained by Postgres rather than by the application. The upsert deletes the
  document's existing rows before inserting, because `chunk_id`'s sequence number is
  positional and re-extraction shifts it, so upserting by primary key alone would leave
  orphans. It refuses entries whose `embedding` is `None` rather than writing a null
  vector, and likewise a `doc_summary` with no `summary_embedding`. `documents` also
  carries `page_count`, `doc_summary`, `summary_embedding vector(N)` with its own HNSW
  cosine index, and `summary_embedding_model`. Those columns reach an already-created
  database only through the `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements in
  `ensure_schema()`, since `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table;
  both forms are needed and both must stay. The document upsert coalesces the summary
  columns rather than assigning them, so re-running `index` after a plain `ingest` does
  not blank a profile that is already there.
- `utils/postgres_search_index.py`: `PostgresSearchIndex` is the read side, satisfying
  `QAAgent`'s `search(query, top_k) -> list[SearchResult]` protocol -- the only retriever
  in this repo, `main.py ask` and `api.py` both build it directly. It
  fuses the two scorers with **Reciprocal Rank Fusion** (`_RRF_K = 60`) in a single SQL
  statement, because `ts_rank_cd` and cosine distance are incompatible scales and fusing by
  rank avoids having to normalise one onto the other. Each branch contributes
  `max(top_k * 10, 50)` candidates so there is enough ranking to fuse over. Rows come back
  as `DocumentChunk` objects rather than `IndexEntry`, built from `display_text` with
  `kind` recovered from `element_types`, since that is what `QAAgent` consumes. The
  embedder is duck-typed on `embed_texts`, so this module never imports `Embedder`
  concretely. A `LEFT JOIN documents` carries
  each row's `doc_summary` onto the `SearchResult`. `search_documents(query, top_k)` is the
  document lane: a separate vector-only query against `documents.summary_embedding` and its
  own HNSW index, returning `DocumentSummary` objects for routing. It is vector-only because
  a profile is prose written to be semantically matched and holds none of the exact figures
  that make fusing in keyword search worthwhile. `QAAgent` calls it only when chunk search
  came back empty. It also implements `document()`, `page()` and `files()`: exact
  `file_name` lookups against `index_entries` (ordered by `page_start` then `chunk_id`),
  so `read_source` fully works against it. Neighbouring-chunk context
  (`context_before`/`context_after`) is also implemented, but gated behind
  `attach_context` (constructor argument, falling back to
  `settings.attach_search_context`, `False` out of the box): fetching and sending it
  costs real tokens on every match regardless of whether that match needed it, so it is
  opt-in rather than automatic. `main.py ask --attach-context` (or
  `ATTACH_SEARCH_CONTEXT=true` in `.env`) turns it on; see
  [search.md](search.md#4-context-expansion-context_beforecontext_after) for the full
  tradeoff.
## Figures

On by default on `ingest` (`--no-caption-figures` opts out), and it works behind either
extractor. This is the one Claude-costing step in `ingest` that is not opt-in -- see
[architecture.md](architecture.md#why-figure-captioning-is-the-one-exception) for why:

- `likely_figure_pages(pdf_path)` in `utils/figure_captioner.py` narrows every file to the
  pages worth a Claude call before `caption_document` runs, by unioning two independent,
  local, no-API-cost signals: pdf_inspector's accessibility structure tree (a `Figure` or
  `Caption` role at a page, when the PDF is tagged) and pypdfium2's page-object scan (an
  embedded raster image object, when it isn't). On the eight fixtures this is 46 of 109
  pages, 42%. Neither signal is complete alone -- the first needs a tagged PDF, several of
  the fixtures aren't; the second only sees embedded bitmaps, not a chart drawn as vector
  paths -- which is why they're unioned rather than either being trusted on its own, and
  why `describe_page`'s own logo/signature classification still does the final filtering.
  If detection itself raises, `ingest` falls back to captioning every page for that file
  rather than silently skipping it. One Claude call per candidate page, not per figure.
- `extract_figure_images(pdf_path, pages, output_dir)` in `utils/document_extractor.py`
  runs the same candidate pages through Docling's layout model and saves what it finds as
  real cropped PNGs (`PictureItem.get_image()`), to `output_dir/<stem>/figures/`. This is
  the one place Docling's picture detection is actually used: elsewhere both extractors
  skip picture items entirely. It is a dedicated converter, not `DocumentExtractor`'s: OCR
  and table structure are off (picture detection comes from the layout model regardless of
  either), and `generate_page_images` is on (a prerequisite for `get_image()`). Crops under
  2% of their page's area are dropped as furniture -- verified against a real fixture page
  that returned seven such crops (10x10 to 70x73px, badges and an ORCID icon) alongside a
  genuine chart on another page that filled over half the page. No crop is yet bound to a
  specific described figure: a page can hold several of each in orders that need not
  align, so the crops exist on disk as their own artifact, not as chunk or entry metadata.
  Runs alongside captioning on the same candidate pages; a crop failure is logged and does
  not stop captioning for that file.
- `utils/figure_captioner.py`: `FigureCaptioner` is the primary path by which a value
  printed inside a chart or map becomes searchable. Both PDF extractors skip picture items,
  and the vision extractor's figure prose lands loose inside a 3000-character page
  transcript where nothing cites it and keyword search scores it against the whole page,
  not the figure specifically. This instead
  emits one `kind="figure"` chunk per figure, its text being the caption followed by an LLM
  description, with the caption repeated into `headings` so it gets the same token weighting
  a section title does. Detection is done by the model on the whole rendered page rather
  than by cropping, which is what keeps it independent of both extractors and lets it reuse
  `PageRenderer`'s cache. Chunk IDs use an `#f` discriminator so they cannot collide with
  the `#c` text chunks numbered over the same pages. Logos and signatures are dropped as
  page furniture, mirroring `FigureElement.is_furniture`. Uses structured output against a
  JSON schema, both to parse reliably and because a schema-bound response cannot leak stray
  tags; a per-page failure is isolated to that page and reported through `failed_pages`,
  which matters because a page that failed and a page with genuinely no figures are
  otherwise indistinguishable.
- **Recovering misclassified text.** The same call also catches a different failure: a
  layout model (Docling's) can box a page's real body text as one giant misclassified
  picture, which neither extractor's picture-skipping logic ever sees inside. The prompt
  asks the model to report any such substantial running prose it finds as its own item
  (`classification: "text"`), transcribed verbatim rather than described. `_to_chunk` gives
  that a `kind="text"` chunk (not `kind="figure"`), so it merges into search and citations
  like any other extracted text. `main.py::_drop_duplicate_recovered_text` then drops a
  recovered chunk if its page already has 500+ characters of otherwise-extracted text, so a
  page the primary extractor handled fine doesn't also gain a re-transcribed duplicate. See
  [architecture.md](architecture.md#why-figure-captioning-is-the-one-exception) for the bug
  that motivated this.
- Runs on `claude_figure_model` (`claude-sonnet-5`), not `claude_model`, as does page
  transcription (`claude_vision_model`). Neither job needs the reasoning `claude_model` pays
  for; both are bounded by how much of the render the model resolves, so both want the
  high-resolution vision tier that matches `PageRenderer`'s 2576px output. `claude-haiku-4-5`
  was tried for figures and is a one-line switch back for a cost-sensitive run, but see the
  accuracy note below. The call shape omits `effort` and `thinking` so that switch stays
  valid -- the cheaper tier rejects the former and lacks the latter -- so it must not be
  copied from `vision_extractor.py`, which does pass `effort`.
- The prompt requires every value to be written as `label: value` and requires the figure's
  own legend to be applied, so a quantity whose sign or series is encoded by colour rather
  than printed is still recovered. Both rules are load-bearing: without them the output is a
  bag of unattributed numbers, which puts the value in the index without making the question
  answerable.
- **Accuracy limit, measured.** On Fig. 5 of the Calvillo fixture -- six panels, eleven UK
  regions, values in small boxes -- `claude-haiku-4-5` read the numbers but could not attach
  them to regions at all, and dropped the colour-encoded signs. `claude-sonnet-5` attaches
  them, recovers the signs, and flags its own uncertainty where markers cluster, but still
  swapped London and South East in the densest panel: `tests/fixtures/eval_questions.json`
  C5 expects South East -5,809 and London -3,748, and the index has both under London. So
  this makes figure-tagged questions answerable and correctly cited where they previously
  could not be answered at all, and it is not an oracle for dense multi-panel figures.
  Quantifying that is what the evaluation harness in `docs/roadmap.md` item 10 is for.

## Document summaries

Opt-in via `ingest --summarise-documents`, one Claude call per source file:

- `utils/document_summarizer.py`: `DocumentSummarizer` writes one `DocumentSummary` per
  source file, in contrast to `MetadataEnhancer`'s one summary per chunk. Its job is
  routing rather than embedding: shown alongside every result from its file, it tells the
  answering model whether it is in the right document and whether an extract is worth
  escalating to `read_source`. It takes chunks rather than a PDF, so it runs after either
  extraction path, and it prompts on a synthesised profile (front matter capped at 6000
  characters, up to 60 section headings, and table and figure counts) rather than the whole
  document, which keeps the cost down and stops the model summarising findings instead of
  scope. The prompt (`prompts/document_summary.prompt`) asks for scope and inventory and
  explicitly forbids findings, so nothing in a profile can be mistaken for evidence. Files
  whose summary comes back empty, including on a refusal, are skipped rather than stored
  blank.
- Generated in `ingest`, after both figure captioning and `--enhance-metadata`, so a
  profile describes the final chunk list. Persisted in `SearchIndex.documents`, which
  means `index` embeds the text it finds there rather than making a second Claude call.
- Two lanes, never one. A profile is **displayed** with results from its own file, and is
  separately **searchable** only through `PostgresSearchIndex.search_documents()`. It is
  never tokenised for keyword ranking, never ranked against chunks and never carries a
  page range, so it cannot displace the chunk holding an actual figure and cannot be
  dressed up as a citation. `prompts/qa_system.prompt` tells the model the same thing.
- Storage form differs by stage, not the text: the extraction-time JSON index keys
  profiles by `file_name` in `SearchIndex.documents`, while `main.py index` re-keys them
  by `doc_id` in columns on Postgres's `documents` table when it embeds and upserts them.

## Metadata enhancement

Opt-in, not run by default:

- `utils/metadata_enhancer.py`: `MetadataEnhancer` writes a short LLM summary for a
  chunk and returns a copy with `summary` populated, leaving the input untouched. Meant
  for chunks whose raw text is a poor embedding target on its own -- a table's exact
  figures matter for keyword search, but a sentence describing them is what should
  actually be embedded. `enhance_chunks()` defaults to summarising only
  `kind="table"` chunks (`DEFAULT_SUMMARISED_KINDS = ("table",)`), because
  `index_entry_builder.py` is the only reader of `summary` and only reads it for tables;
  everything else is passed through unchanged rather than summarised, which is 36 calls
  rather than 384 on the bundled fixtures. A `kinds=` argument widens the set for a
  caller that has a reason to. Each summarised chunk gets the immediately preceding
  chunk's text as context when it shares the same file, since a table's introducing
  sentence usually lands in the chunk before it; that predecessor is tracked from the
  full input list, not from the last chunk actually summarised, so skipping non-table
  chunks cannot hand a table the wrong context. The system prompt lives in
  `prompts/chunk_summary.prompt`. One call per summarised chunk; there is no batching.
- Wired into the CLI behind `--enhance-metadata` on `ingest`, off by default.

## Evaluation

`docs/roadmap.md` item 10, run via `python scripts/run_evals.py` from `src/`:

- `tests/fixtures/eval_questions.json`: hand-maintained directly as `models.eval.
  EvalQuestionBank` JSON (currently 80 questions, 8 short names), each question carrying
  `id`, `section`, `tags`, `input` and its own reference `output`, plus a "short name ->
  real file name" mapping since the reference answers cite sources by short name
  (`"Burlinson p1"`) rather than the real file name. This started as hand-written Markdown
  (`docs/eval_questions.md`) parsed by a one-off `utils/eval_parser.py` /
  `scripts/convert_eval_questions.py` step; both are gone now that the JSON itself is
  edited directly, which is also what the Markdown table's own past inaccuracy argued
  for -- the parsing step once caught two short names (`Taylor`, `Seger`) whose "File"
  column was truncated with `...` in the Markdown rather than the real fixture file name,
  harmless for a human reader but a mapping `EvalGrader` needs verbatim to resolve a
  reference's citation shorthand against what `QAAgent` actually cites.
- `utils/eval_judge.py`: `EvalGrader` grades one `QAAgent` answer against its
  `EvalQuestion` with a single Claude call, returning a `models.eval.EvalResult`
  (`pass`/`partial`/`fail` plus a one-sentence reason). Exact-match scoring isn't viable
  against open-ended prose answers, so a judge reads both the reference and the actual
  answer and checks content and citation separately; the short-name mapping is passed
  along so it can resolve a reference's shorthand against a real file name. A malformed
  or refused grading response defaults to `fail` rather than a silent pass, the same
  discard-over-guess choice `MetadataEnhancer` makes on a truncated summary. A
  `stop_reason == "max_tokens"` response is checked for explicitly and reported as a
  truncated grading call (`GRADE_MAX_TOKENS = 1024`) rather than left to fall through as
  a fail with a blank reasoning string -- adaptive thinking's own tokens count against
  this budget too, and were seen consuming it entirely on a real run before any verdict
  was written.
- `scripts/run_evals.py`: reads `tests/fixtures/eval_questions.json` and runs every (or a
  `--tag`/`--id`/`--limit`-filtered subset of) question through a live `QAAgent` against
  `PostgresSearchIndex` -- grades each answer, writes the full results to
  `data/eval_results.json`, and prints
  a pass-rate summary grouped by tag, so a `trap` or `figure` pass rate is visible on its
  own rather than folded into one number. Needs Postgres running and an index already
  written by `python main.py index`. Costs two Claude calls per question (the answer,
  then the grade) plus whatever `search_knowledge_base`/`read_source` calls `QAAgent`'s
  own loop makes. Questions run concurrently, `--workers` (default 5) via
  `concurrent.futures.ThreadPoolExecutor`: each worker thread lazily builds its own
  `QAAgent` (and so its own `PostgresSearchIndex` and Postgres connection) and its own
  `EvalGrader` on first use and reuses them for every question that lands on that thread
  afterwards, rather than sharing one `QAAgent`/connection across threads -- a psycopg
  connection is not safe for concurrent use from more than one thread, and per-thread
  isolation sidesteps having to reason about it beyond that.

## Data models

- `models/documents.py`: `DocumentChunk` (`chunk_id`, `file_name`, 1-indexed
  `page_number`, `text`, `headings`, `kind` being one of `"text"`, `"table"` or
  `"figure"`, a reserved `embedding`, and a reserved `summary`); `DocumentSummary`
  (per-file routing gloss, keyed by `file_name`, deliberately never ranked or cited);
  `SearchIndex` (`version`, `created_at`, `chunks`, and `documents`, which stays empty on an
  index written before document summaries existed); and `SearchResult` (`chunk`, `score`,
  `context_before` and `context_after` -- the neighbouring chunks attached after ranking and
  never scored -- plus `doc_summary`, carried on the result rather than fetched through a
  second retriever method so the backend protocol stays `search()` alone).
- `models/config.py`: pydantic-settings `Settings`, loaded from `src/.env`. Covers the
  Claude API key, model and token budget, index path, `top_k`, the embedding model name
  and daemon port, and the Postgres connection details.
- `models/elements.py`: a two-layer schema (`Element` -- `TextElement`/`TableElement`/
  `FigureElement`, provenance and structure -- and `IndexEntry`, the searchable unit built
  from elements, with separate `embed_text`/`keyword_text`/`display_text` so a table's
  exact figures can drive keyword search while a description of it drives dense search)
  drafted for a future migration away from the current flat `DocumentChunk`. The
  extractors still produce `DocumentChunk`, not `Element`, so Layer 1 is unpopulated; the
  adapter below bridges straight to Layer 2 in the meantime. See `docs/roadmap.md`.
- `utils/index_entry_builder.py`: `build_index_entries(chunks, pdf_path, image_dir)` converts the
  `DocumentChunk` list for one PDF into `IndexEntry` objects, deliberately skipping Layer 1
  rather than using `IndexEntry.from_text_elements`/`from_table`. It hashes the file once
  to derive `doc_id` (so all chunks from a file share it), re-keys each entry as
  `{doc_id}-p{page:03d}-c{seq:02d}`, and uses `kind` to decide the three text fields: for a
  table, `keyword_text` and `display_text` are the Markdown while `embed_text` is the
  `summary` if `MetadataEnhancer` has run, falling back to the first 500 characters; for
  prose and for figures all three are the same string, a figure chunk's caption-plus-
  description already being both its searchable and its readable form. `element_ids` is
  always empty, since Layer 1 does not exist yet. `asset_paths` is empty too except on a
  figure entry, which records the page image it was read from when the optional `image_dir`
  argument is supplied -- omitted, it stays empty, so a caller that never rendered anything
  does not have to know about it. Pure and deterministic: no embedder, no network.

## CLI

Three subcommands, in pipeline order.

- `python main.py ingest` extracts every PDF in a folder and writes the JSON index.
  `--extractor {fast,vision}` picks the extraction path (default `fast`), `--image-dir`
  sets where rendered pages go (default `data/pages`), figure captioning runs by default
  (one call per candidate page; `--no-caption-figures` turns it off), and
  `--enhance-metadata` summarises every table chunk (one call per table, off by default).
  Figure captioning runs before metadata enhancement, but that ordering no longer matters
  for cost: figure chunks are never summarised by default either way.
- `python main.py index` embeds the chunks in an existing JSON index and upserts them into
  Postgres, grouped by file so each document's rows are replaced in one transaction. Kept
  separate from `ingest` because extraction is free and offline whereas this needs the
  database and the embedding model. `--source` is required in effect, since each file is
  hashed to derive its `doc_id`; a file listed in the index but missing from `--source` is
  skipped with a warning. Embedding is an OpenAI API call, not a local model, so the
  whole bundled corpus (419 chunks, 8 files, plus profiles) embeds in about 11 seconds.
- `python main.py ask "question"` answers a single question and exits. `--top-k <n>`
  overrides `settings.top_k`, `--debug` logs each search query, tool call and LLM round
  trip to stderr, and `--attach-context` forces `PostgresSearchIndex`'s neighbouring-chunk
  context on for this run regardless of `settings.attach_search_context` (off by default;
  see "Embeddings and vector storage" above). Builds a fresh `Embedder`,
  `PostgresSearchIndex` and `QAAgent` on every invocation, since each run is a separate
  process; there is only the one backend to build.

## API

- `api.py`: a FastAPI service wrapping the same `QAAgent` behind `POST /ask` (request
  `{"question": "..."}`, response `{"answer": "..."}`, `422` for a blank question).
  `Embedder`, `PostgresSearchIndex` and the `QAAgent` are constructed once in a `lifespan`
  context manager at startup and reused across requests, rather than once per question as
  the CLI does.
  `get_agent()` is a FastAPI dependency so tests can override it with a mock rather than
  needing a real index or Claude client. Run with `uvicorn api:app --reload` from `src/`.
- `postman/`: a Postman collection and local environment for the endpoint above --
  a happy-path request and one demonstrating the `422` validation.

## Scripts

`src/scripts/` holds standalone checks and exploratory work, kept out of the
application code: `check_postgres.py`, `peek_postgres.py` (prints the 10 most recently
ingested documents and 10 sample `index_entries` rows, for eyeballing what is in the
database without a separate client), `test_tool_use.py` (runs `QAAgent`'s real tool-use loop
against a handful of hand-written chunks and real Claude API calls, to sanity-check
keyword rewriting, multi-search, no-match handling, multi-hop synthesis and prompt-
injection resistance without any PDFs, embeddings or Postgres), `run_evals.py` (see
"Evaluation" above), and the `docling_test.ipynb` and `test.ipynb` notebooks.
