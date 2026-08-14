# Roadmap

Status of the work items in the `TODO` file at the repository root. `TODO` remains the
place to capture new intent; this file records where each item actually stands.

## Done

| Item | Where it landed |
| --- | --- |
| 1. Install Docling | `utils/document_extractor.py`, wrapped with OCR and table structure enabled |
| 2. Research local embedding options | Started with a locally-run BGE-M3 (via a resident daemon so its ~22s load cost was paid once, not per CLI invocation) but **reversed**: measured at ~180 chars/second on ordinary laptop CPUs, it made a bulk `index` run of this repo's own corpus a ~63-minute job. `utils/embedder.py` now calls OpenAI's API (`text-embedding-3-large`, truncated to 512 dimensions), which has no load cost to amortise, so the daemon was deleted rather than kept alongside it. See [architecture.md](architecture.md#why-embeddings-are-a-remote-api-call-not-a-local-model). Postgres with pgvector still chosen for storage, container in `docker-compose.yml` |
| 5. Docling pipeline producing a mix of table, text and figure outputs | Text and tables are extracted; figures are deliberately skipped, per item 4 |
| 6. Consolidate outputs into a single schema | `DocumentChunk` in `models/documents.py`, now also carrying a `kind` (`"text"`/`"table"`) field. See the note on the richer schema below |
| 7. Accept the element class and produce the correct index entry | `_write_index` in `main.py` persists `SearchIndex` (chunks plus any document profiles) to `data/index.json`. Originally landed via a `LexicalSearchIndex` class that also did BM25 keyword search over that JSON; that class and the `bm25` backend built on it were later deleted once Postgres became the only backend, see "Also shipped" below |
| 8. Search system with LLM-written and query-rewritten keywords | `QAAgent`'s `search_knowledge_base` tool. Claude writes the keywords itself and can search repeatedly. A second tool, `read_source`, lets it escalate to a whole document or page when a search result looks decisive but incomplete. Hybrid search is separate, see "In progress" |

Item 3, summary-level chunks per document, was **settled the other way**: a document
summary is not a chunk. `utils/document_summarizer.py` writes one `DocumentSummary` per
file behind `ingest --summarise-documents`; it is displayed with results from its own file
and is separately searchable through `PostgresSearchIndex.search_documents()`, but it is
never tokenised for keyword ranking, never ranked against chunks and never citable. In
Postgres it lives in columns on the `documents` table rather than as a
`granularity='document'` row in `index_entries`, so the separation is structural instead of
a filter every query has to remember. See
[architecture.md](architecture.md#why-document-profiles-are-not-chunks) for why. What is
left is fusing the document lane into ordinary ranking, which needs the eval harness of
item 10 to tune against rather than a guessed weight.

Item 4, what to do with figures, was **reversed**. The original assumption was that figures
would not become their own references, and both PDF extractors still skip picture items.
But a value printed inside a chart or map is unreachable any other way, so
`utils/figure_captioner.py` now emits one `kind="figure"` chunk per figure, which ranks and
cites like any other chunk. `likely_figure_pages()` (below) still bounds this to pages with a
plausible reason to hold an image rather than every page.

**Reversed a second time: captioning is now on by default, not opt-in.** It was `ingest
--caption-figures` until the B2 finding below (a real extraction-loss bug) showed that
skipping picture items has a cost beyond missed chart values: Docling's layout model can
misclassify a page's actual body text as one giant picture region and silently drop it, with
no error and no warning, since neither extractor has any handling for `PictureItem` at all.
`FigureCaptioner` now also recognises this case -- its prompt asks the model to report any
substantial block of running body text it sees that isn't a real figure as its own item
(`classification: "text"`), transcribed verbatim, and `_to_chunk` gives that a `kind="text"`
chunk instead of a `kind="figure"` one, so recovered text merges into search exactly like any
other extracted text rather than reading as a figure caption. `main.py::_caption_figures()`
guards against duplicating text the primary extractor already got right: a recovered `text`
chunk is dropped when its page already has 500+ characters of otherwise-extracted text
(`_drop_duplicate_recovered_text`, calibrated against the 114-character JEL/keywords-only
chunk that was all Burlinson page 1 yielded before this existed). The CLI flag is now
`--no-caption-figures` (opt-out, `caption_figures` defaults to `True` in `ingest()`/
`_ingest_vision()`/`_write_index()`). "Tiny logos" stay filtered by classification
(`logo`/`signature`, unchanged), not by crop pixel area -- true per-crop filtering would need
binding a specific Docling crop to a specific described figure, which is still unsolved, see
below.

`utils/index_entry_builder.py` carries the original figure reversal through to Layer 2: a
figure chunk becomes an `ElementType.FIGURE` entry with its page image in `asset_paths`, which
is the first thing to put anything in that field. No `FigureElement` is constructed on the
way, since Layer 1 is still unpopulated.

That `asset_paths` entry is the whole rendered page, not the figure itself: coarse, but
it existed before real crops did. `extract_figure_images()` in `utils/document_extractor.py`
now produces the real thing, via Docling's own `PictureItem.get_image()` on the same
candidate pages, but is not yet wired into `asset_paths` in its place -- a page can hold
several figures and several crops, detected in orders that need not align, and nothing yet
resolves which crop belongs to which described figure. Until that's solved, the crops sit
on disk as their own artifact and the whole-page fallback stands.

How well it works is measured rather than assumed, and the answer is "enough to change the
outcome, not enough to trust blindly". On Fig. 5 of the Calvillo fixture, six panels of
region-labelled UK maps, the figure chunk now carries the values with their colour-encoded
signs and the question is answerable and correctly cited where before no chunk held the
number at all -- but London and South East are swapped in the densest panel, against
`tests/fixtures/eval_questions.json` C5. The remaining `figure`-tagged questions have not
been scored, which is item 10's job.

## Also shipped, not originally in `TODO`

- **Vision extraction, an alternative first stage.** `utils/page_renderer.py`
  (`PageRenderer`, PDF pages to PNG via pypdfium2) and `utils/vision_extractor.py`
  (`VisionExtractor`, one Claude vision call per page) let `ingest` read a picture of the
  page instead of the PDF's internal structure. `python main.py ingest --extractor
  vision` -- not a fallback the default path picks up on its own, since it costs one
  model call per page regardless of whether the page is scanned or born-digital. Exists
  because the eval question bank's `figure`-tagged questions (values that exist only as
  chart or map labels) fail against a text-extracted index by construction.
- **`api.py`: a FastAPI service.** Wraps the same `QAAgent` behind `POST /ask`, loading
  the index once at startup instead of once per CLI invocation. A Postman collection and
  local environment live in `postman/`.
- **`.vscode/launch.json`.** Debug configurations for `main.py ingest`/`ask`, `api.py`
  under uvicorn, the standalone scripts, and pytest by marker or current file.
- **`src/scripts/test_tool_use.py`.** Exercises `QAAgent`'s real tool-use loop against
  hand-written chunks and real Claude API calls -- keyword rewriting, multi-search,
  no-match handling, multi-hop synthesis, and prompt-injection resistance -- with no PDFs,
  embeddings or Postgres required.
- **`tests/fixtures/eval_questions.json`.** A hand-maintained question bank across the
  eight fixtures (80 questions plus cross-document and retrieval-stress-test questions),
  each tagged by what it exercises (`single`, `multi-hop`, `compute`, `table`, `figure`,
  `footnote`, `cross-doc`, `trap`) and carrying its own expected answer and citation. This
  is most of the way to item 10 below -- it is the question set, not yet an automated
  harness. Originally drafted as Markdown (`docs/eval_questions.md`) and compiled by
  `scripts/convert_eval_questions.py`; that conversion step and the Markdown it read are
  both gone now, and the JSON is edited directly and lives alongside the other fixtures.
- **`QAAgent` search result dedup, later extended to context chunks too.**
  `_seen_chunk_ids`, reset per `answer()` call, strips chunks a previous
  `search_knowledge_base` call in the same conversation already returned as a *match*, so
  a narrowed re-query surfacing the same chunk from a different angle does not resend
  it. A query left with nothing new after that filter gets `DUPLICATE_RESULTS_MESSAGE`
  rather than the plain no-match message, since the query did match and the right next
  move is reading what is already in context, not trying different keywords. Once
  `PostgresSearchIndex` started attaching neighbouring-chunk context on every search
  (below), the same problem showed up one level down: a chunk already shown as
  *context* for one match could resurface as context for a different match in a later
  search, unnoticed by the match-only dedup above. `_format_results` (now an instance
  method rather than a `@staticmethod`, so it can read and update
  `self._seen_chunk_ids`) drops a context line whose chunk was already sent, as an
  earlier match or as earlier context, rather than re-rendering it -- the match itself is
  never dropped this way, only its neighbour lines. See
  [architecture.md](architecture.md#why-the-agent-searches-instead-of-being-handed-context)
  and [search.md](search.md#4-context-expansion-context_beforecontext_after).
- **Fixed: false "file not in the knowledge base" answers on filename-shaped questions.**
  Observed live: asked "what is in `1-s2.0-S2214629624004663-main.pdf`", the agent
  answered that the file wasn't indexed, when `scripts/peek_postgres.py` confirmed it was
  (`doc_id 11a981081a69`). `search_knowledge_base` ranks on chunk content only (RRF over
  `keyword_tsv` and `embedding`, see `utils/postgres_search_index.py`), never on
  `file_name`, and pgvector's branch has no relevance floor -- it always returns its
  top-K nearest neighbours, however weak the match. A filename query tokenizes into
  nothing meaningful for FTS and embeds to an arbitrary point in vector space, so the
  agent got back a handful of confidently-formatted but unrelated results and read that
  as proof of absence, rather than as a bad query. The `search_documents()` profile
  fallback in `_no_results_message` (`utils/qa_agent.py`) never fired either, since it
  only triggers on zero rows, not on weak ones. A system-prompt addition
  (`prompts/qa_system.prompt`) tells the model to call `read_source` with the literal
  file name first when a question names a specific file, instead of searching -- but that
  was only a partial fix on its own, since `read_source` degraded to "not supported"
  under the Postgres backend. The full fix has since landed:
  `PostgresSearchIndex.document()`/`page()`/`files()` do exact `file_name` lookups against
  `index_entries`, ordered by `page_start` then `chunk_id`, so `read_source` now gives a
  definitive yes/no (and the indexed-file listing on a miss) instead of routing every
  filename-shaped question through content search. Verified live against the same
  document that surfaced the bug (`idx.document("1-s2.0-S2214629624004663-main.pdf")`
  returns its 65 chunks), and now has unit coverage
  (`test_postgres_search_index.py`) and an integration test against real DDL
  (`test_postgres_store_integration.py::test_document_page_and_files_against_real_ddl`).
- **Removed: the `bm25`/`LexicalSearchIndex` backend.** Once the filename-question fix
  above made `PostgresSearchIndex` a full drop-in for `read_source`, there was no longer a
  reason to keep two retrieval backends around, so the JSON-backed one was deleted rather
  than kept alongside it: `utils/search_index.py` and its dedicated unit tests are gone,
  `main.py ask` no longer takes `--backend` (always Postgres, and `--index` dropped from
  the `ask` subparser since nothing reads it there anymore), `_write_index` writes
  `SearchIndex` JSON directly instead of building a `LexicalSearchIndex` just to call
  `.save()` on it, and `api.py` builds `PostgresSearchIndex` at startup instead of loading
  the JSON file. Tests that used `LexicalSearchIndex` as a convenient in-memory fake
  (`test_qa_agent.py`, `scripts/test_tool_use.py`) were given small local stand-ins
  instead (a substring-match fake and a token-overlap fake respectively) so they keep
  needing no Postgres connection. One real capability went missing for a while rather
  than being reassigned: `LexicalSearchIndex.search()` attached neighbouring-chunk
  context (`context_before`/`context_after`) after ranking via plain list adjacency, and
  nothing did that under Postgres immediately after the switch. `PostgresSearchIndex`
  has since grown its own version of this, see the next bullet.
- **`PostgresSearchIndex` grows its own neighbouring-chunk context -- opt-in, off by
  default.** `search()` can call `_attach_context` after RRF ranking: one query resolves
  each hit's immediate page-adjacent chunk_id via `LAG`/`LEAD` windowed `PARTITION BY
  file_name ORDER BY page_start, chunk_id` (per-file, so a match at a file boundary can't
  pick up another document's chunk), a second fetches the text for whichever neighbour
  ids actually exist -- two queries regardless of how many results came back, not one
  per hit. Not a port of the removed `LexicalSearchIndex` behaviour, a fresh
  implementation against `index_entries`, since the old one worked off in-memory list
  adjacency that has no Postgres equivalent. `QAAgent._format_results` needed no change
  to render the result: it already knew how to render `context_before`/`context_after`,
  delimiting the match with `>>> MATCH:` and labelling neighbours `...context`, from when
  `LexicalSearchIndex` was the only backend that populated them (it did need a change for
  a related reason, dedup, see the bullet above). Initially shipped unconditional --
  `_attach_context` ran on every non-empty search, no matter whether the match already
  stood on its own -- then gated behind `attach_context` (`PostgresSearchIndex`
  constructor argument, falling back to `settings.attach_search_context`, `False` by
  default) once the token cost of that was made explicit: each neighbour can be up to
  `max_chars`/~750 tokens, so a `top_k=5` search with every result at a non-boundary
  chunk can add ~10 extra chunks, ~7500 tokens, whether or not any of it turned out to be
  needed. `main.py ask --attach-context` forces it on for one run;
  `ATTACH_SEARCH_CONTEXT=true` in `.env` changes the default for the CLI and `api.py`
  both. Unit-covered in `test_postgres_search_index.py` (both-sides attachment, empty at
  a file boundary, per-file windowing, off by default, explicit `attach_context=True`
  independent of the setting). See
  [architecture.md](architecture.md#why-theres-an-escalation-tool-and-how-context-expansion-was-rebuilt-for-postgres)
  and
  [search.md](search.md#unconditional-per-match-and-off-by-default-because-of-it).

## In progress

**Hybrid search: keyword plus dense vector (completes item 2's storage half and extends
item 8) -- done.** `Embedder` calls OpenAI's API for dense vectors, the pgvector
container is up, and `scripts/check_postgres.py` plus integration tests prove the
container and extension work. The write side: `utils/postgres_store.py` has the schema
(`documents` and `index_entries`, HNSW cosine index on a `vector(N)` column sized from
`settings.embedding_dimensions`, GIN index on a generated `tsvector`) and an idempotent
per-document upsert. A real run of the bundled 8-file, 419-chunk corpus embeds and
upserts in about 11 seconds.

The read side exists as well: `utils/postgres_search_index.py`'s `PostgresSearchIndex`
implements `search(query, top_k) -> list[SearchResult]` and fuses the keyword and vector
rankings with Reciprocal Rank Fusion in one SQL statement, with an end-to-end integration
test covering schema, upsert and search. The pipeline is wired end to end:
`python main.py index` embeds an extraction index and upserts it, and
`python main.py ask` queries it -- Postgres is now the only backend, see "Also shipped"
above.

Both gaps this used to list are closed: `PostgresSearchIndex` implements
`document()`/`page()`/`files()` (exact `file_name` lookups against `index_entries`,
ordered by `page_start` then `chunk_id`) so `read_source` fully works, and `search()`
can attach neighbouring-chunk context the same way the removed JSON backend used to --
opt-in rather than always-on this time, off by default (see "Also shipped" above for
both).

What is left:

- Sparse retrieval goes through Postgres FTS (`ts_rank_cd`) rather than a model's own
  sparse weights. OpenAI's embeddings API doesn't produce those the way BGE-M3 did, so
  this would need a different source for them (or stay FTS-only) if it's ever wanted.

`models/elements.py` sketches the schema this would likely feed: `Element`
(`TextElement`/`TableElement`/`FigureElement`, structure and provenance) and `IndexEntry`
(the searchable unit, with separate `embed_text`/`keyword_text`/`display_text` so a
table's exact figures drive keyword search while a summary of it drives dense search).
Its own docstring is explicit that retrieval itself -- score fusion, citations -- belongs
to "the separate Postgres-backed search service that consumes this schema's output," not
to the parser. Layer 2 is now reachable: `utils/index_entry_builder.py` converts a PDF's
`DocumentChunk` list straight into `IndexEntry` objects, using `kind` to diverge the three
text fields for tables and deriving a sha256-based `doc_id` so re-ingest is idempotent.
Layer 1 is still unpopulated, and `upsert_document()` in `utils/postgres_store.py` is the
only consumer of the entries so far.

**Per-chunk LLM summaries -- done.** `utils/metadata_enhancer.py`'s `MetadataEnhancer` and
`DocumentChunk.summary` exist, wired into `ingest` behind `--enhance-metadata`, off by
default. This covers the "per-element summary" half of item 6 (extended) below. Which
chunks are worth summarising is now decided: `enhance_chunks()` defaults to
`kind="table"` only (`DEFAULT_SUMMARISED_KINDS`), since `index_entry_builder.py` is the
only reader of `summary` and only reads it for tables, so summarising plain prose or
figure chunks would cost calls for a value nothing consumes. That took the fixtures from
384 calls to 36. The document-level summary that was item 3 is separate and now shipped,
see above.

**10. Evaluation against the sample questions -- done.** `tests/fixtures/eval_questions.json`
is a hand-maintained fixture -- structured `{id, section, tags, input, output}` questions
plus a short-name-to-real-file-name table -- read directly by `scripts/run_evals.py`. It
started as hand-written Markdown (`docs/eval_questions.md`) compiled by
`scripts/convert_eval_questions.py` via `utils/eval_parser.py`; both the Markdown and the
conversion step were removed once the JSON became the thing edited directly, so there is
no intermediate to keep in sync any more. `run_evals.py` runs each question through a live `QAAgent`
against the Postgres/hybrid-search backend and grades each answer with an LLM judge
(`utils/eval_judge.py`, one Claude call per question) rather than by eye, since these
are open-ended prose answers with their own citation shorthand ("Burlinson p1") that
exact-match scoring can't follow. Questions run concurrently (`--workers`, default 5;
a `concurrent.futures.ThreadPoolExecutor` with one `QAAgent`/`PostgresSearchIndex`/
Postgres connection and one `EvalGrader` built per worker thread, since a psycopg
connection is not safe to share across threads). The judge checks both content and
citation and returns `pass`/`partial`/`fail` with a one-sentence reason, and is checked
for its own `max_tokens` truncation the same way `MetadataEnhancer` is, rather than
letting a cut-off grading call read as a blank fail. Results are written to
`data/eval_results.json` and summarised by tag on stdout, so a `trap`- or
`figure`-tagged pass rate is visible on its own rather than folded into an overall
score. Filterable by `--tag`, `--id` and `--limit` for a partial run.

Deliberately Postgres-only: `PostgresSearchIndex` is now the only backend (the `bm25`/
`LexicalSearchIndex` one was removed, see "Also shipped" above). `read_source` works
fully against it -- `document()`/`page()`/`files()` landed -- so a `multi-hop` question
can escalate to a whole page or document, and `search()` now attaches neighbouring-chunk
context on its own (see "Also shipped" above), so a `footnote` or cut-off-sentence
question no longer needs `read_source` just to get the completing chunk for free.

## Not started

- **6 (extended). The richer element hierarchy**, now drafted as `models/elements.py`
  (see "In progress" above). Layer 2 is partly live: `utils/index_entry_builder.py` builds
  `IndexEntry` objects from flat chunks. Layer 1 is not, and needs
  `document_extractor.py`/`pdf_extractor.py` rewritten to emit `Element` objects, which is
  also what would populate `element_ids` and the figure and table asset paths that the
  adapter currently leaves empty.
- **9. Citations by ID with a post-lookup.** Citations are currently formatted into the
  tool results as `file.pdf, page N` and reproduced by Claude in prose. Switching to
  chunk IDs with a lookup after the answer would make citations verifiable rather than
  copied, and `DocumentChunk.chunk_id` already exists for it. The `read_source` tool's
  degrade-gracefully pattern (checking for `document()`/`page()`/`files()` before
  relying on them) is a template for adding a lookup accessor the same way.
- **11. Structured `/ask` output: `metadata.sources` instead of inline citations.**
  `api.py` currently returns only `{"answer": str}`, with citations left as free text in
  the prose (e.g. "(report.pdf, p. 4)"); the sample question format in `TODO` pairs
  `answer` with `metadata.sources: [{file, page}]` instead. Reliably filling that array
  means replacing `QAAgent.answer()`'s plain-text final turn with a schema-constrained
  response, the same way `utils/figure_captioner.py` already uses `output_config.format`
  for structured output elsewhere in this repo -- touching `QAAgent.answer()`'s return
  type, a new `Source` model, `api.py`'s response model, `main.py`'s CLI printer, and the
  eval grading path (currently compares plain strings). A cheaper fallback considered:
  regex-parsing the existing `(file.pdf, p. 4)` pattern out of the prose answer, no
  agent-loop changes, but fragile against page ranges, short names, or paraphrased
  citations.

## Possible follow-ups not in `TODO`

- Resolve the `asyncio_default_fixture_loop_scope` pytest warning, either by installing
  pytest-asyncio or dropping the line from `tests/pytest.ini`.
- **Reranking after retrieval.** `_RRF_SEARCH_SQL` (`utils/postgres_search_index.py`)
  fuses keyword and vector rank into a score that is only ever relative to the other
  candidates in that one query; it has no absolute notion of "good match" versus "weak
  match," so `search()` always returns exactly `top_k` rows (padding out with its
  weakest candidates when fewer than `top_k` are actually relevant) the same way the
  filename-question bug (see "Also shipped" above) showed pgvector doing with no
  relevance floor. A reranker
  (a cross-encoder, a hosted API such as Cohere/Voyage rerank, or a cheap LLM
  relevance-scoring pass) would score each of the `candidate_limit` rows against the
  query on an absolute scale, letting a threshold cut the list down to however many
  genuinely clear the bar, zero included, instead of a fixed count. It would slot in
  between `_RRF_SEARCH_SQL`'s fetch and `_row_to_search_result`'s conversion, reranking
  either the fused top-`top_k` or the wider candidate pool before the final slice. Not
  yet built: needs a threshold chosen and tuned, which is exactly what `scripts/run_evals.py`
  (item 10) exists to check against rather than guessing, and adds one more model call
  (hosted latency/cost, or a local model to load) to every search.
- **Three ingestion/retrieval bugs found by a `scripts/run_evals.py` run against
  `tests/fixtures/eval_questions.json` (10-question smoke test, 2026-08-14): 7 pass,
  1 partial, 2 fail.** Traced with `scripts/peek_postgres.py` queries and a page-1 render via
  `PageRenderer`, comparing `data/index.json` (fast path) against `data/index_vision.json`
  (vision path) for the same file and page:
  - **B2 (fail): silent extraction loss, not a retrieval miss.** Page 1 of
    `1-s2.0-S0140988325000672-main.pdf` (the Burlinson fixture -- already the pinned
    "mixed document" in `test_mixed_document_reports_its_scanned_pages_needing_ocr`, pages
    1-7 needing OCR) indexes only 2 chunks: the boxed JEL-codes/keywords sidebar and a
    figure caption for the journal-cover thumbnail. The ABSTRACT box and the entire
    two-column Introduction body text -- including the sentence B2's answer depends on,
    "would represent nearly half (47 %) of the UK's 2035 abatement target for the power
    sector (Committee on Climate Change (CCC), 2022)" -- are silently absent; page 2
    picks up mid-sentence, confirming the loss rather than a chunking split. Reproduced
    directly: `DocumentExtractor().extract_chunks_for_pages(pdf_path, [1])` returns the
    same single 114-character JEL/keywords chunk that is in Postgres. `data/index_vision.json`
    (`--extractor vision`) has the full page for the same chunk, abstract and introduction
    included verbatim, so this is a Docling-specific failure, not a document-wide
    OCR-routing failure. A corpus-wide sweep comparing per-page character totals between
    `data/index.json` and `data/index_vision.json` (across all chunk kinds, not just
    `kind="text"`) found this pattern on exactly one page in the whole eight-file corpus;
    it is not widespread.

    Root cause, confirmed by converting page 1 directly and inspecting Docling's raw
    layout elements rather than the final chunks: the model emits 10 small `TextItem`s for
    the JEL/keywords lines plus **one `PictureItem` whose bbox spans nearly the entire page
    body** (`l=4.5, r=524.3, t=698.9, b=7.1` bottom-left-origin, against a 595x794pt page --
    roughly y=95pt to y=787pt, almost the full height below the top masthead strip).
    Docling's layout model classified the title, authors, ARTICLE INFO box, ABSTRACT, and
    the whole two-column Introduction as one giant mislabeled picture region.
    `_document_to_chunks()` has no branch for `PictureItem` at all -- both extractors
    deliberately skip pictures, see item 4 above -- so everything inside that region is
    discarded with no error or warning. `PdfExtractor().pages_needing_ocr()` confirms page 1
    has no usable embedded text layer, so Docling is running layout detection purely off a
    rendered bitmap here, not the PDF's vector structure; page 1 is also the only page with
    a graphics-heavy masthead (grey ScienceDirect banner, green journal-cover thumbnail, red
    "Check for updates" badge), which is plausibly what pulled the layout model's picture
    prediction into overshooting across the rest of the page.

    This is not fixable by turning on `--caption-figures`: `FigureCaptioner` doesn't touch
    Docling's `PictureItem` regions at all -- it renders the whole page independently and
    asks Claude to describe genuine chart/diagram/map/photo content, filtering out anything
    classified `logo`/`signature` as furniture (`_FURNITURE` in `figure_captioner.py`). Page
    1 would qualify as a candidate page (`likely_figure_pages()` flags it via its embedded
    raster images), but Claude would classify the journal-cover thumbnail as `logo` and
    discard it, since the prompt's job is "describe the figures on this page," not
    "transcribe everything." It has no reason to notice or recover the abstract/introduction
    text, because from its point of view there is no figure there worth describing. A
    `PictureItem` that is a genuine chart and one that is Docling's layout model misfiring
    on real paragraph text are indistinguishable to this pipeline today: both get skipped by
    the extractor, and neither gets rescued by figure captioning, which is scoped to finding
    charts, not to flagging a suspiciously low text yield on a page that needed OCR. That
    detection -- comparing extracted character count against some floor for an OCR'd page,
    or falling back to vision transcription when Docling's own output looks this thin --
    does not exist anywhere in the pipeline yet.

    **Fixed:** `FigureCaptioner` now also recognises and transcribes exactly this case, and
    captioning runs by default rather than behind `--caption-figures`. See item 4 above for
    the mechanism (`classification: "text"`, the `kind="text"` chunk this produces, and the
    duplicate-avoidance guard in `_caption_figures()`).
  - **B5 and B6 (fail/partial): a real table-in-the-wrong-chunk problem, not
    hallucination.** Table 5 (p8 of the same fixture, the persistent-adopters/non-adopters
    sub-sample decomposition: θ_I 0.317/0.425/0.408 and ethnicity 6.74 %/9.04 %/0.52 % for
    SOLARPV/SOLARHEAT/HYBRIDEV) is real, correctly-valued, and present in the index --
    but merged into one chunk (`...-p008-c38`) together with Fig. 1's long caption, with
    the table's own rows run together with no line breaks (`θ I0.317 0.425 0.408`,
    `Parental 16.12% 31.57 48.16`). That chunk is only retrieved when a query's wording
    echoes Fig. 1's caption closely, which is how B4 (passed, phrased around "persistent
    adopters/non-adopters sub-sample") surfaced it. B5 ("overall dissimilarity index") and
    B6 ("least contributing factor... ethnicity") instead pull in Table 2 (p5-6) and
    Table 3 (p6-7) -- real, cleanly-chunked, but wrong tables for the question, covering
    the main sample split by wave rather than the sub-sample's combined figures. The model
    answers confidently from those instead, and never calls `read_source` to check page 8
    in full -- confirmed live, `PostgresSearchIndex.page(file_name, 8)` returns all 3 of
    that page's chunks including the Table 5 one, so the escalation tool is not the gap
    (`document()`/`page()`/`files()` landed, see "Also shipped" above). The gap is that
    nothing in a well-formed Table 2/3 hit signals "there's a more specific table
    elsewhere"; `read_source`'s own tool description (`qa_agent.py`) lists triggers like a
    cut-off extract or a table missing its header, none of which fire when the wrong table
    is itself complete and clean. Fixing this needs either the chunk merge below split so
    the right chunk ranks competitively on its own terms, or a prompt/heuristic nudge to
    escalate when a table-tagged question's phrasing doesn't match the retrieved table's
    own caption.
  - **Separately, a lower-severity table-formatting issue on dense appendix tables**
    (e.g. Table A2/A3, p10 of the same fixture): the fast/Docling path keeps all the
    numbers but scrambles multi-row table headers into a single jumbled row and
    substitutes `↑` for minus signs (`↑0.342***` for what the vision path renders as
    `−0.342***`), which would corrupt any numeric answer sourced from those specific
    pages even though no content is missing. Not implicated in any of the three eval
    failures above, but found during the same sweep and worth a test fixture if
    Table A2-style pages are ever cited by a future eval question.
