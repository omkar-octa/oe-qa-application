# Flow charts

How the system runs, end to end. Diagrams are Mermaid, so they render on any Markdown
viewer that supports it.

For what each component does see [features.md](features.md), for why it is built this
way see [architecture.md](architecture.md), for the retrieval mechanics in depth (query
rewriting, RRF, the schema) see [search.md](search.md), and for what is built versus
planned see [roadmap.md](roadmap.md).

## 1. The system at a glance

Three commands joined by stored state, none of which imports another. Extraction is slow
and one-off, offline apart from its opt-in Claude stages; dense indexing is a deliberate
second step because it needs Postgres and the embedding model; question answering is fast,
stateless and runs per question.

```mermaid
flowchart LR
    PDFs[/"PDFs<br>tests/fixtures/*.pdf"/] --> Ingest["<b>main.py ingest</b><br>extract, chunk, index<br><i>minutes, run once</i>"]
    Ingest --> Index[("data/index.json<br>SearchIndex: 384 DocumentChunks<br>+ document profiles, if asked for<br>regenerable, gitignored")]
    Index --> Dense["<b>main.py index</b><br>embed and upsert<br><i>required second step</i>"]
    Dense --> PG[("Postgres + pgvector<br>documents + index_entries")]

    PG --> Ask["<b>main.py ask</b> or <b>POST /ask</b><br>agentic retrieval, then answer<br><i>seconds, per question</i>"]
    Ask --> Answer[/"Answer with<br>(file.pdf, p. N) citations"/]

    Claude{{"Claude API"}} -.-> Ask
    Claude -.-> Ingest

    style Index fill:#e8e8e8,stroke:#333
    style PG fill:#e8e8e8,stroke:#333
```

`PostgresSearchIndex` is the sole retriever, implementing
`search(query, top_k) -> list[SearchResult]` plus `document()`/`page()`/`files()`
(unlocking `read_source`) and `search_documents()` (the document-profile fallback). An
earlier JSON-backed `LexicalSearchIndex` (BM25 keyword search, selected with
`--backend bm25`/`--backend postgres`) existed and has been deleted -- `ask` and `api.py`
always go straight to Postgres now, so `main.py index` is no longer optional the way the
diagram above once showed it.

## 2. Ingestion, default "fast" extractor

`python main.py ingest` with `--extractor fast` (the default). The routing decision is
made **per page**, not per document, which is the whole point of the two-tier design.

```mermaid
flowchart TD
    Start(["main.py ingest --extractor fast"]) --> Glob["sorted(source.glob('*.pdf'))"]
    Glob --> Any{"any PDFs?"}
    Any -- no --> Fail1["print 'No PDFs found'<br>exit 1"]
    Any -- yes --> Loop["for each PDF"]

    Loop --> Fast["<b>PdfExtractor</b> (pdf_inspector, Rust)<br>extract_chunks + pages_needing_ocr"]
    Fast --> Err{"raised?"}
    Err -- yes --> Skip["print FAILED, skip this PDF"] --> Next
    Err -- no --> NeedsOCR{"pages_needing_ocr<br>non-empty?"}

    NeedsOCR -- no --> Collect["chunks from the fast path only"]
    NeedsOCR -- yes --> Lazy["build <b>DocumentExtractor</b> lazily<br>first time only, loads Docling +<br>OCR/layout/table models"]
    Lazy --> Slow["extract_chunks_for_pages(pdf, ocr_pages)<br>contiguous runs via Docling page_range"]
    Slow --> Merge["concatenate, then sort by page_number"]
    Merge --> Collect

    Collect --> Next["accumulate into chunks[]"]
    Next --> More{"more PDFs?"}
    More -- yes --> Loop
    More -- no --> Write["_write_index"]

    Write --> Empty{"chunks empty?"}
    Empty -- yes --> Fail2["print 'No content extracted'<br>exit 1"]
    Empty -- no --> Cap{"--no-caption-figures?"}
    Cap -- no, default --> Figs["<b>FigureCaptioner</b><br>one Claude call per candidate page<br>adds kind='figure' chunks, plus<br>kind='text' for recovered body text<br>(section 5)"]
    Cap -- yes --> Enh
    Figs --> Dedup["_drop_duplicate_recovered_text<br>drops a kind='text' chunk if its page<br>already has 500+ chars extracted"]
    Dedup --> Enh{"--enhance-metadata?"}
    Enh -- yes --> Summ["<b>MetadataEnhancer</b><br>one Claude call per table chunk<br>36 calls for the fixtures"]
    Enh -- no --> Prof
    Summ --> Prof{"--summarise-documents?"}
    Prof -- yes --> Docs["<b>DocumentSummarizer</b><br>one Claude call per file<br>8 calls for the fixtures"]
    Prof -- no --> Build
    Docs --> Build["<b>SearchIndex</b>(chunks=chunks, documents=documents)"]
    Build --> Save[("save to data/index.json<br>chunks and document profiles")]
    Save --> Done(["exit 0"])
```

Points worth knowing before changing this:

- The three optional stages run in that order on purpose. Captioning first means figure
  chunks get summarised too; profiling last means a profile describes the final chunk
  list rather than a list the other flags were about to change.
- Captioning is the one stage of the three that runs by default; the other two stay
  opt-in. See
  [architecture.md](architecture.md#why-figure-captioning-is-the-one-exception) for why.
- `DocumentExtractor` is constructed inside the loop, so a corpus with no scanned pages
  never loads Docling or torch at all.
- A per-document text-layer test instead of a per-page one silently drops the scanned
  front matter of `1-s2.0-S0140988325000672-main.pdf`. See
  [architecture.md](architecture.md#the-bug-this-design-exists-to-prevent).
- Docling's `page_range` preserves absolute page numbers, which is why the merge is a
  plain concatenate and sort with no offset arithmetic. Chunk IDs do collide across
  conversion calls, so `extract_chunks_for_pages` renumbers over the merged list.

## 3. Ingestion, "vision" extractor

`python main.py ingest --extractor vision` is an alternative first stage, not a fallback
the fast path picks up on its own. It reads a picture of the page rather than the PDF's
internal structure, so scanned and born-digital pages cost the same: one Claude call
each.

```mermaid
flowchart TD
    Start(["main.py ingest --extractor vision"]) --> VE["<b>VisionExtractor</b>(image_dir)"]
    VE --> PerPdf["for each PDF"]
    PerPdf --> Render["<b>PageRenderer</b> (pypdfium2)<br>rasterise pages to PNG<br>long edge 2576px"]
    Render --> Cached{"PNG already<br>on disk?"}
    Cached -- yes --> Reuse["reuse it, rendering is<br>deterministic and slow"]
    Cached -- no --> Draw["render and write<br>data/pages/&lt;stem&gt;/pNNNN.png"]
    Reuse --> Pool
    Draw --> Pool["ThreadPoolExecutor, 4 workers<br>_transcribe_or_skip per page"]

    Pool --> Claude["<b>Claude vision</b> on claude_vision_model<br>page image + page_transcription.prompt<br>effort=low, max_tokens=8000"]
    Claude --> Raised{"call raised?"}
    Raised -- yes --> Failed["None, recorded in failed_pages;<br>ingest prints a WARNING.<br>Every other page is unaffected"]
    Raised -- no --> Refusal{"stop_reason<br>== refusal?"}
    Refusal -- yes --> Emptyd["empty transcript.<br><i>Not counted as a failure</i>"]
    Refusal -- no --> Md["Markdown transcript"]

    Failed --> NoChunks["page contributes no chunks"]
    Emptyd --> NoChunks
    Md --> Chunker["PdfExtractor._page_to_chunks<br>shared chunking and heading rules (section 4)"]
    NoChunks --> NextPdf
    Chunker --> Renum["re-sort by page, renumber chunk_ids<br>so order matches a sequential run"]
    Renum --> NextPdf{"more PDFs?"}
    NextPdf -- yes --> PerPdf
    NextPdf -- no --> Write["_write_index<br>same path as section 2"]
```

Both extraction paths converge on the same `_write_index`, and both emit the same
`DocumentChunk` shape, so nothing downstream can tell which produced the index.

A page that raised and a page that came back empty are deliberately distinguishable: only
the first lands in `failed_pages`, which `ingest` prints as a warning, because a silently
short document is worse than a loud failure. Transcription runs on
`claude_vision_model` rather than `claude_model`, since the job is bounded by how much of
the render the model can resolve rather than by reasoning.

## 4. Chunking, common to every extractor

Tables are lifted out of the page first and never share a chunk with the prose around
them. What is left accumulates into a buffer and flushes on the same triggers everywhere,
so a chunk never straddles a page and its citation stays unambiguous.

```mermaid
flowchart TD
    In[/"one page of Markdown,<br>or Docling items"/] --> Blocks["<b>_blocks()</b>: split the page into<br>runs of table rows and runs of prose"]
    Blocks --> Which{"block kind?"}

    Which -- table --> Cap["<b>_take_caption()</b>: move a 'Table N'<br>caption off the end of the preceding<br>prose so it stays with its table"]
    Cap --> Fits{"fits max_chars?"}
    Fits -- yes --> Own["one chunk, the table alone"]
    Fits -- no --> Split["<b>_split_table()</b>: divide by rows,<br>repeating caption and header on every<br>part; drop the caption, then the header,<br>if the prefix crowds out the rows"]
    Split --> Own
    Own --> Kind

    Which -- prose --> Acc["append the line to the buffer"]
    Acc --> T1{"section<br>heading?"}
    T1 -- yes --> Flush
    T1 -- no --> T2{"page change?<br>(Docling path)"}
    T2 -- yes --> Flush
    T2 -- no --> T3{"buffer + line<br>over max_chars (3000)?"}
    T3 -- yes --> Flush["flush the buffer"]
    T3 -- no --> Acc
    Flush --> Kind

    Kind{"two or more<br>table rows in it?"} -- yes --> IsTab["kind = 'table'"]
    Kind -- no --> IsTxt["kind = 'text'"]
    IsTab --> Out[/"DocumentChunk<br>chunk_id = file#pN#cM"/]
    IsTxt --> Out

    Long["a single line over max_chars"] -.-> LSplit["split on word boundaries,<br>hard character split as last resort"]
    LSplit -.-> Acc
```

- `max_chars` is a hard bound, not a target, and the table path honours it too. Docling
  hands back a whole table however wide it is, so `DocumentExtractor` splits it with the
  same `_split_table` rule the fast path uses.
- Every part repeats the header rather than being cut on the character budget alone,
  because a part that is a run of numbers with no column names is worse than useless: the
  values cannot be attributed to a variable, or can be attributed to the wrong one.
- `kind` is counted from the chunk's own content, two or more pipe-prefixed rows, rather
  than tested on its first line: a real table is introduced by its caption, and often a
  running header before that, so a first-line test classifies whole tables as prose. The
  answer matters downstream, because `index_entry_builder` embeds a table's summary but
  keyword-searches its full Markdown.
- Figures never come from here. Both PDF extractors skip picture items, so figure chunks
  are produced by a separate stage, next.

## 5. Figure captioning, on by default (`--no-caption-figures` to skip)

A value that exists only as a label inside a chart or a map is unreachable by every other
path in this pipeline. This stage makes it searchable, and it works behind either
extractor because detection runs on the rendered page rather than on the PDF's structure.
It also recovers body text a layout model boxed as a misclassified picture -- see
[architecture.md](architecture.md#why-figure-captioning-is-the-one-exception) for the bug
that made this the one stage of the three that isn't opt-in.

```mermaid
flowchart TD
    Flag(["_write_index, caption_figures default True"]) --> Group["group the chunks so far by file"]
    Group --> PerPdf["for each PDF"]
    PerPdf --> Render["<b>PageRenderer</b><br>the same PNG cache the vision path<br>fills, so an already-rendered corpus<br>pays no rasterisation cost here"]
    Render --> Pool["ThreadPoolExecutor, 4 workers<br>_describe_or_skip per page"]
    Pool --> Model["<b>claude_figure_model</b><br>figure_description.prompt,<br>structured output against a JSON schema"]

    Model --> Stop{"stop_reason?"}
    Stop -- refusal --> NoFig["no figures for this page"]
    Stop -- max_tokens --> Trunc["raise: under structured output a<br>truncated response is unparseable,<br>so the page lands in failed_pages"]
    Stop -- other --> Parse["parse figures[]"]
    Parse --> Filter["drop logos and signatures as page<br>furniture, drop empty descriptions"]
    Filter --> Kind{"classification == 'text'?"}
    Kind -- yes --> TextChunk["kind = 'text', text = description<br>(verbatim transcription),<br>headings = [caption] if any"]
    Kind -- no --> FigChunk["kind = 'figure',<br>text = caption + description,<br>headings = [caption]"]
    TextChunk --> IdAssign["chunk_id = file#pN#f0, an #f not a #c"]
    FigChunk --> IdAssign

    IdAssign --> Dedup{"kind == 'text' and page already has<br>500+ chars extracted elsewhere?"}
    Dedup -- yes --> Drop["dropped, logged as a likely duplicate<br>(_drop_duplicate_recovered_text)"]
    Dedup -- no --> MergeIn["extend the file's chunk list,<br>then a <b>stable</b> sort by page_number"]
    NoFig --> MergeIn
    Trunc --> MergeIn
```

The sort has to be stable and the figures have to be appended before it rather than
inserted: each page's figures then land after that page's text, with text order within a
page untouched. A recovered `kind="text"` chunk merges into search and citations like any
other extracted text; whether it gains a `context_before`/`context_after` neighbour window
depends on `PostgresSearchIndex`'s `attach_context` setting (section 6), off by default.

The caption is repeated into `headings` so its tokens get the weighting a section title
gets, which is what lets a query naming "Figure 5" rank the figure itself above prose that
merely mentions it. The `#f` discriminator keeps these IDs from colliding with the `#c`
text chunks numbered over the same pages. A page that failed and a page with genuinely no
figures are otherwise indistinguishable in the output, which is why only the first goes
into `failed_pages`.

## 6. Question answering, the agentic loop

`QAAgent` does not retrieve once and stuff the results into the prompt. Claude is given
two tools -- `search_knowledge_base` and `read_source` -- and writes the search keywords
itself, because a natural-language question searched verbatim scores badly under
Postgres full-text search.

```mermaid
flowchart TD
    Q[/"question"/] --> Msgs["messages = [user: question]"]
    Msgs --> Iter{"iteration < 8?<br>MAX_TOOL_ITERATIONS"}
    Iter -- no --> Budget["return SEARCH_BUDGET_MESSAGE"]
    Iter -- yes --> Call["client.messages.create<br>system = prompts/qa_system.prompt<br>tools = [search_knowledge_base, read_source]"]

    Call --> Ref{"stop_reason<br>== refusal?"}
    Ref -- yes --> RefMsg["return REFUSAL_MESSAGE<br><i>checked first: a refusal is HTTP 200<br>with empty content</i>"]
    Ref -- no --> Tool{"stop_reason<br>== tool_use?"}
    Tool -- no --> Final["return the text blocks<br>as the final answer"]

    Tool -- yes --> Which{"for each tool_use block,<br>which tool?"}

    Which -- search_knowledge_base --> Search["<b>index.search(query, top_k)</b><br><b>PostgresSearchIndex</b>: keyword and<br>vector lanes fused by rank (section 8)"]
    Search --> CtxCheck{"attach_context enabled?<br>(off by default)"}
    CtxCheck -- yes --> Expand["attach neighbouring chunks<br>(context_before/context_after),<br>ranking unaffected"]
    Expand --> Attach
    CtxCheck -- no --> Attach["attach doc_summary per result<br>(document profile, if one exists),<br>ranking unaffected"]
    Attach --> Fmt{"any<br>results?"}
    Fmt -- no --> Lane{"index has<br>search_documents()?"}
    Lane -- no --> NoMatch["'No matching extracts found'<br>Claude can retry with new keywords"]
    Lane -- yes --> DocLane["<b>document lane</b><br>vector query over<br>documents.summary_embedding"]
    DocLane --> NoMatchPlus["'No matching extracts found'<br>+ closest documents by profile,<br>marked not citable"]
    Fmt -- yes --> Dedup["drop chunks already surfaced by an<br>earlier search this answer() call<br>(_seen_chunk_ids, reset per question)"]
    Dedup --> Fresh{"any chunks<br>left over?"}
    Fresh -- no --> DupMsg["DUPLICATE_RESULTS_MESSAGE:<br>'already returned by an earlier<br>search; see those results above'"]
    Fresh -- yes --> Blocks["profile preamble, once per<br>distinct source document<br>(marked not citable), then<br>numbered extracts over the fresh<br>chunks only, renumbered from [1]:<br>'[1] Source: file.pdf, page 4'<br>...context, p. N: ...<br>&gt;&gt;&gt; MATCH: text<br>...context, p. N: ..."]
    NoMatch --> Append
    NoMatchPlus --> Append
    DupMsg --> Append
    Blocks --> Append

    Which -- read_source --> Supported{"index has<br>document()/page()/files()?"}
    Supported -- no --> Degrade["'does not support reading<br>whole sources; use search instead'"]
    Supported -- yes --> Scope{"page given?"}
    Scope -- yes --> OnePage["index.page(file_name, page)"]
    Scope -- no --> WholeDoc["index.document(file_name)"]
    OnePage --> Found{"any chunks?"}
    WholeDoc --> Found
    Found -- no --> NotFound["'No content found for ...'<br>+ index.files() as a hint"]
    Found -- yes --> Whole["all matching chunks concatenated,<br>each tagged '[p. N]'"]
    Degrade --> Append
    NotFound --> Append
    Whole --> Append["append assistant turn + tool_result(s)<br>to messages"]
    Append --> Iter

    Final --> Out[/"answer citing (file.pdf, p. N)"/]
```

Claude may call either tool more than once -- narrower keywords on
`search_knowledge_base`, or escalating to `read_source` when a result looks decisive but
incomplete (cut off mid-sentence, a table with no header, a footnote or figure it can't
see) -- before answering. The loop is capped so a model stuck rephrasing the same query,
or alternating between the two tools, cannot run forever.

`_seen_chunk_ids` tracks every chunk ID already sent to the model this `answer()` call and
is reset at the start of the next one, so it never leaks across questions. A rephrased
query that re-surfaces a chunk already in the conversation history sends it once, not
again per search: repeating it would waste context and read as new evidence when it is
the same evidence restated. If every chunk a search returns has already been seen,
Claude gets `DUPLICATE_RESULTS_MESSAGE` rather than an empty-looking result, pointing it
back at the earlier results instead of at `_no_results_message`'s "try different
keywords", since the query did match, just nothing new. Numbering in the formatted
extracts (`[1]`, `[2]`, ...) restarts from 1 over the surviving fresh chunks each call, so
it is only ever local to that tool result, never a running count across the conversation.

Document profiles never enter the ranked list. They are attached after scoring, printed
once per document, and reachable as a ranked thing only through the separate document
lane, which runs only when chunk search found nothing.

There is one backend, `PostgresSearchIndex`, and it implements `document()`, `page()` and
`files()` unconditionally, so `read_source` (the `Supported` branch above) always works.
Neighbouring-chunk context is the one piece that is *not* unconditional: it is gated
behind `attach_context`/`settings.attach_search_context`, off by default, because
fetching and sending it costs real tokens on every match whether or not that match needed
it (see [search.md](search.md#unconditional-per-match-and-off-by-default-because-of-it)
for the full tradeoff). `main.py ask --attach-context` or `ATTACH_SEARCH_CONTEXT=true` in
`.env` turns it on. The `hasattr` degrade-gracefully pattern described in
[architecture.md](architecture.md#the-retriever-protocol) is still how a future retriever
implementing only `search()` would lose the escalation path without erroring; it just
doesn't currently distinguish two backends here, only the one that exists.

## 7. Entry points and process lifetime

Same agent, same single retriever, different lifetimes. The CLI builds a fresh
`PostgresSearchIndex` and `QAAgent` on every invocation; the API builds both once at
process startup and reuses them across every request.

```mermaid
flowchart TD
    subgraph CLI["CLI: python main.py ask"]
        C1["process starts"] --> C2["Embedder()"]
        C2 --> C3["PostgresSearchIndex(embedder,<br>attach_context=args.attach_context)<br>connects to Postgres directly;<br>no index.json read"]
        C3 --> C4["new QAAgent, new Claude client"]
        C4 --> C5["agent.answer(question, top_k)"]
        C5 --> C6["print answer<br>(--debug: also print token counts, elapsed time)"]
    end

    subgraph API["API: uvicorn api:app"]
        A1["lifespan startup"] --> A2["Embedder(), PostgresSearchIndex(embedder)<br>attach_context left unset -&gt;<br>settings.attach_search_context"]
        A2 --> A3["QAAgent built once, held in module state"]
        A3 --> A4["POST /ask"]
        A4 --> A5{"question<br>blank?"}
        A5 -- yes --> A6["422, never reaches Claude"]
        A5 -- no --> A7["agent.answer, reused across requests"]
        A7 --> A4
    end

    Agent["<b>QAAgent.answer</b><br>section 6"]
    C5 --> Agent
    A7 --> Agent
```

`get_agent()` is a FastAPI dependency so tests can override it with a mock instead of
needing a real index or Claude client. The third command, `python main.py index`, is the
write half of section 8, and is a required step before either entry point above has
anything to search: neither `ask` nor `api.py` reads `data/index.json` any more, so an
un-indexed corpus just returns empty results rather than a "run ingest first" error the
way the removed JSON backend used to print.

## 8. The Postgres backend

`python main.py index` embeds an extraction index and loads it; `ask` queries it. Both
call `Embedder` directly -- a remote OpenAI API call, not a locally-run
model, so there is no load cost to amortise and no resident process behind it (there used
to be: a BGE-M3-backed daemon, retired after BGE-M3 measured at ~180 chars/second on
ordinary laptop CPUs, making a bulk `index` run of this repo's own corpus a ~63-minute
job; see docs/architecture.md).

```mermaid
flowchart TD
    Idx[("data/index.json<br>chunks + document profiles")] --> Build["<b>build_index_entries</b><br>doc_id = sha256 prefix<br>embed/keyword/display split"]
    Build --> Batch["one embed_texts call per file:<br>entry texts + the profile,<br>profile popped back off the end"]
    Batch --> Emb["<b>Embedder</b><br>OpenAI text-embedding-3-large,<br>truncated to 512 dimensions"]

    Emb --> Up["<b>upsert_document</b><br>delete this doc_id's entries,<br>then insert, one transaction"]
    Up --> Entries[("<b>index_entries</b><br>embedding vector(N), HNSW cosine<br>keyword_tsv generated, GIN")]
    Up --> Docs[("<b>documents</b><br>one row per file<br>doc_summary + summary_embedding,<br>its own HNSW cosine index")]

    Entries --> RRF["<b>PostgresSearchIndex.search</b><br>ts_rank_cd branch + cosine branch,<br>fused by rank, RRF k=60"]
    Docs -->|"LEFT JOIN, display only"| RRF
    Docs --> Lane["<b>search_documents</b><br>vector-only, separate query;<br>only when search found nothing"]
    RRF --> QA["<b>QAAgent</b>"]
    Lane --> QA
```

A real run of the bundled 8-file, 419-chunk corpus (plus 8 document profiles) embeds and
upserts in about 11 seconds end to end.

Points worth knowing before changing this:

- `index` reads the extraction index but still needs the PDFs themselves under `--source`,
  because `doc_id` is a hash of the file. A file it cannot find is reported as `SKIPPED`
  and the rest of the corpus is indexed anyway.
- Each lane keeps `max(top_k * 10, 50)` candidates before fusion, so there is enough of
  each ranking to fuse over. Fusing by rank at all is what avoids normalising `ts_rank_cd`
  and cosine distance onto a common scale, which they do not share.
- The two tables are reached by two different queries on purpose. `documents` feeds the
  ranked list only as a joined display field; the one place it is *ranked* is
  `search_documents`, which runs on its own and never contributes to `search`'s top_k.
- `ensure_schema()` is the only migration mechanism, so a new column on `documents` needs
  both an entry in the `CREATE TABLE` (for a fresh database) and an
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (for every other one).
- The upsert deletes by `doc_id` before inserting, because `chunk_id`'s sequence number is
  positional and shifts on re-extraction. The `documents` row itself is coalesced rather
  than replaced, so a re-index with no profile does not blank one already there.
- `PostgresSearchIndex` implements `document()`/`page()`/`files()` unconditionally, so
  `read_source` always works. Neighbouring-chunk context (the `>>> MATCH:` window) is
  implemented too, but gated behind `attach_context`/`settings.attach_search_context`,
  off by default -- see section 6 and
  [search.md](search.md#unconditional-per-match-and-off-by-default-because-of-it) for why.

## 9. Evaluation, `scripts/run_evals.py`

`docs/roadmap.md` item 10. A hand-maintained question bank fixture is run concurrently
against a live `QAAgent` on the Postgres backend and graded by a second Claude call
rather than by eye, since these are open-ended prose answers that exact-match scoring
can't score.

```mermaid
flowchart TD
    JSON[("tests/fixtures/eval_questions.json<br>hand-maintained EvalQuestionBank:<br>questions[] + short_names{}<br>e.g. 'Burlinson' -&gt; real file name")] --> Run["<b>scripts/run_evals.py</b><br>filter by --tag / --id / --limit"]
    Run --> Pool["ThreadPoolExecutor<br>--workers, default 5"]
    Pool --> PerThread["first question on a thread:<br>build QAAgent(PostgresSearchIndex(Embedder()))<br>+ EvalGrader(short_name_map), then reuse both<br><i>a psycopg connection isn't thread-safe,<br>so each thread gets its own</i>"]
    PerThread --> Answer["agent.answer(question.input, top_k)<br>same agentic loop as section 6"]
    Answer --> Grade["grader.grade(question, answer)<br>one Claude call on claude_model,<br>prompts/eval_judge.prompt,<br>max_tokens = GRADE_MAX_TOKENS (1024)"]

    Grade --> Stop{"stop_reason?"}
    Stop -- refusal --> Fail1["verdict = fail<br>'Judge refused to grade this answer'"]
    Stop -- max_tokens --> Fail2["verdict = fail<br>'response truncated before a verdict<br>could be written; re-run this question'"]
    Stop -- other --> ParseV["parse 'VERDICT: pass|partial|fail'<br>+ 'REASONING: ...'<br>malformed text also defaults to fail"]

    Fail1 --> Result[("EvalResult<br>question, actual_answer, verdict, reasoning")]
    Fail2 --> Result
    ParseV --> Result

    Result --> Collect["collected back into original<br>question order via future_to_index"]
    Collect --> Write[("data/eval_results.json")]
    Collect --> Summary["print pass-rate by tag,<br>then every non-passing result<br>with its one-sentence reason"]
```

Points worth knowing before changing this:

- `tests/fixtures/eval_questions.json` is edited directly; it used to be compiled from
  hand-written Markdown (`docs/eval_questions.md`) by a since-removed conversion step, but
  there is no intermediate to keep in sync any more.
- A malformed or truncated grading response defaults to `fail`, never a silent `pass` --
  the same discard-over-guess choice `MetadataEnhancer` makes on a truncated summary.
  Adaptive thinking's own tokens count against `GRADE_MAX_TOKENS`, so the `max_tokens` check
  is explicit rather than left to fall through as a blank-reasoning fail.
- Runs against `PostgresSearchIndex`, the only backend there is; `read_source`
  (`document()`/`page()`/`files()`, section 8) works fully against it. Neighbouring-chunk
  context is off by default here too (`attach_context` not passed to `PostgresSearchIndex`
  in `_agent_and_grader`, so it follows `settings.attach_search_context`), so a
  `multi-hop` or `footnote` question relies on `read_source` to get surrounding text
  rather than having it attached automatically -- see section 6 and
  [search.md](search.md#unconditional-per-match-and-off-by-default-because-of-it).
- Costs two Claude calls per question, the answer and the grade, plus whatever
  `search_knowledge_base`/`read_source` calls `QAAgent`'s own loop makes underneath the first
  one.
