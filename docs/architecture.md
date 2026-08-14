# Architecture

Why the system is built this way, and the non-obvious details worth knowing before
changing it. For a component-by-component description, see [features.md](features.md).
For the control flow drawn as diagrams, see [flowchart.md](flowchart.md).

## Two systems, one file between them

Ingestion and question answering are deliberately separate, joined only by
`data/index.json`:

```
PDFs  --->  extraction  --->  DocumentChunk[]  --->  index.json  --->  retrieval  --->  Claude
            (minutes,                                (checked in                        (seconds,
             one-off)                                 nowhere, regenerable)              per question)
```

Extraction is slow and idempotent, so it runs once and its output is persisted. Question
answering is fast and stateless, so it loads the index and exits. Neither imports the
other, and `main.py` imports both lazily inside its subcommand functions, so `ask` never
pays the cost of loading Docling or torch.

## Why two extractors

Scientific PDFs are not uniformly born-digital or uniformly scanned. The same document
can have glossy scanned front matter and a clean text layer further in. A single
extractor forces a bad trade:

- pdf_inspector alone silently returns nothing for scanned pages.
- Docling alone runs OCR and layout models over every page of every document, which
  takes minutes per file for pages that need none of it.

So `PdfExtractor` handles what it can read and reports the rest, and
`DocumentExtractor` is built lazily and called only for those pages.

### The bug this design exists to prevent

An earlier PyMuPDF-based extractor decided whether a document had a text layer using a
**document-wide average** of text per page. `1-s2.0-S0140988325000672-main.pdf` has no
text on pages 1 to 7 and plenty on pages 8 to 16, so the average passed the threshold,
the document took the fast path, and pages 1 to 7 produced no chunks at all. No error,
no warning, just 19 chunks where there should have been 50. Any classification that is
not per page can reintroduce this, which is why
`test_mixed_document_reports_its_scanned_pages_needing_ocr` pins the exact expected
page list for that fixture.

## Page numbering

Page numbers are citations, so they have to be right. Two conventions collide:

| Source | Convention |
| --- | --- |
| `pdf_inspector` `PageMarkdown.page` | **0-indexed** |
| `pdf_inspector` `PagesExtractionResult.pages_needing_ocr` | **1-indexed** |
| Docling `item.prov[0].page_no` | 1-indexed |
| `DocumentChunk.page_number` | 1-indexed, `0` means unknown |

The asymmetry within pdf_inspector's own API is documented in its shipped `.pyi` stub
and is easy to get wrong in either direction. `PdfExtractor` converts with
`page.page + 1` and passes `pages_needing_ocr` through untouched.

Docling's `page_range=(start, end)` preserves **absolute** page numbers in the converted
result, verified empirically rather than assumed. That is the fact the whole merge
strategy rests on: chunks from a partial conversion carry the same page numbers they
would in a full one, so fast-path and OCR chunks can be concatenated and sorted by
`page_number` with no offset arithmetic.

Chunk IDs are assigned per conversion call, so a second `page_range` pass restarts at
`c0` and collides with the first. `extract_chunks_for_pages` renumbers over the merged
list to fix this.

## Chunking

Both extractors accumulate text up to `max_chars` (default 3000) and flush on a section
heading or a page change, so a chunk never straddles a page and its citation stays
unambiguous. `max_chars` is a hard bound, not a target: a single line longer than the
limit, typically a wide Markdown table row, is split on word boundaries, falling back to
a hard character split for a single overlong word.

Tables become their own chunks, and never share one with the prose around them. That is
what makes `index_entry_builder`'s split coherent: it embeds a table's summary but
keyword-searches its Markdown, and neither choice is right for a chunk that is half
prose. A table is also never split from its header. If one is too wide to fit `max_chars`
it is divided into parts that each repeat the caption and header row, because a part that
is a run of numbers with no column names cannot be attributed to a variable, and a reader
or a model can attribute it to the wrong one.

`kind` is derived by counting a chunk's table rows rather than testing its first line.
The first-line test was wrong in both directions on real documents: a genuine table is
introduced by its caption, and on these fixtures pdf_inspector puts the running header
ahead of that again, so whole tables were classified as prose while header-less fragments
were classified as tables. Keeping tables intact removes the fragments, and counting rows
removes the dependence on what happens to sit at the top of the chunk.

## Why summarising is opt-in and separate

`MetadataEnhancer` writes an LLM gloss into `DocumentChunk.summary`. It exists because
raw text is sometimes a poor embedding target: a table's value for keyword search is its
exact figures, but what should be embedded is a sentence describing what the table shows.

It is deliberately not part of extraction. Extraction is free, offline and repeatable;
summarising costs one Claude call per chunk. Folding it into an extractor would make an
expensive, network-dependent, non-deterministic step happen as a side effect of a cheap
one. So it lives behind `--enhance-metadata` on `ingest`, off by default, and the caller
decides.

Which chunks deserve it is answered by `kind`, not left open: `enhance_chunks()` defaults
to summarising `kind="table"` chunks only (`DEFAULT_SUMMARISED_KINDS`), because
`index_entry_builder.py` is the only reader of `summary` and it only reads it for tables --
a table's dense-search fallback is otherwise truncated Markdown, whereas prose already
embeds fine on its own and a figure chunk's text is already an LLM description. That is 36
calls rather than 384 on the bundled fixtures. `enhance_chunks(chunks, kinds=(...))` widens
it if a future reader wants summaries for other kinds too. Still one call at a time with no
batching.

### Why figure captioning is the one exception

`--enhance-metadata` and `--extractor vision` stay opt-in for exactly the reason above: they
trade cost for accuracy, and accuracy-for-cost is the caller's call to make. Figure
captioning (`utils/figure_captioner.py`) started the same way, but graduated to `ingest`'s
default (`--no-caption-figures` opts back out) once it stopped being only an accuracy trade.

Both extractors deliberately skip `PictureItem`/picture regions, on the assumption that a
genuine picture (chart, photo, logo) has no text worth extracting -- its values are baked
into pixels, reachable only via a vision call. That assumption broke on a real fixture:
Docling's layout model boxed nearly an entire page of real body text (the abstract and
introduction of `1-s2.0-S0140988325000672-main.pdf`, its ARTICLE INFO sidebar aside) as one
giant misclassified picture, and since nothing in the extraction path has a branch for
`PictureItem` at all, that text vanished from the index with no error or warning (see
`docs/roadmap.md`, item 4 and the eval-run follow-up entry). A skipped picture that is
genuinely a picture costs nothing lost; a skipped picture that is secretly a paragraph the
layout model misfired on is silent data loss, and there was no signal anywhere to tell the
two apart.

`FigureCaptioner`'s prompt now asks the model to report substantial running body text it
finds this way as its own item (`classification: "text"`), transcribed verbatim, and it
becomes a `kind="text"` chunk rather than a figure caption, so recovered text merges into
search like any other extracted text. `likely_figure_pages()` still bounds which pages pay
the Claude call (a plausible reason to hold an image, not every page), and
`_drop_duplicate_recovered_text` in `main.py` guards against re-transcribing a page the
primary extractor already covered well. Net effect: this one flag moved from "cost traded
for chart-value accuracy" to "default safety net against a confirmed extraction-loss failure
mode," which is why it alone breaks the opt-in pattern the rest of this section describes.

## Why document profiles are not chunks

`DocumentSummarizer` writes one profile per source file. It answers a different question
from a chunk summary: not "what should be embedded for this table", but "is the model
even in the right document, and is this extract worth pulling the whole file for".

That second question is worth answering because of `read_source`. Without a profile, the
model decides whether to pull a whole document from a truncated extract alone, which is a
guess, and the wrong guess is expensive in both directions: a needless whole-document read
burns context, and a skipped one loses the answer. A profile makes that call informed, and
it is also the cheapest way to rule a document *out*.

The design decision is that a profile is **never** ranked against chunks. It is tempting
to store one as a `granularity='document'` row in `index_entries`, since `IndexEntry`
already has the enum value for it. Two things argue against it:

- A profile mentions every topic its document touches, so it matches a large share of
  queries at middling strength. In one ranked list it displaces the chunk that holds the
  actual figure, and dense retrieval is the worst case. Avoiding that inside one table
  means every present and future query carrying a `granularity = 'element'` filter, and
  the first one that forgets it degrades retrieval silently.
- `IndexEntry.page_start` is `ge=1` and `citation` formats from it, so a profile in that
  table needs a fabricated page range that reads exactly like a real citation.

So the summary lives on the `documents` table, one row per file, which already existed for
the `doc_id` foreign key. The separation is then structural rather than a filter someone
has to remember, and a profile has no page range to misuse.

That leaves two lanes with different jobs. The **display** lane joins the profile onto
results from its own file, printed once per document and marked as not citable, which is
where nearly all the value is. The **search** lane is
`PostgresSearchIndex.search_documents()`, its own vector query over its own HNSW index,
used only when chunk search returned nothing, so it never competes for a slot in a ranked
list. Fusing the two lanes into one score is possible but is not done: it needs tuning
against the question bank (roadmap item 10) rather than a guessed weight.

Generation sits in `ingest`, behind `--summarise-documents`, for the same reason
`--enhance-metadata` does, and runs after figure captioning and chunk summarising so the
profile describes the final chunk list. `index` then only embeds the stored text, so the
Claude call happens once and both backends describe a document identically.

## Why embeddings are a remote API call, not a local model

This used to run BGE-M3 locally via FlagEmbedding, with a resident daemon
(`utils/embedding_daemon.py`, a stdlib `ThreadingHTTPServer`) so the model's own load
cost -- about 22 seconds, even from a warm cache -- was paid once per machine rather
than once per one-shot CLI invocation, and an `EmbeddingClient` that spawned the daemon
as a detached process on first use and polled `/health` until ready.

That daemon was retired, not just replaced, because of what it turned up: measured
directly against this repo's own corpus, BGE-M3 on an ordinary laptop CPU (Intel
Core i5-1135G7, 4 cores, no CUDA build of torch, no GPU) ran at roughly 180
characters per second. A single query embeds fine at that rate. Embedding the full
corpus at ingest time does not: 670,000 characters of `embed_text` across 8 files,
plus 8 document profiles, worked out to about 63 minutes. A single file's ~64 entries
(67,576 characters) alone measured past 300 seconds without finishing. The daemon's
22-second load cost was never the bottleneck; the encoding itself was, and no amount
of daemon-vs-in-process redesign changes that, since it is the same `model.encode()`
call either way, on the same CPU. (Investigating this also surfaced a real, separate
bug in the daemon, since fixed before it was deleted: `ThreadingHTTPServer` ran every
request on its own thread with no lock around the shared model, so two overlapping
`/embed` calls competed for the same cores and ran each other slower, and a client
that gave up and closed its socket did not stop the server-side call it abandoned --
so a naive retry added a second concurrent call on top of the first rather than
replacing it.)

`utils/embedder.py`'s `Embedder` now calls OpenAI's API directly
(`text-embedding-3-large`, truncated to `settings.embedding_dimensions` = 512 via the
API's own `dimensions` parameter, which OpenAI's v3 models are trained to support
without a proportional quality loss). A remote call has no local model to load, which
was the daemon's entire reason to exist, so there is nothing left to amortise and
nothing left to run as a resident process: every caller just constructs an `Embedder`
directly. The real corpus now embeds in about 11 seconds, not 63 minutes.

`OPENAI_API_KEY` is read the same way `CLAUDE_API_KEY` is: from `src/.env` or the real
OS environment (both work; an OS-level value takes priority). If `settings.
openai_api_key` ends up unset for any reason, the `openai` SDK's own client falls
back to checking the environment variable itself.

## The retriever protocol

`QAAgent` depends on exactly one method:

```python
search(query: str, top_k: int) -> list[SearchResult]
```

That is the seam for replacing retrieval: only `PostgresSearchIndex` implements it today,
but nothing in `QAAgent` depends on Postgres specifically, so a future backend only has to
satisfy this one method. For the same reason `SearchIndex` persists only chunks and no
ranking statistics: the on-disk format is a content store, independent of whatever
retrieval method reads it, and `DocumentChunk.embedding` is already reserved.

Everything past `search()` is optional and checked with `hasattr` before use, so a
backend can implement one method and still work, losing a capability rather than
raising: `document()`, `page()` and `files()` back `read_source`, and `search_documents()`
backs the document-lane fallback on an empty search. `PostgresSearchIndex` implements all
of these. Document profiles deliberately do **not** follow the same optional-method
pattern, because attaching them was not worth a second protocol method: they ride on
`SearchResult.doc_summary`, so a backend that has them fills a field and one that doesn't
leaves it `None`.

## Why the agent searches instead of being handed context

The obvious design retrieves once on the raw question and stuffs the results into the
prompt. Keyword search punishes that: a natural-language question carries stopwords and
phrasing that do not appear in the source text, and Postgres FTS scores an exact-token
mismatch as no match at all regardless of what the dense side finds. Giving Claude a
`search_knowledge_base` tool
instead lets it rewrite the question into likely source keywords and search again with
narrower terms when the first attempt misses. The cost is a capped loop
(`MAX_TOOL_ITERATIONS = 8`) rather than a single round trip. The cap is higher than a
search-only loop would need, because the `read_source` escalation below costs an extra
round trip on top of the search that motivated it.

Two response paths need explicit handling: `stop_reason == "refusal"` must be checked
before reading content, because a refusal returns HTTP 200 with empty content in
that case, and exhausting the iteration budget returns a plain message rather than
looping.

A narrowed re-query commonly returns a result the model has already seen, since the
narrower keywords still hit the same chunk from a different angle. Handing that chunk
back unchanged both wastes context and reads as corroborating evidence when it is the
same evidence a second time, so `QAAgent` tracks `_seen_chunk_ids` per `answer()` call and
strips already-seen chunks from a fresh search's results before formatting them. A query
that turns up nothing new gets `DUPLICATE_RESULTS_MESSAGE` rather than the "no matching
extracts" message an actually-empty search returns, because the distinction matters to
what Claude should try next: no match at all calls for different keywords, an all-duplicate
result calls for reading what is already in front of it instead of searching again.

The same problem exists one level down, in the neighbouring-chunk context a search
result can carry when that's turned on (`PostgresSearchIndex._attach_context`, gated by
`attach_context`/`settings.attach_search_context`, off by default -- see
[search.md](search.md#4-context-expansion-context_beforecontext_after)): a chunk shown
once as context for one match can resurface as context for a *different* match in a
later search, since the match-level dedup above only ever looked at each result's own
`chunk_id`, never at its neighbours'. `_format_results` closes that gap by checking
`self._seen_chunk_ids` for each context chunk too and dropping ones already sent, adding
newly-shown ones to the same set -- which is also why it is an instance method rather
than a `@staticmethod` now, since it needs to read and mutate that per-`answer()`-call
state. The match itself is never dropped this way, only a redundant context line beside
it; a duplicate match is still caught earlier, by the check in `_run_tools` this section
describes. Note that this dedup only ever stops a chunk being *repeated*: whenever
context expansion is on, attaching it in the first place is unconditional per match,
with no check for whether that match already stands on its own, so the first time a
chunk's neighbour is fetched it is sent regardless -- which is exactly why the feature
defaults to off. See
[search.md](search.md#unconditional-per-match-and-off-by-default-because-of-it)
for the token-cost tradeoff that follows from that.

## Why there's an escalation tool, and how context expansion was rebuilt for Postgres

Chunking splits on a character budget (`max_chars`), not on meaning, so the chunk that
scores highest for a query is very often not self-contained: it starts mid-sentence, it
has table rows with the header one chunk earlier, or it references a figure described
two chunks later.

**`read_source`** is `QAAgent`'s answer for going beyond the matched chunk itself. Some
questions need a whole document (comparing two sections of the same paper) or a whole
page (a footnote, a figure caption) rather than one more chunk either side. Rather than
make `search_knowledge_base` return more and more context by default -- which would
bloat every answer to serve the few questions that need it -- `read_source` lets Claude
ask for exactly that, only when a search result looks decisive but incomplete. It degrades
rather than errors when the retriever cannot support it: `_read_source` checks for
`document()`/`page()`/`files()` on the index and returns a plain "not supported" message
if they are absent, so a retriever only implementing `search()` still works, just without
the escalation path. `PostgresSearchIndex` implements all three, backed by exact
`file_name` lookups against `index_entries`.

**Context expansion, rebuilt rather than carried over, and now opt-in.** The JSON-backed
`LexicalSearchIndex` used to attach the chunks immediately surrounding a match
(`SearchResult.context_before`/`context_after`, after ranking so neighbours never
influenced score or displaced a genuine `top_k` hit) via plain list adjacency, always,
with no way to turn it off, and `QAAgent._format_results` still knows how to render them,
delimiting the match with `>>> MATCH:` and labelling neighbours `...context`. That index
was removed once Postgres became the only backend (see roadmap.md), and for a while
nothing replaced the context it used to attach. `PostgresSearchIndex.search()` closes
that gap with `_attach_context`, but gated behind `attach_context` (constructor argument,
falling back to `settings.attach_search_context`) rather than unconditional: when it
runs, one query resolves each hit's immediate page-adjacent neighbour chunk_id via
`LAG`/`LEAD` windowed `PARTITION BY file_name ORDER BY page_start, chunk_id` (never
globally, so a match at a file boundary can't pick up another document's chunk as its
"neighbour"), and a second fetches the text for whichever neighbour ids actually exist.
Two queries regardless of how many results came back, not one per hit, and both run
after the hits are already scored, so a neighbour still can never influence ranking. The
setting defaults to off because the cost is real and unconditional per match whenever it
does run (see the token-cost discussion above); `read_source` is the fallback either
way for what this can't give for free regardless of the setting: a whole page or
document, when one more chunk either side isn't enough (a footnote, a distant table
header, comparing distant sections).

## Environment traps

- **`TORCHDYNAMO_DISABLE=1`** is set at the top of `utils/document_extractor.py`, before
  the Docling imports. RapidOCR's torch backend tries to JIT-compile through
  torch.compile and inductor, which needs an MSVC C++ compiler that many Windows
  machines do not have on PATH. Running eager costs a slightly slower first OCR call and
  nothing else. Anything importing Docling outside this module needs the same variable
  set first.
- **Console encoding**: `main.py` calls
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`, because Windows consoles
  default to a narrow codepage such as cp1252 and would otherwise crash while printing
  an otherwise successful answer.
- **Working directory**: `Settings` reads `env_file=".env"` and the default index path is
  relative, so both resolve against the current directory. Commands must run from `src/`.
  Tests are insulated from this by `pythonpath = ..` in `tests/pytest.ini`.

## Model choice

`claude-sonnet-5` with `max_tokens` 4096 for answering and for the chunk and document
summaries. Adaptive thinking is on by default on this model, so no `thinking` parameter is
passed, and `temperature` and `top_p` are never sent because this model rejects them.

The two vision jobs are split off onto their own settings, `claude_vision_model` for page
transcription and `claude_figure_model` for figure captioning, both `claude-sonnet-5`. The
split is not about capability tiers in the usual sense: neither job needs the reasoning
`claude_model` is paying for, since the page is in front of the model. What they need is
resolution. `PageRenderer` targets 2576px on the long edge because that is what the
high-resolution vision tier reads at full fidelity, and a model below that tier downsamples
to roughly 1568px. For transcription that costs strokes on 8-9pt two-column body text, which
is the same failure `MIN_LONG_EDGE_PX` guards against from the other direction. For figures
it was measured rather than assumed, and the cost was worse than expected: a cheaper model
read the values off a six-panel map correctly but could not attach any of them to a region,
and lost the signs the legend encodes by colour, which leaves a chunk full of numbers that
answers nothing.

They stay two settings rather than one because figures are where a cost-sensitive run would
trade that accuracy away, and `figure_captioner.py` keeps its call shape free of `effort` and
`thinking` specifically so that swap stays a one-line change. That is why its call must not
be copied from `vision_extractor.py`, which does pass `effort`.
