# Search

How a question becomes an answer: the two tools Claude is given, the query rewriting
that makes them worth calling, the hybrid SQL that ranks a match, and the schema that
makes the SQL possible. For what each file does in isolation see
[features.md](features.md), for why it is built this way see
[architecture.md](architecture.md), and for the same ground as diagrams see
[flowchart.md](flowchart.md#6-question-answering-the-agentic-loop). This doc goes deeper
than any of those three on the retrieval mechanics specifically: the actual SQL, the
actual fusion formula, and the actual schema.

There is one retrieval backend, `utils/postgres_search_index.py`'s `PostgresSearchIndex`.
An earlier keyword-only backend (`LexicalSearchIndex`, a BM25 index over a local JSON
file) existed and has been deleted; everything below is Postgres plus pgvector.

## 1. The two tools, and why Claude writes the keywords

`QAAgent` does not embed the question and search once. It gives Claude two tools and
lets it decide what to do with them (`utils/qa_agent.py`):

- **`search_knowledge_base(query, top_k?)`.** Its tool description tells Claude to
  rewrite the question into "the keywords most likely to appear in the source text, not
  the question itself." This matters for the keyword half of the hybrid search below: a
  natural-language question is full of stopwords and phrasing that will never appear
  verbatim in a scientific paper, so Postgres full-text search scores it badly. Claude
  can call this tool more than once, narrowing or changing the keywords if the first
  attempt misses. There is nothing automatic about the retry: the tool's own description
  is what tells the model to try again with different keywords rather than giving up
  after one call.
- **`read_source(file_name, page?)`.** An escalation tool for when a search result
  "looks decisive but incomplete": cut off mid-sentence, a table with no header, a
  footnote or figure the extract doesn't show, or a question that needs several facts
  spread across one document. Omitting `page` reads the whole document. The system
  prompt (`prompts/qa_system.prompt`) also tells Claude to reach for this tool *instead
  of* searching whenever a question names a specific file, because
  `search_knowledge_base` matches content, never file names, and has no way to say a
  file doesn't exist. This is a real fix for an real, observed bug: content search has
  no relevance floor (section 3), so a filename-shaped query used to come back with a
  handful of confidently-formatted but unrelated extracts, and the model read that as
  proof the file was missing. `read_source` bypasses ranking entirely and can give a
  definitive yes/no.

Both tools are plain JSON-schema tool definitions passed to `client.messages.create`
alongside `prompts/qa_system.prompt`, which explains the citation format, the
`>>> MATCH:`/`...context` markup search results arrive in (section 4), and how to read a
document-profile preamble as background rather than as a citable source (section 6).

## 2. The agentic loop

`QAAgent.answer(question, top_k)` runs up to `MAX_TOOL_ITERATIONS` (8) round trips:

1. Call Claude with the running message list, both tools, and the system prompt.
2. `stop_reason == "refusal"` is checked before anything else, since a refusal is an
   HTTP 200 with empty content: return `REFUSAL_MESSAGE`.
3. Any other `stop_reason` besides `"tool_use"` means Claude is done: return the text
   blocks as the final answer.
4. Otherwise, run every `tool_use` block in the response (`_run_tools`), append the
   assistant turn and the tool results to the message list, and loop.
5. Exhausting all 8 iterations without a final answer returns `SEARCH_BUDGET_MESSAGE`.

Two things happen inside `_run_tools` that are easy to miss reading the tool
descriptions alone:

- **Chunk-id dedup, scoped to one `answer()` call.** `_seen_chunk_ids` is a set on the
  agent instance, reset at the start of every `answer()`. After a `search_knowledge_base`
  call returns non-empty results, they're filtered to `fresh` -- chunks not already in
  `_seen_chunk_ids` -- before formatting. A narrowed re-query very often re-surfaces a
  chunk already in the conversation from a different angle; resending it would waste
  context and read as new corroborating evidence when it's the same evidence again. If
  every result was already seen, Claude gets `DUPLICATE_RESULTS_MESSAGE` ("see those
  results above") rather than the plain no-match message, since the query did match --
  the right next move is reading what's already there, not trying different keywords.
  Numbering in the formatted output (`[1]`, `[2]`, ...) restarts from 1 over the
  surviving fresh chunks each call; it's local to that tool result, never a running
  count across the conversation.
- **`read_source` degrades instead of erroring.** `_read_source` checks
  `hasattr(self._index, name)` for `document`, `page` and `files` before using them.
  `PostgresSearchIndex` implements all three (section 5), but the check exists so a
  future retriever implementing only `search()` still works, just without the
  escalation path, rather than throwing.

## 3. Hybrid search: the RRF query

`PostgresSearchIndex.search(query, top_k=5)` is where a query becomes ranked rows. Empty
or whitespace-only queries return `[]` immediately without calling the embedder. Otherwise:

```python
query_vector = self._embedder.embed_texts([query])[0]
candidate_limit = max(top_k * 10, 50)
```

One embedding call per search (see section 7), and a candidate pool sized well above
`top_k` -- generous on purpose, so there's enough of each ranking left to fuse over even
when `top_k` itself is small (asking for `top_k=1` shouldn't mean each branch only gets
to nominate one candidate).

The query itself, `_RRF_SEARCH_SQL`, is one statement with two independently-ranked CTEs
and a fusion step:

```sql
WITH keyword_ranked AS (
    SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(keyword_tsv, query) DESC) AS rank
    FROM index_entries, plainto_tsquery('english', %(query_text)s) AS query
    WHERE keyword_tsv @@ query
    LIMIT %(candidate_limit)s
),
vector_ranked AS (
    SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %(query_vector)s) AS rank
    FROM index_entries
    ORDER BY embedding <=> %(query_vector)s
    LIMIT %(candidate_limit)s
),
fused AS (
    SELECT chunk_id, SUM(1.0 / (60 + rank)) AS score
    FROM (
        SELECT * FROM keyword_ranked
        UNION ALL
        SELECT * FROM vector_ranked
    ) AS combined
    GROUP BY chunk_id
)
SELECT index_entries.*, documents.doc_summary AS doc_summary, fused.score AS fused_score
FROM fused
JOIN index_entries USING (chunk_id)
LEFT JOIN documents USING (doc_id)
ORDER BY fused.score DESC
LIMIT %(top_k)s
```

### The keyword branch

`keyword_tsv` is a **generated** column on `index_entries`:
`GENERATED ALWAYS AS (to_tsvector('english', keyword_text)) STORED`. Postgres maintains
it automatically from `keyword_text` on every insert; nothing ever writes to it
directly. `plainto_tsquery('english', query_text)` turns the raw query string into a
`tsquery` with English stemming and stopword removal and an implicit AND between terms
(no `tsquery` operator syntax reaches this far -- the query is Claude's own rewritten
keywords, plain text). `keyword_tsv @@ query` is a real match predicate: a row that
shares no lexeme with the query is excluded outright, not merely ranked low.
`ts_rank_cd` (cover density ranking) then orders the matching rows, rewarding matched
terms that appear close together over the same terms scattered far apart in the
document.

That `to_tsvector` keeps the sign on a number: `-0.59%` indexes as the lexeme `-0.59`,
so a query for `0.59` alone will not match it. Worth knowing if a table value seems to
vanish from keyword search.

### The vector branch

`embedding <=> %(query_vector)s` is pgvector's cosine **distance** operator: it returns
`1 - cosine_similarity`, so **smaller means more similar**, and
`ORDER BY embedding <=> query_vector` (ascending, the default) puts the nearest neighbour
first. This is why `vector_ranked`'s `ORDER BY` and the `row_number()` window use the
same expression: they have to agree on which direction is "better."

Unlike the keyword branch, there is no match predicate here -- every row is a candidate,
ranked only by distance. pgvector's HNSW index (`vector_cosine_ops`, see section 5)
makes that ranking sublinear rather than a full table scan, but it never *excludes* a
row for being a bad match. That asymmetry is a known characteristic, not a bug: the
vector branch alone has no relevance floor, so a query with no genuine match still
returns its `candidate_limit` nearest neighbours, however weak. (This is exactly what
made the filename-shaped-question bug in section 1 possible before `read_source` grew
its own accessors: a filename embeds to some arbitrary point in vector space and still
gets back a full page of "candidates.")

`vector_cosine_ops` specifically, rather than `vector_ip_ops` (inner product) or
`vector_l2_ops` (Euclidean), because OpenAI documents its embeddings as L2-normalised:
for unit-length vectors cosine similarity and inner product rank identically, so cosine
is chosen as the more intuitive of the two equivalent options, not because it's
uniquely correct here.

### Fusion: by rank, not by score

`ts_rank_cd` and cosine distance are not on comparable scales -- one is a
document/term-frequency-dependent ranking score with no fixed range, the other is a
bounded distance -- so blending them with, say, a weighted sum would need one or both
normalised onto a common scale first, which is itself a tuned, corpus-dependent choice.
Reciprocal Rank Fusion sidesteps that by throwing away the scores and fusing on **rank**
alone:

```
score(chunk) = sum over branches the chunk appears in of  1 / (60 + rank_in_that_branch)
```

`60` (`_RRF_K`) is the constant from the original RRF paper: high enough to damp the
gap between rank 1 and rank 2 so one branch's top pick doesn't automatically dominate,
and it needs no tuning per query or per corpus. A chunk ranked 1st on keywords and 3rd
on vector similarity scores `1/61 + 1/63`; a chunk that only one branch found at all
contributes just that one term. `UNION ALL` before the `GROUP BY` is what lets a chunk
found by both branches accumulate both terms and rank above a chunk only one branch
surfaced -- that additive boost for cross-branch agreement is the entire point of
fusing this way rather than, say, taking each branch's best rank alone.

The final `SELECT` joins the fused scores back to the full `index_entries` row (so
`display_text`, `heading_path`, `element_types`, etc. all come along) and left-joins
`documents` purely to carry `doc_summary` onto the result for display -- that join
never affects which rows were selected or their order. `search()` always returns
exactly `top_k` rows if the corpus has that many (padding out with its weakest
candidates when fewer are genuinely relevant), since RRF fusion has no absolute
"good match" threshold either, only a relative ordering. See `docs/roadmap.md`'s
reranking follow-up for the standing idea to fix that.

### Reconstructing rows

Rows come back as `DocumentChunk` objects, not `IndexEntry` -- that's what `QAAgent`
consumes. `display_text` becomes `chunk.text`, `heading_path` becomes `chunk.headings`,
and `kind` is recovered from `element_types` as `"table"` if that tag is present,
`"text"` otherwise (a retrieved figure chunk also comes back tagged `"text"` this way;
harmless, since nothing downstream branches on `kind` after retrieval -- it only matters
at index-build time, section 6).

## 4. Context expansion: `context_before`/`context_after`

Chunking splits on a character budget, not on meaning (`docs/architecture.md`), so the
chunk that scores highest is very often not self-contained: it starts mid-sentence, or a
table's rows land in a different chunk from its header.

**Off by default.** `PostgresSearchIndex(embedder, attach_context=...)` gates this whole
feature; when the argument isn't given, it falls back to
`settings.attach_search_context`, which is `False` out of the box. `main.py ask` exposes
a `--attach-context` flag that forces it on for one run regardless of the setting; set
`ATTACH_SEARCH_CONTEXT=true` in `.env` to change the default everywhere (CLI and
`api.py` both read the same setting). The rest of this section describes what happens
when it's turned on; see the subsection below for why off is the default.

When enabled, after the RRF query returns its scored rows, `_attach_context` runs two
more queries to fill in `SearchResult.context_before`/`context_after`, one immediate
neighbour each side:

```sql
WITH ordered AS (
    SELECT chunk_id,
           LAG(chunk_id) OVER (PARTITION BY file_name ORDER BY page_start, chunk_id) AS prev_chunk_id,
           LEAD(chunk_id) OVER (PARTITION BY file_name ORDER BY page_start, chunk_id) AS next_chunk_id
    FROM index_entries
    WHERE file_name = ANY(%(file_names)s)
)
SELECT chunk_id, prev_chunk_id, next_chunk_id
FROM ordered
WHERE chunk_id = ANY(%(chunk_ids)s)
```

`LAG`/`LEAD` are windowed `PARTITION BY file_name`, so a match sitting at the very start
or end of a file gets `NULL` on that side rather than picking up another document's
chunk as a false neighbour -- raw list adjacency would get this wrong at a file
boundary. A second query (`_CHUNKS_BY_ID_SQL`) then fetches display text for whichever
neighbour ids actually came back non-null; if none did, that second query is skipped
entirely. This is exactly two queries regardless of how many results the RRF query
returned, not one pair per result.

Context is attached strictly *after* ranking, to already-scored `SearchResult`s: a
neighbour never contributes to a score and never displaces a genuine match from
`top_k`. `QAAgent._format_results` renders the match delimited with `>>> MATCH:` and the
neighbours labelled `...context`, so Claude can read a complete sentence or a table's
header without citing the neighbour's page by mistake -- the citation always belongs to
the marked match.

### Unconditional per match, and off by default because of it

Turned on, `_attach_context` runs on every non-empty `search()` call, for every result
in it. There is no check anywhere for whether the matched chunk already reads as
complete on its own; a match that's already fully self-contained gets its neighbours
fetched and sent to Claude exactly the same as one that isn't. That's a real,
unconditional-per-match token cost, not a hypothetical one: each neighbour can be up to
`max_chars` (3000 characters, roughly 750 tokens), bounded to one chunk each side per
match, so a `search_knowledge_base` call at the default `top_k=5` where every result has
both neighbours (none at a file boundary) can add up to ~10 extra chunks, on the order
of 7500 tokens, on top of the 5 matches themselves. That's the reason
`settings.attach_search_context` defaults to `False` (section 4's top): the feature
itself has no way to tell, per chunk, whether that cost was worth paying.

The alternative -- deciding per chunk whether context is worth attaching -- would need
either a model call to judge "is this complete" (defeating the purpose: spending a
round trip to decide whether to spend a smaller number of tokens) or a heuristic
unreliable enough to risk the failure this feature exists to prevent: a chunk that
*reads* as complete but silently references a table header or a footnote it doesn't
show, producing a confidently wrong answer built from a partial extract. Compare that
to `read_source` (section 5), the tool Claude *does* have to explicitly ask for: it
exists precisely because that's a full extra round trip (another `messages.create` call,
input tokens for the whole conversation history again, output tokens for another tool
call), which is a much larger cost than an inline extra chunk or two. So turning this
feature on trades a smaller, certain, per-match overhead for a lower chance Claude ever
needs to reach for `read_source` on a merely-truncated match; turning it off (the
default) leans on `read_source` to catch those cases instead, at the cost of an extra
round trip on the ones it does catch. This isn't new to the Postgres rewrite either: the
original BM25-backed index attached context this same unconditional-per-match way,
though it had no equivalent off switch.

What gets deduplicated, when this is on, is repetition *across calls* within one
`answer()`: `_seen_chunk_ids` (section 2) tracks every chunk id already sent to the
model, whether as a match or as another result's context, and `_format_results` drops
(rather than re-renders) a context line whose chunk was already shown -- as an earlier
match, or as an earlier result's context, in this same call or an earlier one in the
conversation. That closes the gap where the same neighbour text could otherwise repeat
indefinitely across several `search_knowledge_base` calls in one question; it does not
reduce the per-call cost of attaching context that turns out to be unneeded the first
time it's sent, which is the cost the default-off setting exists to avoid paying at all
until it's asked for.

## 5. The escalation lane: `document()`/`page()`/`files()`

`read_source`'s three accessors do no ranking at all -- exact `file_name` (and
optionally `page_start`) lookups against `index_entries`:

```sql
-- document(file_name)
SELECT chunk_id, file_name, page_start, display_text, heading_path, element_types
FROM index_entries
WHERE file_name = %(file_name)s
ORDER BY page_start, chunk_id

-- page(file_name, page_number)
SELECT chunk_id, file_name, page_start, display_text, heading_path, element_types
FROM index_entries
WHERE file_name = %(file_name)s AND page_start = %(page_number)s
ORDER BY chunk_id
```

Ordering by `page_start` then `chunk_id` reconstructs reading order for a whole document
the same way a sequential extraction would have produced it. `files()` is a distinct
`SELECT file_name FROM documents ORDER BY file_name` -- `documents`, not
`index_entries`, since it's one row per file regardless of chunk count, and it's what
backs the "indexed files are: ..." hint `_read_source` appends when a lookup finds
nothing.

## 6. The document lane: `search_documents`

A separate, vector-only query against a separate table:

```sql
SELECT file_name, doc_summary, page_count, 1 - (summary_embedding <=> %(query_vector)s) AS score
FROM documents
WHERE summary_embedding IS NOT NULL AND doc_summary IS NOT NULL
ORDER BY summary_embedding <=> %(query_vector)s
LIMIT %(top_k)s
```

Its own HNSW index, on `documents.summary_embedding`, is what keeps this from ever
competing with `index_entries` search for a slot in the ranked results -- it's a
structurally separate query, not a `granularity` filter every future query would have to
remember to apply. Vector-only because a document profile is prose written to be
semantically matched (what a document covers, how it's structured) and holds none of the
exact figures that make fusing in keyword search worthwhile for a chunk.

`QAAgent` calls this only through `_no_results_message`, and only when
`search_knowledge_base` came back with zero rows -- never as a first-class ranked lane.
What it returns is explicitly *routing* information: a file name and profile to point
`read_source` at, never a citable answer. The score here (a cosine-similarity-derived
`1 - distance`, roughly 0..1) is not comparable to `search()`'s RRF-fused score (a sum of
reciprocal ranks) despite both fields being named `score` -- they come from different
formulas over different things.

`WHERE summary_embedding IS NOT NULL AND doc_summary IS NOT NULL` is why this lane
returns `[]` on a corpus that never ran `ingest --summarise-documents`: that's the normal
state, not an error.

## 7. Embeddings

Every vector in this system, query-time or index-time, comes from one place:
`utils/embedder.py`'s `Embedder`, a thin wrapper over OpenAI's embeddings API
(`text-embedding-3-large`), truncated to `settings.embedding_dimensions` (512) via the
API's own `dimensions` parameter rather than truncated after the fact. OpenAI's v3
embedding models are trained so that a shorter requested dimension still outperforms the
older `ada-002` at full length, which is what makes asking for fewer dimensions a
reasonable trade rather than a lossy hack: it buys a smaller vector column and a faster
HNSW build/search.

`PostgresSearchIndex.search()` calls `embed_texts([query])[0]` once per call: one
network round trip, embedding just the (Claude-rewritten) keyword string, not the
original question. The same `Embedder` embeds chunk text at index time
(`main.py index`) and document profiles, so query-time and index-time vectors are
produced by the exact same model and dimension -- necessary for cosine distance between
them to mean anything.

There is no local embedding model and no resident daemon. An earlier version of this
repo ran BGE-M3 locally specifically to avoid paying network latency per query, but a
bulk `index` run over this repo's own ~670,000-character corpus worked out to roughly an
hour at BGE-M3's measured ~180 characters/second on ordinary laptop CPUs. A remote API
call has no model-load cost to amortise, which was the entire justification for the
daemon, so once embeddings moved to OpenAI's API there was nothing left to run as a
resident process (`docs/architecture.md` has the full measurement).

## 8. The schema

The full column-by-column reference, the `doc_id`/`chunk_id` identity scheme, the
delete-then-insert upsert that keeps a re-`index` run from duplicating rows, the three
indexes, and the `register_vector()`/extension setup all live in
[docs/postgres.md](postgres.md) now, since none of that is specific to ranking. The one
fact this section (and section 3 above) leans on directly: `index_entries.keyword_tsv`
is a **generated** column, `GENERATED ALWAYS AS (to_tsvector('english', keyword_text))
STORED`, maintained by Postgres on every insert, never written to directly.

## 9. Why a table's three text fields diverge

`IndexEntry` (`models/elements.py`) carries `embed_text`, `keyword_text` and
`display_text` because a single string cannot serve dense search, keyword search and
display equally well for every chunk kind. `utils/index_entry_builder.py` is where the
split is actually decided per chunk:

- **Text and figure chunks:** all three fields are the same string -- the chunk's own
  text (a figure chunk's text already being caption plus an LLM-written description, so
  there's nothing more useful to embed separately).
- **Table chunks:** `keyword_text` and `display_text` are the full Markdown table,
  every cell, verbatim; `embed_text` is `chunk.summary` if `MetadataEnhancer` produced
  one (`ingest --enhance-metadata`), otherwise the first 500 characters of the Markdown
  as a fallback.

This split is the entire reason hybrid search exists for tables specifically. A query
like "who bears the cost of heat pumps" can only ever be answered by the *dense* branch,
via a semantic summary; a raw table cell like `-0.59%` has no semantic neighbourhood; a
sentence-shaped embedding of it would drift toward whatever the LLM's summary happened
to say around it. Conversely, a query for the literal figure `-0.59` can only ever be
answered by the *keyword* branch matching that exact lexeme in `keyword_text`'s
Markdown -- an embedding of a one-sentence summary may not mention the number at all.
Collapsing the three fields into one loses either recall (the keyword match) or
precision (the semantic match), which is why they're kept apart per chunk rather than
merged.

## 10. Worked example: one question, end to end

Asking `python main.py ask "who bears the cost of heat pumps"`:

1. `main.py`'s `ask()` builds one `Embedder` and one `PostgresSearchIndex(embedder=...)`,
   and a `QAAgent` over it.
2. `QAAgent.answer()` sends the question plus both tool definitions to Claude. Claude
   rewrites it (section 1) and calls `search_knowledge_base` with something like
   `"heat pump cost labour skills wages"`.
3. `PostgresSearchIndex.search()` embeds that string once, runs the RRF query
   (section 3) with a `candidate_limit` of 50, and gets back up to `top_k` fused rows;
   `_attach_context` (section 4) adds one neighbour chunk each side of every match.
4. `QAAgent` dedups against `_seen_chunk_ids` (section 2; nothing seen yet on the first
   call), then `_format_results` renders each hit as
   `[1] Source: file.pdf, page 4` with `...context`/`>>> MATCH:`/`...context` lines, plus
   a profile preamble if any matched file has a `DocumentSummary`.
5. That text goes back to Claude as the tool result. If it's decisive, Claude answers,
   citing `(file.pdf, p. 4)` against the match, never the context. If it looks
   incomplete (a table with no header, say), Claude calls `read_source` on that file
   instead of searching again, and section 5's exact lookup returns the whole page or
   document.
6. If a call to `search_knowledge_base` ever comes back with zero rows, `_no_results_message`
   additionally tries `search_documents` (section 6) so Claude has a document to point
   `read_source` at instead of a dead end.
7. The loop repeats until Claude stops calling tools (final answer), refuses, or the
   8-iteration budget runs out.
