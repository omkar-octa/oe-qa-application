"""Hybrid (keyword + vector) search over the Postgres-backed IndexEntry store.

Satisfies the retriever protocol QAAgent expects: search(query, top_k) ->
list[SearchResult], plus the optional document()/page()/files() accessors
that unlock the read_source escalation tool. QAAgent consumes
SearchResult.chunk as a DocumentChunk, so rows fetched from index_entries
are reconstructed into DocumentChunks here rather than exposed as
IndexEntry objects. search() can also attach each hit's immediate
page-adjacent neighbour as context_before/context_after, after ranking,
so QAAgent._format_results has a completed sentence or table header to
show instead of just the matched chunk in isolation -- gated behind
attach_context/settings.attach_search_context, off by default, since it
costs extra tokens on every match whether or not that match needed it.
"""

from __future__ import annotations

import logging

from pgvector import Vector
from psycopg.rows import dict_row

from models.config import settings
from models.documents import DocumentChunk, DocumentSummary, SearchResult
from utils.postgres_store import connect

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion: keyword and vector ranking use incompatible score
# scales (ts_rank_cd vs. cosine distance), so they are fused by rank rather
# than by normalising one score onto the other's scale. 60 is the constant
# from the original RRF paper; it damps the influence of any single rank
# without needing to be tuned per query.
_RRF_K = 60

_RRF_SEARCH_SQL = f"""
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
    SELECT chunk_id, SUM(1.0 / ({_RRF_K} + rank)) AS score
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
"""

# The document lane. A separate query against a separate table, so a profile
# can never take a slot from a chunk in the ranked results above; vector only,
# because a profile is prose written to be semantically matched and has none of
# the exact figures that make keyword search worth fusing in.
_DOCUMENT_SEARCH_SQL = """
SELECT file_name, doc_summary, page_count, 1 - (summary_embedding <=> %(query_vector)s) AS score
FROM documents
WHERE summary_embedding IS NOT NULL AND doc_summary IS NOT NULL
ORDER BY summary_embedding <=> %(query_vector)s
LIMIT %(top_k)s
"""

# read_source's escalation lane: exact file_name lookups, no ranking. Ordered
# by page then chunk_id so a multi-chunk page comes back in reading order the
# same way document order does for a whole file.
_DOCUMENT_CHUNKS_SQL = """
SELECT chunk_id, file_name, page_start, display_text, heading_path, element_types
FROM index_entries
WHERE file_name = %(file_name)s
ORDER BY page_start, chunk_id
"""

_PAGE_CHUNKS_SQL = """
SELECT chunk_id, file_name, page_start, display_text, heading_path, element_types
FROM index_entries
WHERE file_name = %(file_name)s AND page_start = %(page_number)s
ORDER BY chunk_id
"""

# documents, not index_entries: one row per file regardless of how many chunks
# it produced, and the table read_source's "indexed files are" listing means.
_FILE_NAMES_SQL = "SELECT file_name FROM documents ORDER BY file_name"

# Neighbour lookup for search()'s context expansion. Windowed over each
# matched chunk's own file rather than globally, so a match at a file
# boundary never picks up another document's chunk as its "neighbour" the
# way raw list adjacency could. Applied after ranking, same as the removed
# JSON backend: neighbours are attached to already-scored hits, never scored
# themselves, so they can't displace a genuine top_k match.
_NEIGHBOUR_IDS_SQL = """
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
"""

_CHUNKS_BY_ID_SQL = """
SELECT chunk_id, file_name, page_start, display_text, heading_path, element_types
FROM index_entries
WHERE chunk_id = ANY(%(chunk_ids)s)
"""

# Candidates considered per branch before fusion; generous relative to
# top_k so RRF has enough of the ranking to fuse over.
_MIN_CANDIDATE_LIMIT = 50
_CANDIDATE_LIMIT_MULTIPLIER = 10


class PostgresSearchIndex:
    """The sole search backend: hybrid keyword + vector retrieval over Postgres.

    `embedder` is duck-typed: anything with embed_texts(list[str]) ->
    list[list[float]] works (Embedder or EmbeddingClient both qualify), so
    this module doesn't import either concretely.

    `attach_context` gates whether search() fetches and attaches a
    neighbouring chunk each side of every match (see _attach_context).
    Defaults to settings.attach_search_context (itself False by default) when
    not given explicitly, so every caller that doesn't ask for it -- main.py's
    `ask`, api.py -- gets the cheaper behaviour unless the setting or the
    argument says otherwise. Off by default because attaching context is
    unconditional per match whenever it runs: real token cost with no check
    for whether the match already stood on its own (docs/search.md).
    """

    def __init__(self, embedder, conn=None, attach_context: bool | None = None):
        self._embedder = embedder
        self._conn = conn if conn is not None else connect()
        self._attach_context_enabled = (
            settings.attach_search_context if attach_context is None else attach_context
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        query_vector = self._embedder.embed_texts([query])[0]
        candidate_limit = max(top_k * _CANDIDATE_LIMIT_MULTIPLIER, _MIN_CANDIDATE_LIMIT)
        logger.debug(
            "RRF search: query=%r top_k=%d candidate_limit=%d", query, top_k, candidate_limit
        )

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                _RRF_SEARCH_SQL,
                {
                    "query_text": query,
                    "query_vector": Vector(query_vector),
                    "candidate_limit": candidate_limit,
                    "top_k": top_k,
                },
            )
            rows = cur.fetchall()

        logger.debug("RRF search: %d row(s) returned for query=%r", len(rows), query)
        if not rows:
            return []

        results = [self._row_to_search_result(row) for row in rows]
        if self._attach_context_enabled:
            self._attach_context(results)
        return results

    def search_documents(self, query: str, top_k: int = 3) -> list[DocumentSummary]:
        """Which documents are about this, rather than which extracts match it.

        Ranked over `documents.summary_embedding`, its own HNSW index on its
        own table, so this never competes with `search` and never returns
        anything citable. What comes back is routing information: a file name
        to point read_source at when chunk search has come up empty.

        Returns [] when nothing has been summarised, which is the normal state
        unless `ingest --summarise-documents` ran.
        """
        if not query or not query.strip():
            return []

        query_vector = self._embedder.embed_texts([query])[0]
        logger.debug("document search: query=%r top_k=%d", query, top_k)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                _DOCUMENT_SEARCH_SQL,
                {"query_vector": Vector(query_vector), "top_k": top_k},
            )
            rows = cur.fetchall()

        logger.debug("document search: %d row(s) returned for query=%r", len(rows), query)
        return [
            DocumentSummary(
                file_name=row["file_name"],
                summary=row["doc_summary"],
                page_count=row["page_count"] or 0,
            )
            for row in rows
        ]

    def document(self, file_name: str) -> list[DocumentChunk]:
        """Every chunk of one source document, in page order."""
        return self._chunks_for(_DOCUMENT_CHUNKS_SQL, {"file_name": file_name})

    def page(self, file_name: str, page_number: int) -> list[DocumentChunk]:
        """Every chunk of one page, in reading order."""
        return self._chunks_for(
            _PAGE_CHUNKS_SQL, {"file_name": file_name, "page_number": page_number}
        )

    def files(self) -> list[str]:
        """Every indexed file name, for read_source's miss message."""
        with self._conn.cursor() as cur:
            cur.execute(_FILE_NAMES_SQL)
            return [row[0] for row in cur.fetchall()]

    def _chunks_for(self, sql: str, params: dict) -> list[DocumentChunk]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_chunk(row: dict) -> DocumentChunk:
        element_types = row.get("element_types") or []
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            file_name=row["file_name"],
            page_number=row["page_start"],
            text=row["display_text"],
            headings=row["heading_path"] or [],
            kind="table" if "table" in element_types else "text",
        )

    @classmethod
    def _row_to_search_result(cls, row: dict) -> SearchResult:
        return SearchResult(
            chunk=cls._row_to_chunk(row),
            score=row["fused_score"],
            doc_summary=row.get("doc_summary"),
        )

    def _attach_context(self, results: list[SearchResult]) -> None:
        """Fills in context_before/context_after with each hit's immediate
        page-adjacent neighbour, one chunk each side, mirroring what the
        removed LexicalSearchIndex used to attach. Runs after ranking (the
        results are already scored `SearchResult`s), so a neighbour can never
        influence score or displace a genuine top_k match, and only two
        queries are needed regardless of how many results came back: one to
        resolve neighbour chunk_ids via a per-file window function, one to
        fetch the text for whichever of those ids actually exist.
        """
        file_names = list({result.chunk.file_name for result in results})
        chunk_ids = [result.chunk.chunk_id for result in results]

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_NEIGHBOUR_IDS_SQL, {"file_names": file_names, "chunk_ids": chunk_ids})
            neighbour_ids = {
                row["chunk_id"]: (row.get("prev_chunk_id"), row.get("next_chunk_id"))
                for row in cur.fetchall()
            }

        wanted = {cid for pair in neighbour_ids.values() for cid in pair if cid}
        if not wanted:
            return

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_CHUNKS_BY_ID_SQL, {"chunk_ids": list(wanted)})
            chunks_by_id = {row["chunk_id"]: self._row_to_chunk(row) for row in cur.fetchall()}

        for result in results:
            prev_id, next_id = neighbour_ids.get(result.chunk.chunk_id, (None, None))
            if prev_id in chunks_by_id:
                result.context_before = [chunks_by_id[prev_id]]
            if next_id in chunks_by_id:
                result.context_after = [chunks_by_id[next_id]]
