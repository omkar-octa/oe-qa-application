from unittest.mock import MagicMock

import pytest

from utils.postgres_search_index import PostgresSearchIndex


def make_row(**overrides) -> dict:
    row = {
        "chunk_id": "doc1-p003-c00",
        "doc_id": "doc1",
        "file_name": "paper.pdf",
        "file_type": "pdf",
        "granularity": "element",
        "page_start": 3,
        "page_end": 3,
        "element_ids": ["doc1-e00001"],
        "element_types": ["table"],
        "heading_path": ["Results"],
        "embed_text": "summary",
        "keyword_text": "-0.59% GDP",
        "display_text": "| GDP | -0.59% |",
        "asset_paths": [],
        "n_chars": 20,
        "embedding_model": "text-embedding-3-large",
        "embedding": None,
        "fused_score": 0.032,
    }
    row.update(overrides)
    return row


def make_embedder(vector=None) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_texts.return_value = [vector or [0.1, 0.2, 0.3]]
    return embedder


def make_conn(*fetchall_results: list[dict]):
    """Each positional arg is one call's worth of cur.fetchall() rows, in
    order. search() issues exactly one (the RRF search) by default, or up to
    three (RRF search, neighbour ids, neighbour chunks) when constructed with
    attach_context=True and the RRF query returns at least one row; document(),
    page() and files() issue exactly one. side_effect on a list is consumed
    once per call and raises StopIteration once exhausted, so pass one arg per
    call actually expected."""
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.side_effect = list(fetchall_results)
    return conn, cursor


@pytest.mark.unit
def test_search_calls_embedder_with_query_text():
    embedder = make_embedder()
    conn, _ = make_conn([make_row()])
    index = PostgresSearchIndex(embedder, conn=conn)

    index.search("gdp contraction", top_k=3)

    embedder.embed_texts.assert_called_once_with(["gdp contraction"])


@pytest.mark.unit
def test_search_query_contains_keyword_and_vector_ranking():
    embedder = make_embedder()
    conn, cursor = make_conn([make_row()])
    index = PostgresSearchIndex(embedder, conn=conn)

    index.search("gdp contraction")

    # The only query issued by default: attach_context is off, see the
    # dedicated context-expansion tests below for that behaviour.
    executed_sql = cursor.execute.call_args_list[0].args[0]
    assert "ts_rank_cd" in executed_sql
    assert "plainto_tsquery" in executed_sql
    assert "<=>" in executed_sql


@pytest.mark.unit
def test_search_reconstructs_document_chunk_and_search_result():
    embedder = make_embedder()
    conn, _ = make_conn([make_row()])
    index = PostgresSearchIndex(embedder, conn=conn)

    results = index.search("gdp contraction")

    assert len(results) == 1
    result = results[0]
    assert result.score == pytest.approx(0.032)
    chunk = result.chunk
    assert chunk.chunk_id == "doc1-p003-c00"
    assert chunk.file_name == "paper.pdf"
    assert chunk.page_number == 3
    assert chunk.text == "| GDP | -0.59% |"
    assert chunk.headings == ["Results"]
    assert chunk.kind == "table"
    assert chunk.embedding is None
    assert chunk.summary is None


@pytest.mark.unit
def test_text_row_reconstructs_as_text_kind():
    embedder = make_embedder()
    conn, _ = make_conn([make_row(element_types=["text"])])
    index = PostgresSearchIndex(embedder, conn=conn)

    result = index.search("anything")[0]

    assert result.chunk.kind == "text"


@pytest.mark.unit
def test_empty_query_returns_empty():
    embedder = make_embedder()
    conn, _ = make_conn([make_row()])
    index = PostgresSearchIndex(embedder, conn=conn)

    assert index.search("") == []
    embedder.embed_texts.assert_not_called()


@pytest.mark.unit
def test_no_match_returns_empty():
    embedder = make_embedder()
    conn, _ = make_conn([])
    index = PostgresSearchIndex(embedder, conn=conn)

    assert index.search("zebra quantum banjo") == []


@pytest.mark.unit
def test_search_passes_top_k_and_candidate_limit():
    embedder = make_embedder()
    conn, cursor = make_conn([])
    index = PostgresSearchIndex(embedder, conn=conn)

    index.search("gdp", top_k=2)

    params = cursor.execute.call_args.args[1]
    assert params["top_k"] == 2
    assert params["candidate_limit"] == 50  # max(2 * 10, 50)


# =========================================================================
# search()'s context expansion -- context_before/context_after
#
# Off by default (settings.attach_search_context, threaded through
# PostgresSearchIndex's attach_context constructor argument): real,
# unconditional token cost on every match, so it only runs when explicitly
# asked for. See docs/search.md, "This is unconditional, and it costs tokens
# even when the match alone was enough."
# =========================================================================


@pytest.mark.unit
def test_context_not_attached_by_default():
    embedder = make_embedder()
    conn, cursor = make_conn([make_row()])
    index = PostgresSearchIndex(embedder, conn=conn)

    result = index.search("gdp contraction")[0]

    assert result.context_before == []
    assert result.context_after == []
    # Only the RRF query itself; neighbour resolution never runs.
    assert cursor.execute.call_count == 1


@pytest.mark.unit
def test_search_attaches_neighbours_on_both_sides_when_enabled():
    embedder = make_embedder()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.side_effect = [
        [make_row(chunk_id="doc1-p003-c01")],  # RRF match
        [
            {
                "chunk_id": "doc1-p003-c01",
                "prev_chunk_id": "doc1-p003-c00",
                "next_chunk_id": "doc1-p003-c02",
            }
        ],
        [
            make_chunk_row(chunk_id="doc1-p003-c00", display_text="Before text."),
            make_chunk_row(chunk_id="doc1-p003-c02", display_text="After text."),
        ],
    ]
    index = PostgresSearchIndex(embedder, conn=conn, attach_context=True)

    result = index.search("gdp contraction")[0]

    assert [c.text for c in result.context_before] == ["Before text."]
    assert [c.text for c in result.context_after] == ["After text."]


@pytest.mark.unit
def test_search_leaves_context_empty_at_a_file_boundary_when_enabled():
    """The first or last chunk of a file has no neighbour on that side; the
    window function returns NULL rather than another file's chunk."""
    embedder = make_embedder()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.side_effect = [
        [make_row()],
        [{"chunk_id": "doc1-p003-c00", "prev_chunk_id": None, "next_chunk_id": None}],
    ]
    index = PostgresSearchIndex(embedder, conn=conn, attach_context=True)

    result = index.search("gdp contraction")[0]

    assert result.context_before == []
    assert result.context_after == []
    # No neighbour ids to resolve, so the second lookup is skipped entirely.
    assert cursor.execute.call_count == 2


@pytest.mark.unit
def test_neighbour_lookup_is_windowed_per_file_when_enabled():
    embedder = make_embedder()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.side_effect = [
        [make_row(chunk_id="doc1-p003-c00", file_name="paper.pdf")],
        [{"chunk_id": "doc1-p003-c00", "prev_chunk_id": None, "next_chunk_id": None}],
    ]
    index = PostgresSearchIndex(embedder, conn=conn, attach_context=True)

    index.search("gdp contraction")

    neighbour_sql, neighbour_params = cursor.execute.call_args_list[1].args
    assert "PARTITION BY file_name" in neighbour_sql
    assert neighbour_params["file_names"] == ["paper.pdf"]
    assert neighbour_params["chunk_ids"] == ["doc1-p003-c00"]


@pytest.mark.unit
def test_attach_context_true_is_independent_of_the_setting_default():
    """Explicit attach_context passed at construction always wins over
    settings.attach_search_context, whichever way that default is set."""
    from models.config import settings

    assert settings.attach_search_context is False  # the documented default
    embedder = make_embedder()
    conn, cursor = make_conn([make_row()], [{"chunk_id": "doc1-p003-c00", "prev_chunk_id": None, "next_chunk_id": None}])
    index = PostgresSearchIndex(embedder, conn=conn, attach_context=True)

    index.search("gdp contraction")

    assert cursor.execute.call_count == 2


# =========================================================================
# document() / page() / files() -- the read_source accessors
# =========================================================================


def make_chunk_row(**overrides) -> dict:
    row = {
        "chunk_id": "doc1-p001-c00",
        "file_name": "paper.pdf",
        "page_start": 1,
        "display_text": "Some extracted text.",
        "heading_path": ["Introduction"],
        "element_types": ["text"],
    }
    row.update(overrides)
    return row


@pytest.mark.unit
def test_document_returns_chunks_ordered_by_page_then_chunk_id():
    conn, cursor = make_conn(
        [
            make_chunk_row(chunk_id="doc1-p001-c00", page_start=1, display_text="First."),
            make_chunk_row(chunk_id="doc1-p002-c00", page_start=2, display_text="Second."),
        ]
    )
    index = PostgresSearchIndex(make_embedder(), conn=conn)

    chunks = index.document("paper.pdf")

    executed_sql, params = cursor.execute.call_args.args
    assert "WHERE file_name = %(file_name)s" in executed_sql
    assert "ORDER BY page_start, chunk_id" in executed_sql
    assert params == {"file_name": "paper.pdf"}
    assert [c.text for c in chunks] == ["First.", "Second."]
    assert [c.page_number for c in chunks] == [1, 2]


@pytest.mark.unit
def test_document_returns_no_chunks_for_an_unknown_file():
    conn, _ = make_conn([])
    index = PostgresSearchIndex(make_embedder(), conn=conn)

    assert index.document("missing.pdf") == []


@pytest.mark.unit
def test_page_filters_by_file_name_and_page_number():
    conn, cursor = make_conn([make_chunk_row(page_start=3, display_text="Page three.")])
    index = PostgresSearchIndex(make_embedder(), conn=conn)

    chunks = index.page("paper.pdf", 3)

    executed_sql, params = cursor.execute.call_args.args
    assert "page_start = %(page_number)s" in executed_sql
    assert params == {"file_name": "paper.pdf", "page_number": 3}
    assert chunks[0].text == "Page three."


@pytest.mark.unit
def test_document_and_page_reconstruct_table_kind_from_element_types():
    conn, _ = make_conn([make_chunk_row(element_types=["table"])])
    index = PostgresSearchIndex(make_embedder(), conn=conn)

    assert index.document("paper.pdf")[0].kind == "table"


@pytest.mark.unit
def test_files_lists_distinct_indexed_file_names():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [("a.pdf",), ("b.pdf",)]
    index = PostgresSearchIndex(make_embedder(), conn=conn)

    files = index.files()

    executed_sql = cursor.execute.call_args.args[0]
    assert "FROM documents" in executed_sql
    assert files == ["a.pdf", "b.pdf"]
