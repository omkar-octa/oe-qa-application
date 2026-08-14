import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import _drop_duplicate_recovered_text, ingest
from models.documents import DocumentChunk


def make_chunk(file_name: str, page: int) -> DocumentChunk:
    return DocumentChunk(chunk_id=f"{file_name}#p{page}#c0", file_name=file_name, page_number=page, text="Body text.")


@pytest.mark.unit
def test_ingest_falls_back_to_docling_for_pages_needing_ocr(tmp_path):
    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "mixed.pdf").write_bytes(b"%PDF-1.4 fake")
    output = tmp_path / "index.json"

    fast = MagicMock()
    fast.extract_chunks.return_value = [make_chunk("mixed.pdf", page=8)]
    fast.pages_needing_ocr.return_value = [1, 2, 3]
    slow = MagicMock()
    slow.extract_chunks_for_pages.return_value = [make_chunk("mixed.pdf", page=1)]

    with (
        patch("utils.pdf_extractor.PdfExtractor", return_value=fast),
        patch("utils.document_extractor.DocumentExtractor", return_value=slow),
    ):
        result = ingest(source, output, caption_figures=False)

    assert result == 0
    slow.extract_chunks_for_pages.assert_called_once_with(source / "mixed.pdf", [1, 2, 3])
    written_pages = {chunk["page_number"] for chunk in json.loads(output.read_text())["chunks"]}
    assert written_pages == {1, 8}


@pytest.mark.unit
def test_ingest_uses_fast_path_without_building_docling(tmp_path):
    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "digital.pdf").write_bytes(b"%PDF-1.4 fake")
    output = tmp_path / "index.json"

    fast = MagicMock()
    fast.extract_chunks.return_value = [make_chunk("digital.pdf", page=1)]
    fast.pages_needing_ocr.return_value = []

    with (
        patch("utils.pdf_extractor.PdfExtractor", return_value=fast),
        patch("utils.document_extractor.DocumentExtractor") as slow_cls,
    ):
        result = ingest(source, output, caption_figures=False)

    assert result == 0
    slow_cls.assert_not_called()


@pytest.mark.unit
def test_ingest_accepts_a_single_pdf_file_as_source(tmp_path):
    pdf_path = tmp_path / "solo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    output = tmp_path / "index.json"

    fast = MagicMock()
    fast.extract_chunks.return_value = [make_chunk("solo.pdf", page=1)]
    fast.pages_needing_ocr.return_value = []

    with patch("utils.pdf_extractor.PdfExtractor", return_value=fast):
        result = ingest(pdf_path, output, caption_figures=False)

    assert result == 0
    fast.extract_chunks.assert_called_once_with(pdf_path)
    written_files = {chunk["file_name"] for chunk in json.loads(output.read_text())["chunks"]}
    assert written_files == {"solo.pdf"}


@pytest.mark.unit
def test_ingest_captions_only_the_likely_figure_pages(tmp_path):
    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    output = tmp_path / "index.json"

    fast = MagicMock()
    fast.extract_chunks.return_value = [make_chunk("paper.pdf", page=1)]
    fast.pages_needing_ocr.return_value = []

    captioner = MagicMock()
    captioner.caption_document.return_value = []
    captioner.failed_pages = []
    captioner._renderer.page_count.return_value = 5

    with (
        patch("utils.pdf_extractor.PdfExtractor", return_value=fast),
        patch("utils.figure_captioner.FigureCaptioner", return_value=captioner),
        patch("utils.figure_captioner.likely_figure_pages", return_value=[3]),
        patch("utils.document_extractor.extract_figure_images", return_value={3: []}) as crop_images,
    ):
        result = ingest(source, output, caption_figures=True)

    assert result == 0
    captioner.caption_document.assert_called_once_with(source / "paper.pdf", pages=[3])
    # Cropping runs on the same candidate pages captioning does, not every page.
    crop_images.assert_called_once_with(source / "paper.pdf", [3], Path("data/pages"))


@pytest.mark.unit
def test_ingest_keeps_captioning_if_crop_extraction_fails(tmp_path):
    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    output = tmp_path / "index.json"

    fast = MagicMock()
    fast.extract_chunks.return_value = [make_chunk("paper.pdf", page=1)]
    fast.pages_needing_ocr.return_value = []

    captioner = MagicMock()
    captioner.caption_document.return_value = []
    captioner.failed_pages = []
    captioner._renderer.page_count.return_value = 5

    with (
        patch("utils.pdf_extractor.PdfExtractor", return_value=fast),
        patch("utils.figure_captioner.FigureCaptioner", return_value=captioner),
        patch("utils.figure_captioner.likely_figure_pages", return_value=[3]),
        patch("utils.document_extractor.extract_figure_images", side_effect=RuntimeError("boom")),
    ):
        result = ingest(source, output, caption_figures=True)

    assert result == 0
    captioner.caption_document.assert_called_once_with(source / "paper.pdf", pages=[3])


@pytest.mark.unit
def test_ingest_skips_captioning_a_file_with_no_candidate_pages(tmp_path):
    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    output = tmp_path / "index.json"

    fast = MagicMock()
    fast.extract_chunks.return_value = [make_chunk("paper.pdf", page=1)]
    fast.pages_needing_ocr.return_value = []

    captioner = MagicMock()
    captioner._renderer.page_count.return_value = 5

    with (
        patch("utils.pdf_extractor.PdfExtractor", return_value=fast),
        patch("utils.figure_captioner.FigureCaptioner", return_value=captioner),
        patch("utils.figure_captioner.likely_figure_pages", return_value=[]),
    ):
        result = ingest(source, output, caption_figures=True)

    assert result == 0
    captioner.caption_document.assert_not_called()


@pytest.mark.unit
def test_ingest_captions_every_page_if_candidate_detection_fails(tmp_path):
    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    output = tmp_path / "index.json"

    fast = MagicMock()
    fast.extract_chunks.return_value = [make_chunk("paper.pdf", page=1)]
    fast.pages_needing_ocr.return_value = []

    captioner = MagicMock()
    captioner.caption_document.return_value = []
    captioner.failed_pages = []
    captioner._renderer.page_count.return_value = 5

    with (
        patch("utils.pdf_extractor.PdfExtractor", return_value=fast),
        patch("utils.figure_captioner.FigureCaptioner", return_value=captioner),
        patch("utils.figure_captioner.likely_figure_pages", side_effect=RuntimeError("boom")),
    ):
        result = ingest(source, output, caption_figures=True)

    assert result == 0
    captioner.caption_document.assert_called_once_with(source / "paper.pdf", pages=None)


@pytest.mark.unit
def test_ingest_skips_a_file_that_fails_extraction(tmp_path):
    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "broken.pdf").write_bytes(b"%PDF-1.4 fake")
    (source / "good.pdf").write_bytes(b"%PDF-1.4 fake")
    output = tmp_path / "index.json"

    fast = MagicMock()
    fast.extract_chunks.side_effect = [RuntimeError("corrupt"), [make_chunk("good.pdf", page=1)]]
    fast.pages_needing_ocr.return_value = []

    with patch("utils.pdf_extractor.PdfExtractor", return_value=fast):
        result = ingest(source, output, caption_figures=False)

    assert result == 0
    assert output.exists()


def make_text_chunk(file_name: str, page: int, text: str, kind: str = "text") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{file_name}#p{page}#c0", file_name=file_name, page_number=page, text=text, kind=kind
    )


@pytest.mark.unit
def test_recovered_text_is_dropped_when_the_page_already_has_ample_text():
    existing = [make_text_chunk("paper.pdf", page=1, text="x" * 600)]
    recovered = [make_text_chunk("paper.pdf", page=1, text="Recovered paragraph.")]

    kept, dropped = _drop_duplicate_recovered_text(recovered, existing)

    assert kept == []
    assert dropped == 1


@pytest.mark.unit
def test_recovered_text_is_kept_when_the_page_is_thin():
    # Mirrors the real Burlinson page 1 case: only 114 characters of existing
    # text (a JEL/keywords box), so the recovered paragraph is genuinely new.
    existing = [make_text_chunk("paper.pdf", page=1, text="x" * 114)]
    recovered = [make_text_chunk("paper.pdf", page=1, text="Recovered paragraph.")]

    kept, dropped = _drop_duplicate_recovered_text(recovered, existing)

    assert kept == recovered
    assert dropped == 0


@pytest.mark.unit
def test_recovered_figure_chunks_are_never_dropped_by_the_text_dedup_guard():
    existing = [make_text_chunk("paper.pdf", page=1, text="x" * 600)]
    recovered = [make_text_chunk("paper.pdf", page=1, text="A chart description.", kind="figure")]

    kept, dropped = _drop_duplicate_recovered_text(recovered, existing)

    assert kept == recovered
    assert dropped == 0


@pytest.mark.unit
def test_caption_figures_defaults_to_on_in_the_cli():
    import main

    with (
        patch("main.ingest", return_value=0) as ingest_mock,
        patch("sys.argv", ["main.py", "ingest"]),
    ):
        with pytest.raises(SystemExit):
            main.main()

    assert ingest_mock.call_args.args[5] is True


@pytest.mark.unit
def test_no_caption_figures_flag_turns_it_off():
    import main

    with (
        patch("main.ingest", return_value=0) as ingest_mock,
        patch("sys.argv", ["main.py", "ingest", "--no-caption-figures"]),
    ):
        with pytest.raises(SystemExit):
            main.main()

    assert ingest_mock.call_args.args[5] is False


@pytest.mark.unit
def test_index_embeds_and_upserts_each_file(tmp_path):
    from main import index
    from models.documents import SearchIndex

    index_path = tmp_path / "index.json"
    index_path.write_text(SearchIndex(chunks=[make_chunk("doc.pdf", page=1)]).model_dump_json(), encoding="utf-8")

    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "doc.pdf").write_bytes(b"%PDF-1.4 fake")

    built_entry = MagicMock(chunk_id="d-p001-c00", doc_id="d", file_type="pdf", embed_text="hi", embedding=None)
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1, 0.2]]
    conn = MagicMock()

    with (
        patch("utils.index_entry_builder.build_index_entries", return_value=[built_entry]) as build_entries,
        patch("utils.embedder.Embedder", return_value=embedder),
        patch("utils.postgres_store.connect", return_value=conn),
        patch("utils.postgres_store.ensure_schema") as ensure_schema,
        patch("utils.postgres_store.upsert_document") as upsert_document,
    ):
        result = index(index_path, source)

    assert result == 0
    build_entries.assert_called_once()
    ensure_schema.assert_called_once_with(conn)
    embedder.embed_texts.assert_called_once_with(["hi"])
    assert built_entry.embedding == [0.1, 0.2]
    upsert_document.assert_called_once()
    # Check the positional args this test owns; extra kwargs (e.g. document-level
    # summary fields) belong to a different feature and aren't this test's concern.
    assert upsert_document.call_args.args == (conn, "d", "doc.pdf", "pdf", [built_entry])


@pytest.mark.unit
def test_index_skips_a_file_not_found_under_source(tmp_path):
    from main import index
    from models.documents import SearchIndex

    index_path = tmp_path / "index.json"
    index_path.write_text(
        SearchIndex(chunks=[make_chunk("missing.pdf", page=1)]).model_dump_json(), encoding="utf-8"
    )
    source = tmp_path / "pdfs"
    source.mkdir()

    with (
        patch("utils.embedder.Embedder"),
        patch("utils.postgres_store.connect"),
        patch("utils.postgres_store.ensure_schema"),
        patch("utils.postgres_store.upsert_document") as upsert_document,
    ):
        result = index(index_path, source)

    assert result == 0
    upsert_document.assert_not_called()


@pytest.mark.unit
def test_ask_builds_a_postgres_search_index():
    from main import ask

    search_backend = MagicMock()
    embedder = MagicMock()

    with (
        patch("utils.embedder.Embedder", return_value=embedder),
        patch("utils.postgres_search_index.PostgresSearchIndex", return_value=search_backend) as pg_index_cls,
        patch("utils.qa_agent.QAAgent") as qa_agent_cls,
    ):
        qa_agent_cls.return_value.answer.return_value = "answer text"
        result = ask("question?", top_k=5)

    assert result == 0
    # attach_context not passed to ask(), so it forwards None -- PostgresSearchIndex
    # falls back to settings.attach_search_context (off by default) itself.
    pg_index_cls.assert_called_once_with(embedder=embedder, attach_context=None)
    qa_agent_cls.assert_called_once_with(search_backend)
    qa_agent_cls.return_value.answer.assert_called_once_with("question?", top_k=5)


@pytest.mark.unit
def test_ask_forwards_attach_context_when_given():
    from main import ask

    with (
        patch("utils.embedder.Embedder", return_value=MagicMock()),
        patch("utils.postgres_search_index.PostgresSearchIndex", return_value=MagicMock()) as pg_index_cls,
        patch("utils.qa_agent.QAAgent") as qa_agent_cls,
    ):
        qa_agent_cls.return_value.answer.return_value = "answer text"
        ask("question?", top_k=5, attach_context=True)

    assert pg_index_cls.call_args.kwargs["attach_context"] is True


@pytest.mark.unit
def test_attach_context_defaults_to_the_setting_in_the_cli():
    import main

    with (
        patch("main.ask", return_value=0) as ask_mock,
        patch("sys.argv", ["main.py", "ask", "a question"]),
    ):
        with pytest.raises(SystemExit):
            main.main()

    # settings.attach_search_context is False by default; nothing on the
    # command line should turn it on.
    assert ask_mock.call_args.args[3] is False


@pytest.mark.unit
def test_attach_context_flag_forces_it_on_for_one_run():
    import main

    with (
        patch("main.ask", return_value=0) as ask_mock,
        patch("sys.argv", ["main.py", "ask", "a question", "--attach-context"]),
    ):
        with pytest.raises(SystemExit):
            main.main()

    assert ask_mock.call_args.args[3] is True
