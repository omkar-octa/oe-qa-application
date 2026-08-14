import pytest

from utils.page_renderer import MAX_LONG_EDGE_PX, MIN_LONG_EDGE_PX, PageRenderer, RenderedPage


@pytest.mark.unit
def test_long_edge_is_clamped_to_the_readable_range(tmp_path):
    assert PageRenderer(tmp_path, long_edge_px=99999).long_edge_px == MAX_LONG_EDGE_PX
    assert PageRenderer(tmp_path, long_edge_px=10).long_edge_px == MIN_LONG_EDGE_PX
    assert PageRenderer(tmp_path, long_edge_px=2000).long_edge_px == 2000


@pytest.mark.unit
def test_rendered_page_reads_its_own_bytes(tmp_path):
    image_path = tmp_path / "p0001.png"
    image_path.write_bytes(b"\x89PNG fake")

    page = RenderedPage(page_number=1, image_path=image_path)

    assert page.read_bytes() == b"\x89PNG fake"
    assert page.media_type == "image/png"


@pytest.mark.unit
def test_render_reuses_existing_images(tmp_path, monkeypatch):
    """Rasterising is the slow, deterministic part of a re-ingest, so an
    already-rendered page must not be redrawn."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    page_dir = tmp_path / "out" / "paper"
    page_dir.mkdir(parents=True)
    (page_dir / "p0001.png").write_bytes(b"already rendered")

    renderer = PageRenderer(tmp_path / "out")
    monkeypatch.setattr(
        "utils.page_renderer.pdfium.PdfDocument", lambda _path: _FakeDocument(page_count=2)
    )
    redrawn: list[int] = []
    monkeypatch.setattr(
        PageRenderer,
        "_render_page",
        lambda self, doc, page_number, image_path: redrawn.append(page_number),
    )

    pages = renderer.render(pdf_path)

    assert [page.page_number for page in pages] == [1, 2]
    assert redrawn == [2]


@pytest.mark.unit
def test_render_honours_an_explicit_page_list(tmp_path, monkeypatch):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    renderer = PageRenderer(tmp_path / "out")
    monkeypatch.setattr(
        "utils.page_renderer.pdfium.PdfDocument", lambda _path: _FakeDocument(page_count=10)
    )
    monkeypatch.setattr(
        PageRenderer, "_render_page", lambda self, doc, page_number, image_path: None
    )

    pages = renderer.render(pdf_path, pages=[3, 7])

    assert [page.page_number for page in pages] == [3, 7]
    assert [page.image_path.name for page in pages] == ["p0003.png", "p0007.png"]


class _FakeDocument:
    def __init__(self, page_count: int):
        self._page_count = page_count

    def __len__(self) -> int:
        return self._page_count

    def close(self) -> None:
        pass
