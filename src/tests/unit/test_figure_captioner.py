import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from utils.figure_captioner import PAGE_MAX_TOKENS, FigureCaptioner, likely_figure_pages
from utils.page_renderer import RenderedPage


class StubRenderer:
    """Stands in for PageRenderer so no PDF is opened and nothing is rasterised."""

    def __init__(self, pages: list[RenderedPage]):
        self._pages = pages

    def page_count(self, pdf_path) -> int:
        return len(self._pages)

    def render(self, pdf_path, pages: list[int] | None = None) -> list[RenderedPage]:
        if pages is None:
            return list(self._pages)
        return [page for page in self._pages if page.page_number in pages]


def make_pages(tmp_path: Path, count: int) -> list[RenderedPage]:
    rendered = []
    for page_number in range(1, count + 1):
        image_path = tmp_path / f"p{page_number:04d}.png"
        image_path.write_bytes(b"fake png bytes")
        rendered.append(RenderedPage(page_number=page_number, image_path=image_path))
    return rendered


def figure(caption: str, description: str, classification: str = "chart") -> dict:
    return {
        "caption": caption,
        "description": description,
        "classification": classification,
    }


def json_response(figures: list[dict], stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=json.dumps({"figures": figures}))],
    )


def make_captioner(tmp_path: Path, client, page_count: int = 1) -> FigureCaptioner:
    return FigureCaptioner(
        tmp_path,
        client=client,
        renderer=StubRenderer(make_pages(tmp_path, page_count)),
        # Serialise the pool so side_effect order maps to page order.
        max_workers=1,
    )


@pytest.mark.unit
def test_each_figure_becomes_its_own_chunk(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = json_response(
        [
            figure("Fig. 1. Adoption by region.", "Bar chart. UK 42%, France 31%."),
            figure("Fig. 2. Cost over time.", "Line chart, 2010 to 2020, GBP per unit."),
        ]
    )
    captioner = make_captioner(tmp_path, client)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert len(chunks) == 2
    assert all(chunk.kind == "figure" for chunk in chunks)
    assert all(chunk.file_name == "paper.pdf" for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == 2


@pytest.mark.unit
def test_chunk_text_carries_caption_and_description(tmp_path):
    # Retrieval reads chunk.text and chunk.headings and nothing else, so a
    # description that is not in text is invisible to search and to the model.
    client = MagicMock()
    client.messages.create.return_value = json_response(
        [figure("Fig. 5. Regional split.", "Choropleth map. Germany 12.4, France 9.8.")]
    )
    captioner = make_captioner(tmp_path, client)

    chunk = captioner.caption_document(Path("paper.pdf"))[0]

    assert "Fig. 5. Regional split." in chunk.text
    assert "Germany 12.4" in chunk.text
    assert chunk.headings == ["Fig. 5. Regional split."]


@pytest.mark.unit
def test_chunk_ids_do_not_collide_with_extractor_text_chunks(tmp_path):
    # Both extractors number their chunks file.pdf#p{page}#c{n} over the same
    # pages; figures use #f so a merged list has no duplicate ids.
    client = MagicMock()
    client.messages.create.return_value = json_response(
        [figure("Fig. 1.", "A chart."), figure("Fig. 2.", "Another chart.")]
    )
    captioner = make_captioner(tmp_path, client)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert [chunk.chunk_id for chunk in chunks] == [
        "paper.pdf#p1#f0",
        "paper.pdf#p1#f1",
    ]


@pytest.mark.unit
def test_logos_and_signatures_are_dropped(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = json_response(
        [
            figure("Elsevier", "Publisher logo.", classification="logo"),
            figure("Fig. 1. Results.", "Scatter plot of cost against uptake."),
            figure("J. Smith", "Handwritten signature.", classification="signature"),
        ]
    )
    captioner = make_captioner(tmp_path, client)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert len(chunks) == 1
    assert chunks[0].text.startswith("Fig. 1. Results.")


@pytest.mark.unit
def test_recovered_text_becomes_a_text_chunk_not_a_figure_chunk(tmp_path):
    # classification="text" marks body prose a layout model misclassified as
    # a picture; it must merge into search like ordinary text, not read like
    # a figure caption.
    client = MagicMock()
    client.messages.create.return_value = json_response(
        [
            figure(
                "1. Introduction",
                "The adoption of low-carbon technologies is central to net zero.",
                classification="text",
            ),
            figure("Fig. 1. Results.", "Scatter plot of cost against uptake."),
        ]
    )
    captioner = make_captioner(tmp_path, client)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert len(chunks) == 2
    recovered = next(chunk for chunk in chunks if chunk.kind == "text")
    figure_chunk = next(chunk for chunk in chunks if chunk.kind == "figure")
    assert recovered.text == "The adoption of low-carbon technologies is central to net zero."
    assert recovered.headings == ["1. Introduction"]
    assert figure_chunk.text.startswith("Fig. 1. Results.")


@pytest.mark.unit
def test_page_with_no_figures_yields_no_chunks_and_is_not_a_failure(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = json_response([])
    captioner = make_captioner(tmp_path, client)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert chunks == []
    assert captioner.failed_pages == []


@pytest.mark.unit
def test_refusal_degrades_that_page_rather_than_raising(tmp_path):
    # A refusal is HTTP 200 with empty content, so reading content first would
    # raise on a response that is not actually an error.
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(stop_reason="refusal", content=[])
    captioner = make_captioner(tmp_path, client)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert chunks == []
    assert captioner.failed_pages == []


@pytest.mark.unit
def test_truncated_json_is_reported_rather_than_silently_dropped(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        stop_reason="max_tokens",
        content=[SimpleNamespace(type="text", text='{"figures": [{"caption": "Fig')],
    )
    captioner = make_captioner(tmp_path, client)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert chunks == []
    assert captioner.failed_pages == [1]


@pytest.mark.unit
def test_one_failing_page_does_not_lose_the_others(tmp_path):
    client = MagicMock()
    client.messages.create.side_effect = [
        json_response([figure("Fig. 1.", "A chart on page one.")]),
        RuntimeError("transient API error"),
        json_response([figure("Fig. 2.", "A chart on page three.")]),
    ]
    captioner = make_captioner(tmp_path, client, page_count=3)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert [chunk.page_number for chunk in chunks] == [1, 3]
    assert captioner.failed_pages == [2]


@pytest.mark.unit
def test_chunks_come_back_in_page_order(tmp_path):
    # Chunk order in the written index is page order; a chunk out of place
    # here would read wrong wherever surrounding context is derived from it.
    client = MagicMock()
    client.messages.create.side_effect = [
        json_response([figure(f"Fig. {n}.", f"Chart on page {n}.")]) for n in (1, 2, 3)
    ]
    captioner = make_captioner(tmp_path, client, page_count=3)

    chunks = captioner.caption_document(Path("paper.pdf"))

    assert [chunk.page_number for chunk in chunks] == [1, 2, 3]


@pytest.mark.unit
def test_request_omits_effort_which_the_cheap_model_rejects(tmp_path):
    # The one thing that makes this call shape differ from every other Claude
    # call in the repo. It keeps claude_figure_model swappable down to a
    # cheaper model, which rejects output_config.effort outright and has no
    # adaptive thinking to configure.
    client = MagicMock()
    client.messages.create.return_value = json_response([])
    captioner = make_captioner(tmp_path, client)

    captioner.caption_document(Path("paper.pdf"))

    kwargs = client.messages.create.call_args.kwargs
    assert "effort" not in kwargs["output_config"]
    assert "thinking" not in kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["max_tokens"] == PAGE_MAX_TOKENS


@pytest.mark.unit
def test_request_uses_the_figure_model_not_the_default_model(tmp_path):
    from models.config import settings

    client = MagicMock()
    client.messages.create.return_value = json_response([])
    captioner = make_captioner(tmp_path, client)

    captioner.caption_document(Path("paper.pdf"))

    assert client.messages.create.call_args.kwargs["model"] == settings.claude_figure_model


@pytest.mark.unit
def test_caption_document_only_renders_the_given_pages(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = json_response([figure("Fig. 1.", "A chart.")])
    captioner = make_captioner(tmp_path, client, page_count=5)

    chunks = captioner.caption_document(Path("paper.pdf"), pages=[3])

    assert client.messages.create.call_count == 1
    assert chunks[0].page_number == 3


@pytest.mark.unit
def test_page_image_is_sent_as_the_first_content_block(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = json_response([])
    captioner = make_captioner(tmp_path, client)

    captioner.caption_document(Path("paper.pdf"))

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"


class _FakeObj:
    def __init__(self, obj_type: int):
        self.type = obj_type


class _FakePage:
    def __init__(self, object_types: list[int]):
        self._object_types = object_types
        self.closed = False

    def get_objects(self, max_depth=None):
        return [_FakeObj(t) for t in self._object_types]

    def close(self) -> None:
        self.closed = True


class _FakeDocument:
    """Stands in for pdfium.PdfDocument: page_types[i] lists the object
    types (pdfium.raw.FPDF_PAGEOBJ_*) present on 1-indexed page i + 1."""

    def __init__(self, page_types: list[list[int]]):
        self.pages = [_FakePage(types) for types in page_types]

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self.pages[index]

    def close(self) -> None:
        pass


def structure_element(page: int, role: str) -> SimpleNamespace:
    return SimpleNamespace(page=page, mcid=0, role=role)


@pytest.mark.unit
def test_likely_figure_pages_unions_structure_tags_and_embedded_images(monkeypatch):
    import pypdfium2 as pdfium

    from utils.figure_captioner import likely_figure_pages

    # Page 2 tagged Figure in the structure tree, page 3 has an embedded
    # image object, page 1 has neither -- the union should be [2, 3].
    monkeypatch.setattr(
        "utils.figure_captioner.pdf_inspector.extract_structure_elements",
        lambda path: [structure_element(2, "Figure")],
    )
    monkeypatch.setattr(
        "utils.figure_captioner.pdfium.PdfDocument",
        lambda path: _FakeDocument([[], [], [pdfium.raw.FPDF_PAGEOBJ_IMAGE]]),
    )

    assert likely_figure_pages("paper.pdf") == [2, 3]


@pytest.mark.unit
def test_likely_figure_pages_finds_untagged_pdfs_via_embedded_images(monkeypatch):
    import pypdfium2 as pdfium

    from utils.figure_captioner import likely_figure_pages

    # An untagged PDF returns no structure elements at all; the image check
    # must still work on its own.
    monkeypatch.setattr(
        "utils.figure_captioner.pdf_inspector.extract_structure_elements", lambda path: []
    )
    monkeypatch.setattr(
        "utils.figure_captioner.pdfium.PdfDocument",
        lambda path: _FakeDocument([[pdfium.raw.FPDF_PAGEOBJ_IMAGE], []]),
    )

    assert likely_figure_pages("paper.pdf") == [1]


@pytest.mark.unit
def test_likely_figure_pages_ignores_unrelated_structure_roles(monkeypatch):
    from utils.figure_captioner import likely_figure_pages

    monkeypatch.setattr(
        "utils.figure_captioner.pdf_inspector.extract_structure_elements",
        lambda path: [structure_element(1, "H1"), structure_element(1, "Table")],
    )
    monkeypatch.setattr(
        "utils.figure_captioner.pdfium.PdfDocument", lambda path: _FakeDocument([[]])
    )

    assert likely_figure_pages("paper.pdf") == []


@pytest.mark.unit
def test_likely_figure_pages_returns_empty_when_neither_signal_fires(monkeypatch):
    import pypdfium2 as pdfium

    from utils.figure_captioner import likely_figure_pages

    monkeypatch.setattr(
        "utils.figure_captioner.pdf_inspector.extract_structure_elements", lambda path: []
    )
    monkeypatch.setattr(
        "utils.figure_captioner.pdfium.PdfDocument",
        lambda path: _FakeDocument(
            [[pdfium.raw.FPDF_PAGEOBJ_PATH], [pdfium.raw.FPDF_PAGEOBJ_TEXT]]
        ),
    )

    assert likely_figure_pages("paper.pdf") == []


@pytest.mark.unit
def test_likely_figure_pages_closes_every_page_it_opens(monkeypatch):
    from utils.figure_captioner import likely_figure_pages

    document = _FakeDocument([[], []])
    monkeypatch.setattr(
        "utils.figure_captioner.pdf_inspector.extract_structure_elements", lambda path: []
    )
    monkeypatch.setattr("utils.figure_captioner.pdfium.PdfDocument", lambda path: document)

    likely_figure_pages("paper.pdf")

    assert all(page.closed for page in document.pages)
