from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from utils.page_renderer import RenderedPage
from utils.vision_extractor import VisionExtractor


def text_response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
    )


class FakeRenderer:
    """Stands in for PageRenderer so tests never touch a real PDF."""

    def __init__(self, pages: dict[int, bytes]):
        self._pages = pages

    def render(self, pdf_path, pages=None) -> list[RenderedPage]:
        wanted = pages if pages is not None else sorted(self._pages)
        return [RenderedPage(page_number=n, image_path=_FakePath(self._pages[n])) for n in wanted]


class _FakePath:
    def __init__(self, data: bytes):
        self._data = data

    def read_bytes(self) -> bytes:
        return self._data


def make_extractor(client, pages=None) -> VisionExtractor:
    return VisionExtractor(
        image_dir="unused",
        client=client,
        renderer=FakeRenderer(pages or {1: b"page one", 2: b"page two"}),
        max_workers=1,
    )


@pytest.mark.unit
def test_transcription_becomes_chunks_with_page_provenance():
    client = MagicMock()
    client.messages.create.return_value = text_response("## Results\n\nThe model converged.")

    chunks = make_extractor(client, pages={4: b"page four"}).extract_chunks("paper.pdf")

    assert len(chunks) == 1
    assert chunks[0].file_name == "paper.pdf"
    assert chunks[0].page_number == 4
    assert chunks[0].headings == ["Results"]
    assert "The model converged." in chunks[0].text


@pytest.mark.unit
def test_chunk_ids_are_unique_across_pages():
    """_page_to_chunks numbers within a page, so ids must be reassigned over
    the document or page 2's first chunk collides with page 1's."""
    client = MagicMock()
    client.messages.create.return_value = text_response("Body text.")

    chunks = make_extractor(client).extract_chunks("paper.pdf")

    assert [chunk.chunk_id for chunk in chunks] == [
        "paper.pdf#p1#c0",
        "paper.pdf#p2#c1",
    ]


@pytest.mark.unit
def test_blank_page_produces_no_chunks():
    client = MagicMock()
    client.messages.create.return_value = text_response("   \n  ")

    assert make_extractor(client).extract_chunks("paper.pdf") == []


@pytest.mark.unit
def test_refusal_is_treated_as_an_empty_page_not_a_crash():
    client = MagicMock()
    client.messages.create.return_value = text_response("", stop_reason="refusal")

    page = RenderedPage(page_number=1, image_path=_FakePath(b"bytes"))

    assert make_extractor(client).transcribe_page(page) == ""


@pytest.mark.unit
def test_page_image_is_sent_before_the_instruction():
    """Claude attends to an image placed ahead of the text that refers to it."""
    client = MagicMock()
    client.messages.create.return_value = text_response("Body text.")

    make_extractor(client, pages={1: b"page one"}).extract_chunks("paper.pdf")

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1]["type"] == "text"


@pytest.mark.unit
def test_max_chars_is_a_hard_bound_on_chunk_text():
    client = MagicMock()
    client.messages.create.return_value = text_response("word " * 500)

    extractor = make_extractor(client, pages={1: b"page one"})
    extractor.max_chars = 200

    chunks = extractor.extract_chunks("paper.pdf")

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)


@pytest.mark.unit
def test_one_model_call_per_page():
    client = MagicMock()
    client.messages.create.return_value = text_response("Body text.")

    make_extractor(client, pages={1: b"a", 2: b"b", 3: b"c"}).extract_chunks("paper.pdf")

    assert client.messages.create.call_count == 3
