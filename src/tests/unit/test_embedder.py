from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.documents import DocumentChunk
from utils.embedder import Embedder


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=chunk_id, file_name="paper.pdf", page_number=1, text=text)


def embeddings_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=vector) for vector in vectors],
        usage=SimpleNamespace(total_tokens=len(vectors) * 5),
    )


def mocked_client(vectors: list[list[float]]) -> MagicMock:
    client = MagicMock()
    client.embeddings.create.return_value = embeddings_response(vectors)
    return client


@pytest.mark.unit
def test_embed_texts_returns_dense_vectors():
    client = mocked_client([[0.1, 0.2], [0.3, 0.4]])

    vectors = Embedder(client=client).embed_texts(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.unit
def test_embed_texts_empty_list_returns_empty_without_calling_the_api():
    client = mocked_client([])

    vectors = Embedder(client=client).embed_texts([])

    assert vectors == []
    client.embeddings.create.assert_not_called()


@pytest.mark.unit
def test_embed_texts_passes_model_and_dimensions():
    client = mocked_client([[0.1]])

    Embedder(client=client, model="text-embedding-3-large", dimensions=512).embed_texts(["hello"])

    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-large"
    assert kwargs["dimensions"] == 512
    assert kwargs["input"] == ["hello"]


@pytest.mark.unit
def test_embed_chunks_populates_embedding_without_mutating_input():
    chunk = make_chunk("c0", "hello world")
    client = mocked_client([[0.5, 0.6]])

    result = Embedder(client=client).embed_chunks([chunk])

    assert chunk.embedding is None
    assert result[0].embedding == [0.5, 0.6]
    assert result[0].chunk_id == "c0"


@pytest.mark.unit
def test_embed_chunks_empty_list_returns_empty_without_calling_the_api():
    client = mocked_client([])

    result = Embedder(client=client).embed_chunks([])

    assert result == []
    client.embeddings.create.assert_not_called()
