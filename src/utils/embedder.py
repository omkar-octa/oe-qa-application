"""Dense embeddings via OpenAI's API.

Replaces the local BGE-M3 stack this repo used to run (utils/embedder.py's
own previous implementation, plus utils/embedding_daemon.py and
utils/embedding_client.py, both deleted). BGE-M3 measured at ~180
characters/second on this machine's CPU: fine for a single query at ask
time, but a bulk `index` run over even this small corpus (~670,000
characters of embed_text) worked out to roughly an hour. A remote API call
has no local model to load, which was the daemon's entire reason to exist --
there is nothing left to amortise, so there is nothing left to run as a
resident process. Every caller just constructs an Embedder and calls it.

Requires OPENAI_API_KEY, in src/.env or the real OS environment (both work:
Settings reads either, and if for any reason settings.openai_api_key ends up
unset, the OpenAI SDK's own client falls back to checking the environment
variable itself).
"""

import logging

from openai import OpenAI

from models.config import settings
from models.documents import DocumentChunk

logger = logging.getLogger(__name__)


class Embedder:
    """Calls OpenAI's embeddings API. Same public interface the previous,
    locally-run Embedder had (embed_texts, embed_chunks), so callers that
    just need *an* embedder -- main.py, PostgresSearchIndex -- did not need
    to change."""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        self._client = client or OpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dimensions
        # Cumulative across every embed_texts() call this instance has made,
        # so a caller sharing one Embedder across a whole `ask` (search +
        # search_documents, possibly several rounds) can read a running total
        # rather than summing per-call return values itself.
        self.total_tokens = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.debug(
            "embedding %d text(s) with %s (dimensions=%s)", len(texts), self._model, self._dimensions
        )
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        if response.usage:
            self.total_tokens += response.usage.total_tokens
            logger.debug(
                "embedding usage: %d token(s) (cumulative %d)",
                response.usage.total_tokens,
                self.total_tokens,
            )
        return [item.embedding for item in response.data]

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Returns new DocumentChunks with `embedding` populated; the input
        chunks are left unmodified."""
        if not chunks:
            return []
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        return [chunk.model_copy(update={"embedding": vector}) for chunk, vector in zip(chunks, vectors)]
