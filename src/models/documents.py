from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single searchable extract of a source document.

    page_number is 1-indexed; 0 means the page could not be determined.
    embedding is reserved for a future vector search index.
    summary is an LLM-written gloss of the chunk (utils.metadata_enhancer),
    useful for chunks -- tables especially -- whose raw text alone is a poor
    embedding target.
    kind distinguishes table chunks from prose, needed downstream to decide
    which chunks get a divergent embed/keyword/display text split. A "figure"
    chunk is written by utils.figure_captioner: its text is a figure's caption
    followed by an LLM-written description of what the figure shows, which is
    the only way a value printed on a chart or map becomes searchable at all.
    """

    chunk_id: str
    file_name: str
    page_number: int
    text: str
    headings: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    summary: str | None = None
    kind: Literal["text", "table", "figure"] = "text"


class DocumentSummary(BaseModel):
    """An LLM-written gloss of a whole source file, written once per document
    rather than once per chunk (utils.document_summarizer).

    It is scope and inventory, not an abstract: what the document covers, over
    what period and territory, which sections it has, and where its numbers
    live. That is what answers "am I in the right document, and is this extract
    worth escalating to read_source", which is the only job it has.

    Deliberately not a DocumentChunk. It is never ranked against chunks, never
    returned as a match and never cited: it rides along with the results from
    its own file. Keyed by file_name because that is what chunks and citations
    use; the Postgres store keys the same summary by doc_id instead, since it
    has the file hash.
    """

    file_name: str
    summary: str
    page_count: int = 0
    embedding: list[float] | None = None


class SearchIndex(BaseModel):
    """The persisted knowledge base: a pure content store with no search
    statistics, so the on-disk format is independent of the retrieval method.

    `documents` is empty unless `ingest --summarise-documents` ran, which keeps
    an index written before that flag existed loadable."""

    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chunks: list[DocumentChunk] = Field(default_factory=list)
    documents: list[DocumentSummary] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A matching chunk plus the chunks that surround it in the source document.

    Chunking splits on a character budget, not on meaning, so a match very
    often starts or ends mid-sentence and a table's rows can land in a
    different chunk from its header. The neighbours carry no score and are
    never cited; they exist so the matched text reads as it does on the page.

    doc_summary is the DocumentSummary text for the file this chunk came from,
    when one exists. It is carried on the result rather than fetched through a
    second retriever method so the backend protocol stays search() alone; every
    result from the same file repeats it, and the formatter prints it once.
    """

    chunk: DocumentChunk
    score: float
    context_before: list[DocumentChunk] = Field(default_factory=list)
    context_after: list[DocumentChunk] = Field(default_factory=list)
    doc_summary: str | None = None
