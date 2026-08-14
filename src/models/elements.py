"""Schema for the PDF asset pipeline: the object output of parsing.

TWO layers, one join.

  Layer 1  Element     what the parser found. Structure and provenance. One
                       row per thing on a page. Never embedded directly.

  Layer 2  IndexEntry  what retrieval searches. One row per searchable unit.
                       Built FROM elements, references them by id.

Why split: a table is one Element but needs three different text
representations for retrieval (see IndexEntry.embed_text / keyword_text /
display_text). A document summary is an IndexEntry with no single Element
behind it. Figures were originally Elements that never became IndexEntries,
per docs/roadmap.md item 4; that was reversed. utils/index_entry_builder.py
now emits an ElementType.FIGURE entry, carrying the page image in asset_paths,
for every kind="figure" chunk utils/figure_captioner.py writes. No
FigureElement is constructed on the way, because Layer 1 is still unpopulated.

Out of scope on purpose: retrieval results (score fusion, citations) belong
to the separate Postgres-backed search service that consumes this schema's
output, not to the parser that produces it.

No bounding boxes: figures and tables are referenced by page number plus an
asset image path, not by geometry, since nothing downstream reconstructs
layout.

Page numbers are 1-indexed throughout, matching DocumentChunk.page_number
elsewhere in this repo (see docs/architecture.md#page-numbering). Do not
reintroduce a 0-indexed internal field with a computed 1-indexed property;
that asymmetry is exactly the bug class this repo already had to fix once
for pdf_inspector's own API.

Not yet wired to the extractors. This is a target schema for a future
migration of utils/document_extractor.py and utils/pdf_extractor.py away
from the current flat DocumentChunk; see docs/roadmap.md item 6 (extended).

Requires: pydantic>=2
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# =========================================================================
# Shared primitives
# =========================================================================


class SourceFile(Base):
    """The 'shared' fields from the original sketch. One per document, not per element."""

    file_name: str = Field(description="Bare filename, as it appears in citations")
    file_type: str = Field(description="pdf | docx | pptx")
    file_path: str = Field(description="Where it lives now")
    sha256: str
    page_count: int = Field(ge=0)
    title: Optional[str] = None
    ingested_at: datetime = Field(default_factory=_utcnow)


class ParserInfo(Base):
    name: str
    version: str
    model: Optional[str] = None
    parsed_at: datetime = Field(default_factory=_utcnow)


# =========================================================================
# LAYER 1 -- Elements
# =========================================================================


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class TextRole(str, Enum):
    TITLE = "title"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_FURNITURE = "page_furniture"  # header, footer, page number. Drop these.
    FORMULA = "formula"
    CODE = "code"


class ElementBase(Base):
    element_id: str = Field(description="{doc_id}-e{seq:05d}. Stable across reparse.")
    doc_id: str

    # 1-indexed, matching DocumentChunk.page_number elsewhere in this repo.
    page_number: int = Field(ge=1)
    reading_order: int = Field(ge=0, description="Global across the document")

    heading_path: list[str] = Field(
        default_factory=list,
        description="Breadcrumb of enclosing section headers. Prepended to chunk "
        "text at index time, which measurably improves retrieval.",
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    parser: ParserInfo

    def raw_text(self) -> str:
        """Verbatim content. Never summarised, never truncated."""
        raise NotImplementedError


class TextElement(ElementBase):
    type: Literal[ElementType.TEXT] = ElementType.TEXT
    role: TextRole = TextRole.PARAGRAPH
    text: str
    level: Optional[int] = Field(default=None, ge=1)

    def raw_text(self) -> str:
        return self.text


class TableCell(Base):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    text: str = ""
    is_header: bool = False


class TableElement(ElementBase):
    type: Literal[ElementType.TABLE] = ElementType.TABLE
    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)
    markdown: str = Field(description="Rendered. This is the keyword-searchable form.")
    caption: Optional[str] = None
    summary: Optional[str] = Field(
        default=None,
        description="LLM-written, one or two sentences, generated at index time "
        "(utils/metadata_enhancer.py). This is the embeddable form, not the parser's output.",
    )
    csv_path: Optional[str] = None
    image_path: Optional[str] = None

    def raw_text(self) -> str:
        return self.markdown


class FigureElement(ElementBase):
    type: Literal[ElementType.FIGURE] = ElementType.FIGURE
    image_path: str = Field(description="Crop from rendered page. Always populated.")
    render_dpi: int = 144
    caption: Optional[str] = Field(default=None, description="Detected nearby caption")
    description: Optional[str] = Field(
        default=None, description="LLM-written alt text from the rendered crop. Optional, costs money."
    )
    ocr_text: Optional[str] = Field(default=None, description="Text inside the figure")
    classification: Optional[str] = Field(
        default=None, description="chart | diagram | photo | logo | signature"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_furniture(self) -> bool:
        """Logos and signatures are noise. Filter on this, not on size."""
        return self.classification in {"logo", "signature"}

    def raw_text(self) -> str:
        return "\n".join(p for p in (self.caption, self.description, self.ocr_text) if p)


Element = Annotated[
    Union[TextElement, TableElement, FigureElement], Field(discriminator="type")
]


class ParsedDocument(Base):
    doc_id: str = Field(description="First 12 hex of sha256. Short, stable, no uuid.")
    source: SourceFile
    parser: ParserInfo
    elements: list[Element] = Field(default_factory=list)
    doc_summary: Optional[str] = Field(
        default=None, description="LLM-written. Feeds the summary-level IndexEntry (TODO item 3)."
    )

    @staticmethod
    def make_doc_id(sha256: str) -> str:
        return sha256[:12]

    @staticmethod
    def hash_file(path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while block := fh.read(1 << 20):
                h.update(block)
        return h.hexdigest()


# =========================================================================
# LAYER 2 -- Index entries
# =========================================================================


class Granularity(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    ELEMENT = "element"


class IndexEntry(Base):
    """
    One searchable unit. THREE text fields, deliberately.

    Consider the sample question in TODO, which needs "-0.59% GDP contraction"
    retrieved from a table on page 12:

      embed_text    caption + LLM summary. Dense search finds this from
                    "who bears the cost of heat pumps". Would never match
                    "-0.59%", because a number has no semantic neighbourhood.
      keyword_text  full markdown, every cell. Postgres FTS matches "-0.59"
                    exactly. This is the entire justification for hybrid search.
      display_text  what the LLM sees after retrieval. Full markdown, so it
                    can read the actual value rather than a summary of it.

    For plain prose all three are the same string. For tables they differ,
    and collapsing them loses either recall or precision.
    """

    chunk_id: str = Field(
        description="{doc_id}-p{page:03d}-c{seq:02d}, e.g. 'a3f2c19b40e1-p012-c04'. "
        "Short and typo-resistant: an LLM has to emit this verbatim for citations."
    )

    doc_id: str
    file_name: str
    file_type: str

    granularity: Granularity = Granularity.ELEMENT

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    element_ids: list[str] = Field(
        default_factory=list,
        description="Join back to Layer 1. Empty for document-level summaries.",
    )
    element_types: list[ElementType] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)

    embed_text: str
    keyword_text: str
    display_text: str

    asset_paths: list[str] = Field(
        default_factory=list, description="Table CSVs, figure crops, page images"
    )
    n_chars: int = 0
    embedding: Optional[list[float]] = Field(
        default=None, description="Dense vector over embed_text. None until an embedding step fills it in."
    )
    embedding_model: Optional[str] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"{self.file_name} p.{self.page_start}"
        return f"{self.file_name} pp.{self.page_start}-{self.page_end}"

    @classmethod
    def from_text_elements(
        cls, doc: ParsedDocument, elements: list[TextElement], seq: int
    ) -> "IndexEntry":
        """Prose chunk. Heading path prepended to the embedded form only."""
        body = "\n\n".join(e.text for e in elements)
        crumb = " > ".join(elements[0].heading_path) if elements[0].heading_path else ""
        pages = [e.page_number for e in elements]
        return cls(
            chunk_id=f"{doc.doc_id}-p{min(pages):03d}-c{seq:02d}",
            doc_id=doc.doc_id,
            file_name=doc.source.file_name,
            file_type=doc.source.file_type,
            granularity=Granularity.ELEMENT,
            page_start=min(pages),
            page_end=max(pages),
            element_ids=[e.element_id for e in elements],
            element_types=[ElementType.TEXT],
            heading_path=elements[0].heading_path,
            embed_text=f"{crumb}\n\n{body}".strip(),
            keyword_text=body,
            display_text=body,
            n_chars=len(body),
        )

    @classmethod
    def from_table(cls, doc: ParsedDocument, el: TableElement, seq: int) -> "IndexEntry":
        """The interesting case. Note the three fields diverge."""
        crumb = " > ".join(el.heading_path)
        semantic = "\n".join(p for p in (crumb, el.caption, el.summary) if p) or el.markdown[:500]
        return cls(
            chunk_id=f"{doc.doc_id}-p{el.page_number:03d}-c{seq:02d}",
            doc_id=doc.doc_id,
            file_name=doc.source.file_name,
            file_type=doc.source.file_type,
            granularity=Granularity.ELEMENT,
            page_start=el.page_number,
            page_end=el.page_number,
            element_ids=[el.element_id],
            element_types=[ElementType.TABLE],
            heading_path=el.heading_path,
            embed_text=semantic,
            keyword_text="\n".join(p for p in (el.caption, el.markdown) if p),
            display_text="\n\n".join(p for p in (el.caption, el.markdown) if p),
            asset_paths=[p for p in (el.csv_path, el.image_path) if p],
            n_chars=len(el.markdown),
        )
