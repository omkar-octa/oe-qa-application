"""Adapter from the flat DocumentChunk extraction still produces today to the
IndexEntry target schema in models/elements.py.

Deliberately skips Layer 1 (TextElement/TableElement): nothing in this
codebase produces those yet, so this goes straight from DocumentChunk to
IndexEntry instead of using IndexEntry.from_text_elements/from_table.

Pure and deterministic: no embedder calls, no network, no side effects.
Embedding is a separate step that runs elsewhere."""

from pathlib import Path

from models.documents import DocumentChunk
from models.elements import ElementType, IndexEntry, ParsedDocument

# IndexEntry.from_table falls back to el.markdown[:500] when there is no
# LLM summary yet; mirrored here for chunks that haven't been through
# MetadataEnhancer.
_EMBED_TEXT_FALLBACK_CHARS = 500


def build_index_entries(
    chunks: list[DocumentChunk], pdf_path: Path, image_dir: Path | None = None
) -> list[IndexEntry]:
    """Converts DocumentChunks from a single source PDF into IndexEntries.

    All chunks must come from the same source file: the file is hashed once
    to derive doc_id, not once per chunk.

    image_dir is where PageRenderer wrote the rendered pages. Supplied, figure
    chunks carry their page image in asset_paths; omitted, they get an empty
    list like everything else, so a caller that never rendered anything does
    not have to know about it.
    """
    sha256 = ParsedDocument.hash_file(pdf_path)
    doc_id = ParsedDocument.make_doc_id(sha256)
    file_type = pdf_path.suffix.lstrip(".")

    entries = []
    for seq, chunk in enumerate(chunks):
        asset_paths: list[str] = []
        if chunk.kind == "table":
            keyword_text = chunk.text
            display_text = chunk.text
            embed_text = chunk.summary if chunk.summary else chunk.text[:_EMBED_TEXT_FALLBACK_CHARS]
            element_types = [ElementType.TABLE]
        elif chunk.kind == "figure":
            # A figure chunk's text is already caption plus description, which
            # is both the only searchable form of the figure and the only
            # readable one, so the three fields do not diverge the way a
            # table's do.
            embed_text = keyword_text = display_text = chunk.text
            element_types = [ElementType.FIGURE]
            if image_dir is not None:
                asset_paths = [
                    str(Path(image_dir) / pdf_path.stem / f"p{chunk.page_number:04d}.png")
                ]
        else:
            embed_text = keyword_text = display_text = chunk.text
            element_types = [ElementType.TEXT]

        entries.append(
            IndexEntry(
                chunk_id=f"{doc_id}-p{chunk.page_number:03d}-c{seq:02d}",
                doc_id=doc_id,
                file_name=chunk.file_name,
                file_type=file_type,
                page_start=chunk.page_number,
                page_end=chunk.page_number,
                element_ids=[],
                element_types=element_types,
                heading_path=chunk.headings,
                embed_text=embed_text,
                keyword_text=keyword_text,
                display_text=display_text,
                asset_paths=asset_paths,
                n_chars=len(chunk.text),
                embedding=None,
                embedding_model=None,
            )
        )

    return entries
