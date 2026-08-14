"""Fast text extraction for PDFs that already carry an embedded text layer.

This is the cheap path alongside utils.document_extractor.DocumentExtractor:
no OCR, no layout model, just the text pdf_inspector (Rust, no heavy
dependencies) can read natively, with native heading, table, and
multi-column detection. pdf_inspector classifies per page, so a document
that mixes born-digital and scanned pages is handled correctly: this
extractor returns chunks for the readable pages only, and callers use
pages_needing_ocr() to route the remaining pages to Docling.
"""

import re
from pathlib import Path

import pdf_inspector

from models.documents import DocumentChunk

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW_PATTERN = re.compile(r"^\s*\|")
_TABLE_CAPTION_PATTERN = re.compile(r"^\s*[*_]*\s*Table\s+[A-Za-z]?\d+", re.IGNORECASE)

# A chunk is a table when it carries at least this many table rows. One
# pipe-prefixed line in prose is not a table, and classifying it as one would
# send it down the summarise-for-embedding path in index_entry_builder.
_MIN_TABLE_ROWS = 2

# Caption lines pulled down into a table block. Journal captions run to a
# label line plus a description line ("Table 5" / "Decomposition of ...").
_MAX_CAPTION_LINES = 2


class PdfExtractor:
    """Extracts markdown text from PDFs via pdf_inspector.

    pdf_inspector reports per-page OCR need, so a page requiring OCR is
    simply excluded from extract_chunks() rather than failing the whole
    document. Use pages_needing_ocr() to find out which pages were skipped.
    """

    def __init__(self, max_chars: int = 3000):
        self.max_chars = max_chars

    def pages_needing_ocr(self, pdf_path: Path | str) -> list[int]:
        """1-indexed page numbers that need OCR instead of this extractor."""
        return pdf_inspector.extract_pages_markdown(str(pdf_path)).pages_needing_ocr

    def extract_chunks(
        self, pdf_path: Path | str, max_chars: int | None = None
    ) -> list[DocumentChunk]:
        """Split the readable pages into DocumentChunk objects carrying file
        name and page number provenance. Pages needing OCR are skipped; use
        pages_needing_ocr() to find them and route them to Docling instead."""
        max_chars = max_chars or self.max_chars
        file_name = Path(pdf_path).name

        result = pdf_inspector.extract_pages_markdown(str(pdf_path))

        chunks: list[DocumentChunk] = []
        for page in result.pages:
            if page.needs_ocr or not page.markdown.strip():
                continue
            # pdf_inspector's per-page `page` field is 0-indexed; our schema
            # (and pages_needing_ocr, confusingly) is 1-indexed.
            page_number = page.page + 1
            chunks.extend(self._page_to_chunks(file_name, page_number, page.markdown, max_chars))

        return chunks

    @classmethod
    def _page_to_chunks(
        cls, file_name: str, page_number: int, markdown: str, max_chars: int
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        buffer: list[str] = []
        current_headings: list[str] = []

        def flush() -> None:
            nonlocal buffer
            text = "\n".join(buffer).strip()
            if text:
                # pdf_inspector reports table structure per page, not per
                # chunk, so kind is derived from the chunk's own content. It
                # counts table rows rather than testing the first line: a real
                # table is introduced by its caption, and on these fixtures a
                # running header lands ahead of that again, so a first-line
                # test classifies whole tables as prose and header-less
                # fragments as tables. Splitting tables out below means a
                # fragment no longer exists to misclassify.
                rows = [
                    line for line in text.splitlines() if _TABLE_ROW_PATTERN.match(line)
                ]
                kind = "table" if len(rows) >= _MIN_TABLE_ROWS else "text"
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{file_name}#p{page_number}#c{len(chunks)}",
                        file_name=file_name,
                        page_number=page_number,
                        text=text,
                        headings=list(current_headings),
                        kind=kind,
                    )
                )
            buffer = []

        def append_segment(segment: str) -> None:
            # A full buffer starts a new chunk; +1 accounts for the join()
            # newline so max_chars is a true bound on the resulting chunk.
            if buffer and sum(map(len, buffer)) + len(buffer) + len(segment) > max_chars:
                flush()
            buffer.append(segment)

        for block_kind, block_lines in cls._blocks(markdown.splitlines()):
            if block_kind == "table":
                # A table gets its own chunk, never sharing one with the prose
                # around it. index_entry_builder embeds a table's summary but
                # keyword-searches its Markdown, and that split is only
                # coherent if the chunk is a table and nothing else.
                for part in cls._split_table(block_lines, max_chars):
                    flush()
                    for line in part:
                        for segment in cls._split_long_line(line, max_chars):
                            append_segment(segment)
                flush()
                continue

            for line in block_lines:
                heading_match = _HEADING_PATTERN.match(line.strip())
                if heading_match:
                    flush()
                    current_headings = [heading_match.group(2).strip()]

                # A single markdown line (e.g. a wide table row) can itself
                # exceed max_chars; split it on word boundaries so the hard
                # bound holds regardless of line length.
                for segment in cls._split_long_line(line, max_chars):
                    append_segment(segment)

        flush()
        return chunks

    @classmethod
    def _blocks(cls, lines: list[str]) -> list[tuple[str, list[str]]]:
        """Group page lines into blocks, where a run of Markdown table rows and
        the caption introducing it form one block that must not be split.

        A blank line ends a table run: consecutive pipe-prefixed lines are the
        table, anything else is prose."""
        blocks: list[tuple[str, list[str]]] = []
        text_lines: list[str] = []
        table_lines: list[str] = []

        def close_text() -> None:
            nonlocal text_lines
            if text_lines:
                blocks.append(("text", text_lines))
                text_lines = []

        def close_table() -> None:
            nonlocal table_lines
            if table_lines:
                blocks.append(("table", table_lines))
                table_lines = []

        for line in lines:
            if _TABLE_ROW_PATTERN.match(line):
                if not table_lines:
                    caption = cls._take_caption(text_lines)
                    close_text()
                    table_lines.extend(caption)
                table_lines.append(line)
                continue
            close_table()
            text_lines.append(line)

        close_table()
        close_text()
        return blocks

    @staticmethod
    def _take_caption(text_lines: list[str]) -> list[str]:
        """Move a "Table N" caption off the end of the preceding prose so it
        stays with the table it introduces. Mutates text_lines; returns [] and
        leaves it untouched when the preceding lines are ordinary prose."""
        while text_lines and not text_lines[-1].strip():
            text_lines.pop()

        taken: list[str] = []
        for _ in range(_MAX_CAPTION_LINES):
            if not text_lines or not text_lines[-1].strip():
                break
            taken.insert(0, text_lines.pop())
            if _TABLE_CAPTION_PATTERN.match(taken[0]):
                return taken

        text_lines.extend(taken)
        return []

    @classmethod
    def _split_table(cls, block_lines: list[str], max_chars: int) -> list[list[str]]:
        """Split an over-long table so every part repeats the caption and header.

        A part that is a run of numbers with no column names is worse than
        useless: the values cannot be attributed to a variable, and a reader
        (or a model) can attribute them to the wrong one."""
        if len("\n".join(block_lines)) <= max_chars:
            return [block_lines]

        caption = [line for line in block_lines if not _TABLE_ROW_PATTERN.match(line)]
        rows = [line for line in block_lines if _TABLE_ROW_PATTERN.match(line)]
        separator = next(
            (i for i, line in enumerate(rows) if cls._is_table_separator(line)), None
        )
        header = rows[: separator + 1] if separator is not None else rows[:1]
        body = rows[len(header) :]

        # Repeating a prefix only helps if it leaves room for actual rows.
        # Drop the caption first, then the header, rather than emitting parts
        # that are almost entirely repeated preamble.
        prefix = caption + header
        if len("\n".join(prefix)) * 2 > max_chars:
            prefix = header
        if len("\n".join(prefix)) * 2 > max_chars:
            prefix = []

        parts: list[list[str]] = []
        current: list[str] = []
        for row in body:
            if current and len("\n".join(prefix + current + [row])) > max_chars:
                parts.append(prefix + current)
                current = []
            current.append(row)
        if current:
            parts.append(prefix + current)
        return parts or [block_lines]

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        """True for a Markdown header rule such as `| --- | :---: |`."""
        stripped = line.strip()
        return (
            stripped.startswith("|")
            and "-" in stripped
            and set(stripped) <= set("|-: \t")
        )

    @staticmethod
    def _split_long_line(line: str, max_chars: int) -> list[str]:
        if len(line) <= max_chars:
            return [line]

        segments: list[str] = []
        current = ""
        for word in line.split(" "):
            candidate = f"{current} {word}" if current else word
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                segments.append(current)
            current = word
            while len(current) > max_chars:
                segments.append(current[:max_chars])
                current = current[max_chars:]
        if current:
            segments.append(current)
        return segments
