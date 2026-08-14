import argparse
import logging
import sys
import time
from pathlib import Path

from models.config import settings


def ingest(
    source: Path,
    output: Path,
    enhance_metadata: bool = False,
    extractor: str = "fast",
    image_dir: Path = Path("data/pages"),
    caption_figures: bool = True,
    summarise_documents: bool = False,
) -> int:
    from utils.document_extractor import DocumentExtractor
    from utils.pdf_extractor import PdfExtractor

    # A single file runs the whole pipeline on just that PDF, for interrogating
    # its output without the rest of the corpus in the way.
    pdf_paths = [source] if source.is_file() else sorted(source.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {source}")
        return 1

    if extractor == "vision":
        return _ingest_vision(
            pdf_paths, output, image_dir, enhance_metadata, caption_figures, summarise_documents
        )

    # PdfExtractor (pdf_inspector) is the cheap path: no OCR, no layout
    # model. It classifies per page, so a document that mixes born-digital
    # and scanned pages is handled correctly -- only the specific pages it
    # reports as needing OCR are routed to DocumentExtractor (Docling),
    # built lazily so it never loads its OCR/layout models unless some PDF
    # actually needs them.
    fast_extractor = PdfExtractor()
    slow_extractor = None
    chunks = []
    for pdf_path in pdf_paths:
        print(f"Extracting {pdf_path.name} ...")
        try:
            file_chunks = fast_extractor.extract_chunks(pdf_path)
            ocr_pages = fast_extractor.pages_needing_ocr(pdf_path)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        if ocr_pages:
            print(f"  pages needing OCR: {ocr_pages}, falling back to Docling for those ...")
            if slow_extractor is None:
                slow_extractor = DocumentExtractor()
            try:
                file_chunks = file_chunks + slow_extractor.extract_chunks_for_pages(pdf_path, ocr_pages)
            except Exception as exc:
                print(f"  OCR fallback FAILED: {exc}")
            file_chunks.sort(key=lambda chunk: chunk.page_number)

        print(f"  {len(file_chunks)} chunks")
        chunks.extend(file_chunks)

    return _write_index(
        chunks, output, pdf_paths, enhance_metadata, caption_figures, image_dir, summarise_documents
    )


def _ingest_vision(
    pdf_paths: list[Path],
    output: Path,
    image_dir: Path,
    enhance_metadata: bool,
    caption_figures: bool = True,
    summarise_documents: bool = False,
) -> int:
    """Transcribe every page with Claude's vision instead of parsing the PDF.

    There is no fast/slow split here: the model reads a rendered page whether
    or not the PDF carries a text layer, so scanned and born-digital pages
    take the same path and cost the same.
    """
    from utils.vision_extractor import VisionExtractor

    extractor = VisionExtractor(image_dir)
    chunks = []
    for pdf_path in pdf_paths:
        page_count = extractor._renderer.page_count(pdf_path)
        print(f"Transcribing {pdf_path.name} ({page_count} pages) ...")
        try:
            file_chunks = extractor.extract_chunks(pdf_path)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        if extractor.failed_pages:
            print(f"  WARNING: no text from pages {extractor.failed_pages}")
        print(f"  {len(file_chunks)} chunks")
        chunks.extend(file_chunks)

    return _write_index(
        chunks, output, pdf_paths, enhance_metadata, caption_figures, image_dir, summarise_documents
    )


# A recovered "text" chunk (see utils.figure_captioner) exists to catch body
# text a layout model dropped, not to duplicate text the primary extractor
# already got right. 500 characters comfortably clears the 114-character
# JEL/keywords-only chunk that was all Burlinson page 1 yielded before this
# existed (see docs/roadmap.md), while staying low enough to still recover a
# genuinely thin page.
_DEDUP_TEXT_THRESHOLD_CHARS = 500


def _drop_duplicate_recovered_text(figure_chunks: list, existing_chunks: list) -> tuple[list, int]:
    """Filters recovered-text chunks for pages the primary extractor already
    covered well, so a page that extracted fine doesn't also gain a
    re-transcribed duplicate. Only kind="text" chunks are ever dropped here;
    genuine kind="figure" chunks always pass through untouched."""
    existing_lengths: dict[int, int] = {}
    for chunk in existing_chunks:
        if chunk.kind in ("text", "table"):
            existing_lengths[chunk.page_number] = (
                existing_lengths.get(chunk.page_number, 0) + len(chunk.text)
            )

    kept, dropped = [], 0
    for chunk in figure_chunks:
        thin_enough = existing_lengths.get(chunk.page_number, 0) < _DEDUP_TEXT_THRESHOLD_CHARS
        if chunk.kind != "text" or thin_enough:
            kept.append(chunk)
        else:
            dropped += 1
    return kept, dropped


def _caption_figures(chunks: list, pdf_paths: list[Path], image_dir: Path) -> list:
    """Adds one chunk per figure, merged into each file's chunks in page order.

    Detection runs on the rendered page rather than on the PDF's structure, so
    this is independent of which extractor produced `chunks` and works the same
    behind --extractor fast and --extractor vision. Rendered pages are cached,
    so a corpus already ingested with --extractor vision pays no rasterisation
    cost here.

    Each file is first narrowed to its likely_figure_pages() candidates
    (structure-tree tags unioned with embedded image objects, see
    utils.figure_captioner), so a page with neither signal never costs a
    Claude call. If that detection itself fails, this falls back to captioning
    every page rather than silently skipping the file -- but does not also
    fall back to Docling-cropping every page, since that would spend the one
    cost this whole filter exists to bound.

    Those same candidates are also run through Docling's layout model to save
    real cropped figure images alongside the Claude-written descriptions (see
    utils.document_extractor.extract_figure_images); a crop failure is logged
    and does not stop captioning for that file. Nothing yet binds a specific
    crop to a specific described figure -- a page can hold more than one of
    each, in orders that need not line up -- so the crops exist on disk as
    their own artifact rather than as chunk metadata.

    FigureCaptioner also returns kind="text" chunks recovering body text a
    layout model misclassified as a picture (see its own docstring); those go
    through _drop_duplicate_recovered_text first so a page the primary
    extractor already covered well doesn't also gain a re-transcribed
    duplicate.
    """
    from utils.document_extractor import extract_figure_images
    from utils.figure_captioner import FigureCaptioner, likely_figure_pages

    captioner = FigureCaptioner(image_dir)
    by_file: dict[str, list] = {}
    for chunk in chunks:
        by_file.setdefault(chunk.file_name, []).append(chunk)

    total = 0
    total_crops = 0
    for pdf_path in pdf_paths:
        page_count = captioner._renderer.page_count(pdf_path)
        try:
            candidates = likely_figure_pages(pdf_path)
        except Exception as exc:
            print(f"  {pdf_path.name}: figure-page detection FAILED ({exc}); captioning every page")
            candidates = None

        if candidates is not None and not candidates:
            print(f"Captioning figures in {pdf_path.name}: no candidate pages found, skipping")
            continue

        if candidates:
            try:
                crops = extract_figure_images(pdf_path, candidates, image_dir)
                n_crops = sum(len(paths) for paths in crops.values())
                total_crops += n_crops
                print(f"  {n_crops} figure crops saved via Docling")
            except Exception as exc:
                print(f"  figure crop extraction FAILED: {exc}")

        described = len(candidates) if candidates is not None else page_count
        print(f"Captioning figures in {pdf_path.name} ({described} of {page_count} pages) ...")
        try:
            figure_chunks = captioner.caption_document(pdf_path, pages=candidates)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        if captioner.failed_pages:
            print(f"  WARNING: no figures read from pages {captioner.failed_pages}")

        file_chunks = by_file.setdefault(pdf_path.name, [])
        figure_chunks, dropped = _drop_duplicate_recovered_text(figure_chunks, file_chunks)
        if dropped:
            print(f"  dropped {dropped} recovered-text chunk(s) as likely duplicates of existing extraction")
        print(f"  {len(figure_chunks)} figures")
        total += len(figure_chunks)

        # Sorting after the append relies on sort being stable: each page's
        # figures land after that page's text rather than interleaved with it,
        # and text chunk order within a page is untouched. Chunk order in the
        # written index is page order, so a chunk out of place here stays
        # visibly wrong in the JSON rather than only at query time.
        file_chunks.extend(figure_chunks)
        file_chunks.sort(key=lambda chunk: chunk.page_number)

    print(f"Added {total} figure chunks, {total_crops} crop images saved")
    return [chunk for file_chunks in by_file.values() for chunk in file_chunks]


def _write_index(
    chunks: list,
    output: Path,
    pdf_paths: list[Path],
    enhance_metadata: bool,
    caption_figures: bool = True,
    image_dir: Path = Path("data/pages"),
    summarise_documents: bool = False,
) -> int:
    from models.documents import SearchIndex

    if not chunks:
        print("No content extracted; index not written.")
        return 1

    # Before enhance_metadata, so figure chunks are summarised too when both
    # flags are on.
    if caption_figures:
        chunks = _caption_figures(chunks, pdf_paths, image_dir)

    if enhance_metadata:
        from utils.metadata_enhancer import DEFAULT_SUMMARISED_KINDS, MetadataEnhancer

        summarisable = sum(1 for chunk in chunks if chunk.kind in DEFAULT_SUMMARISED_KINDS)
        print(f"Summarising {summarisable} of {len(chunks)} chunks with Claude ...")
        enhancer = MetadataEnhancer()
        chunks = enhancer.enhance_chunks(chunks)
        if enhancer.failed_chunks:
            print(f"  WARNING: no summary produced for {enhancer.failed_chunks}")

    # Last, so a profile is written over the final chunk list: with
    # figure captioning on (the default), that is what lets it say the
    # document's charts were read, and with --no-caption-figures, what stops
    # it claiming they were.
    documents = []
    if summarise_documents:
        from utils.document_summarizer import DocumentSummarizer

        file_count = len({chunk.file_name for chunk in chunks})
        print(f"Profiling {file_count} documents with Claude ...")
        documents = DocumentSummarizer().summarize_documents(chunks)
        print(f"  wrote {len(documents)} document profiles")

    index = SearchIndex(chunks=chunks, documents=documents)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks from {len(pdf_paths)} PDFs to {output}")
    return 0


def index(index_path: Path, source: Path, image_dir: Path = Path("data/pages")) -> int:
    """Embeds every chunk in the extraction index and upserts it into Postgres.

    Separate from `ingest`: extraction is free and offline, this step costs
    one embedding call per file and needs Postgres reachable, so it stays a
    deliberate second step rather than a side effect of ingest.
    """
    from collections import defaultdict

    from models.documents import SearchIndex
    from utils.embedder import Embedder
    from utils.index_entry_builder import build_index_entries
    from utils.postgres_store import connect, ensure_schema, upsert_document

    # A single file's own parent directory is exactly where its lookup below
    # (source / file_name) needs to find it, so accept it directly rather
    # than requiring a directory even when there is only one PDF to index.
    if source.is_file():
        source = source.parent

    if not index_path.exists():
        print(f"Index not found at {index_path}. Run 'python main.py ingest' first.")
        return 1

    search_index = SearchIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    chunks_by_file: dict[str, list] = defaultdict(list)
    for chunk in search_index.chunks:
        chunks_by_file[chunk.file_name].append(chunk)

    # Empty unless `ingest --summarise-documents` ran. Profiles are written
    # there rather than here so the Claude call happens once, at extraction
    # time, and both backends read the same text; this step only embeds it.
    summaries = {doc.file_name: doc for doc in search_index.documents}

    embedder = Embedder()
    conn = connect()
    ensure_schema(conn)

    indexed_files = 0
    for file_name, file_chunks in chunks_by_file.items():
        pdf_path = source / file_name
        if not pdf_path.exists():
            print(f"  SKIPPED {file_name}: not found under {source}")
            continue

        entries = build_index_entries(file_chunks, pdf_path, image_dir)
        summary = summaries.get(file_name)
        print(f"Embedding {file_name} ({len(entries)} entries) ...")

        # The profile rides along in the same batch rather than costing its own
        # call, then is popped back off the end.
        texts = [entry.embed_text for entry in entries]
        if summary:
            texts.append(summary.summary)
        vectors = embedder.embed_texts(texts)
        summary_vector = vectors.pop() if summary else None
        for entry, vector in zip(entries, vectors):
            entry.embedding = vector

        upsert_document(
            conn,
            entries[0].doc_id,
            file_name,
            entries[0].file_type,
            entries,
            page_count=max(chunk.page_number for chunk in file_chunks),
            doc_summary=summary.summary if summary else None,
            summary_embedding=summary_vector,
            summary_embedding_model=settings.embedding_model if summary else None,
        )
        indexed_files += 1
        profiled = ", with a document profile" if summary else ""
        print(f"  upserted {len(entries)} entries under doc_id {entries[0].doc_id}{profiled}")

    conn.close()
    print(f"Indexed {len(search_index.chunks)} chunks from {indexed_files} files into Postgres")
    return 0


def ask(question: str, top_k: int, debug: bool = False, attach_context: bool | None = None) -> int:
    from utils.embedder import Embedder
    from utils.postgres_search_index import PostgresSearchIndex
    from utils.qa_agent import QAAgent

    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    if debug:
        # Only our own modules go to DEBUG. Leaving the root logger at WARNING
        # keeps anthropic/openai/httpx/httpcore quiet -- at DEBUG they dump full
        # request/response bodies (system prompt, conversation history, raw
        # headers) on every call, drowning out the query-level detail this
        # flag exists for.
        logging.getLogger("utils").setLevel(logging.DEBUG)

    embedder = Embedder()
    search_backend = PostgresSearchIndex(embedder=embedder, attach_context=attach_context)

    agent = QAAgent(search_backend)
    start = time.perf_counter()
    answer_text = agent.answer(question, top_k=top_k)
    elapsed = time.perf_counter() - start

    print(answer_text)
    if debug:
        print(
            f"main: {elapsed:.2f}s elapsed, "
            f"input_tokens={agent.total_input_tokens} output_tokens={agent.total_output_tokens} "
            f"embedding_tokens={embedder.total_tokens}",
            file=sys.stderr,
        )
    return 0


def main() -> None:
    # Windows consoles default to a narrow codepage (e.g. cp1252) that cannot
    # print every character a model response may contain; UTF-8 with
    # replacement avoids a crash on otherwise-successful output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="PDF knowledge base question answering.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Extract and index PDFs.")
    ingest_parser.add_argument(
        "--source",
        type=Path,
        default=Path("tests/fixtures"),
        help="A folder of PDFs, or a single .pdf file to run the pipeline on just that document.",
    )
    ingest_parser.add_argument("--output", type=Path, default=settings.index_path)
    ingest_parser.add_argument(
        "--enhance-metadata",
        action="store_true",
        help=(
            "Add an LLM-written summary to each table chunk, used as its dense-search "
            "text instead of truncated Markdown (one Claude call per table)."
        ),
    )
    ingest_parser.add_argument(
        "--extractor",
        choices=["fast", "vision"],
        default="fast",
        help=(
            "fast: pdf_inspector with a Docling fallback for scanned pages. "
            "vision: render every page and have Claude read it (one call per page)."
        ),
    )
    ingest_parser.add_argument(
        "--no-caption-figures",
        dest="caption_figures",
        action="store_false",
        default=True,
        help=(
            "Skip figure captioning and misclassified-text recovery (on by default, "
            "one Claude call per page with a plausible image; works with either extractor)."
        ),
    )
    ingest_parser.add_argument(
        "--summarise-documents",
        action="store_true",
        help=(
            "Write a profile of each whole document (one Claude call per file). "
            "Shown alongside search results so the model can tell whether it is "
            "in the right document; never citable."
        ),
    )
    ingest_parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/pages"),
        help="Where the vision extractor and figure captioner write rendered page images.",
    )

    index_parser = subparsers.add_parser(
        "index", help="Embed the extraction index and upsert it into Postgres."
    )
    index_parser.add_argument("--index", type=Path, default=settings.index_path)
    index_parser.add_argument(
        "--source",
        type=Path,
        default=Path("tests/fixtures"),
        help="The folder the indexed PDFs live in, or a single .pdf file's own path.",
    )
    index_parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/pages"),
        help="Where rendered page images live; recorded as asset_paths for figure chunks.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask the knowledge base a question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=settings.top_k)
    ask_parser.add_argument(
        "--debug",
        action="store_true",
        help="Log each search query, tool call, and LLM round trip to stderr.",
    )
    ask_parser.add_argument(
        "--attach-context",
        action="store_true",
        default=settings.attach_search_context,
        help=(
            "Attach one neighbouring chunk each side of every search match, for a "
            "completed sentence or table header at the cost of extra tokens per search "
            "(off unless ATTACH_SEARCH_CONTEXT=true in .env; this flag forces it on "
            "for one run)."
        ),
    )

    args = parser.parse_args()
    if args.command == "ingest":
        sys.exit(
            ingest(
                args.source,
                args.output,
                args.enhance_metadata,
                args.extractor,
                args.image_dir,
                args.caption_figures,
                args.summarise_documents,
            )
        )
    elif args.command == "index":
        sys.exit(index(args.index, args.source, args.image_dir))
    else:
        sys.exit(ask(args.question, args.top_k, args.debug, args.attach_context))


if __name__ == "__main__":
    main()
