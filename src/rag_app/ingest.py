"""Ingest command: load documents, chunk them, and write the results to disk.

Run from the project root:

    uv run python -m rag_app.ingest

Reads every supported file in Settings.raw_data_dir and writes two files to
Settings.processed_data_dir:

    documents.jsonl               one cleaned Document per line
    chunks_<strategy>.jsonl       one Chunk per line, e.g. chunks_recursive.jsonl

JSONL means "one JSON object per line". It is easy to read in any editor,
easy to load line by line, and each strategy gets its own chunks file so
strategies can be compared side by side. Both files are generated output
and are git-ignored.
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_app.chunking.chunker import ChunkingError, chunk_documents
from rag_app.config import Settings, load_settings
from rag_app.ingestion.loaders import load_documents
from rag_app.models import Chunk

DOCUMENTS_FILE_NAME = "documents.jsonl"


@dataclass
class IngestionResult:
    """What one ingestion run produced.

    Attributes:
        documents: Loaded Document objects.
        skipped: SkippedFile objects for files that could not be loaded.
        chunks: All Chunk objects from all documents.
        documents_path: Where the documents file was written.
        chunks_path: Where the chunks file was written.
    """

    documents: list
    skipped: list
    chunks: list
    documents_path: Path
    chunks_path: Path


def write_jsonl(items: list, output_path: Path) -> None:
    """Write dataclass objects to a JSONL file, one object per line.

    ensure_ascii=False keeps characters like "é" readable instead of
    turning them into codes like "\\u00e9".
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for item in items:
            line = json.dumps(asdict(item), ensure_ascii=False)
            file.write(line + "\n")


def chunks_file_path(settings: Settings) -> Path:
    """Return where the chunks for the configured strategy are stored."""
    return settings.processed_data_dir / f"chunks_{settings.chunk_strategy}.jsonl"


def read_chunks(path: Path) -> list:
    """Read a chunks JSONL file (written by this command) back into Chunk objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    chunks = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    metadata=row["metadata"],
                )
            )
    return chunks


def compute_chunk_stats(chunks: list) -> dict:
    """Return simple length statistics for a list of chunks.

    Returns:
        A dict with chunk_count and the min, average and max chunk length
        in characters. All values are 0 when there are no chunks.
    """
    if len(chunks) == 0:
        return {"chunk_count": 0, "min_chars": 0, "avg_chars": 0, "max_chars": 0}

    lengths = []
    for chunk in chunks:
        lengths.append(len(chunk.text))
    return {
        "chunk_count": len(chunks),
        "min_chars": min(lengths),
        "avg_chars": round(sum(lengths) / len(lengths)),
        "max_chars": max(lengths),
    }


def run_ingestion(settings: Settings) -> IngestionResult:
    """Load, chunk and save every document in the raw data folder.

    Raises:
        FileNotFoundError: If the raw data folder does not exist.
        ChunkingError: If any chunk fails a safety check.
    """
    load_result = load_documents(settings.raw_data_dir)
    chunks = chunk_documents(load_result.documents, settings)

    documents_path = settings.processed_data_dir / DOCUMENTS_FILE_NAME
    chunks_path = chunks_file_path(settings)
    write_jsonl(load_result.documents, documents_path)
    write_jsonl(chunks, chunks_path)

    return IngestionResult(
        documents=load_result.documents,
        skipped=load_result.skipped,
        chunks=chunks,
        documents_path=documents_path,
        chunks_path=chunks_path,
    )


def print_report(result: IngestionResult, settings: Settings) -> None:
    """Print a human-readable summary of an ingestion run."""
    print(f"Raw data folder:  {settings.raw_data_dir}")
    print(
        f"Chunking:         strategy={settings.chunk_strategy}, "
        f"size={settings.chunk_size}, overlap={settings.chunk_overlap}"
    )
    print(f"Documents loaded: {len(result.documents)}")
    print(f"Files skipped:    {len(result.skipped)}")
    for skipped in result.skipped:
        print(f"  - {skipped.path}: {skipped.reason}")

    print()
    print("Per document:")
    for document in result.documents:
        document_chunk_count = 0
        for chunk in result.chunks:
            if chunk.doc_id == document.doc_id:
                document_chunk_count = document_chunk_count + 1
        pages = document.metadata.get("page_count", "-")
        empty_pages = document.metadata.get("empty_page_count", "-")
        print(
            f"  {document.doc_id}: {document.metadata['char_count']} chars, "
            f"pages={pages}, empty_pages={empty_pages}, chunks={document_chunk_count}"
        )

    stats = compute_chunk_stats(result.chunks)
    print()
    print(
        f"Chunks: {stats['chunk_count']} total, length min={stats['min_chars']}, "
        f"avg={stats['avg_chars']}, max={stats['max_chars']} characters"
    )
    print(f"Wrote {result.documents_path}")
    print(f"Wrote {result.chunks_path}")


def main() -> int:
    """Entry point for `python -m rag_app.ingest`.

    Returns:
        0 on success, 1 if settings are invalid, the data folder is missing,
        chunking fails, or no documents could be loaded.
    """
    # The Windows console often uses an old encoding (cp1252) that cannot show
    # characters such as "ﬁ". Instead of crashing on such a character,
    # print it as an escape code like ﬁ. Files on disk are always UTF-8.
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")

    try:
        settings = load_settings()
        result = run_ingestion(settings)
    except (ValueError, FileNotFoundError, NotADirectoryError, ChunkingError) as error:
        # ChunkingError and settings problems are reported, not hidden.
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1

    print_report(result, settings)
    if len(result.documents) == 0:
        print("No documents were loaded.", file=sys.stderr)
        return 1
    return 0


# This block runs only when the file is executed as a program
# (python -m rag_app.ingest), not when it is imported by other code or tests.
if __name__ == "__main__":
    sys.exit(main())
