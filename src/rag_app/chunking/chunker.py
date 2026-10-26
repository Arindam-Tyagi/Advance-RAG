"""Split documents into chunks using LangChain text splitters.

LangChain does the actual splitting. This module wraps it so that:
    - the rest of the app only ever sees our own Chunk objects, not
      LangChain objects (we could swap LangChain out without changing callers);
    - every chunk's start_char/end_char is checked against the document text,
      so a wrong position can never silently reach citations;
    - PDF chunks are tagged with the page numbers they come from.

Strategies (chosen with Settings.chunk_strategy):
    "fixed":     cut every chunk_size characters, ignoring words and
                 sentences. The simplest possible baseline.
    "recursive": try to cut at paragraph breaks first, then line breaks,
                 then spaces, and only cut inside a word as a last resort.
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from rag_app.config import Settings
from rag_app.models import Chunk, Document


class ChunkingError(Exception):
    """Raised when a chunk fails a safety check (wrong position or too long)."""


def create_splitter(settings: Settings) -> TextSplitter:
    """Create the LangChain splitter for the configured strategy.

    Lengths are measured in characters (LangChain's default, Python's len).
    add_start_index=True asks LangChain to record where each chunk starts.

    Raises:
        ValueError: If settings.chunk_strategy is not a known strategy.
    """
    if settings.chunk_strategy == "fixed":
        # separator="" means "split between any two characters", so chunks
        # are cut at exactly chunk_size characters, even mid-word.
        return CharacterTextSplitter(
            separator="",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )
    if settings.chunk_strategy == "recursive":
        # Default separators, tried in order: "\n\n" (paragraph),
        # "\n" (line), " " (word), "" (any character).
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )
    raise ValueError(f"unknown chunk strategy '{settings.chunk_strategy}'")


def find_page_number(page_start_chars: list, char_index: int) -> int:
    """Return the 1-based page number that contains a character position.

    page_start_chars comes from the PDF loader: entry 0 is where page 1
    starts, entry 1 where page 2 starts, and so on. The answer is the last
    page whose start is at or before char_index. If an empty page has the
    same start as the next page, the next (non-empty) page is chosen.
    """
    page_number = 1
    for index in range(len(page_start_chars)):
        if page_start_chars[index] <= char_index:
            page_number = index + 1
    return page_number


def chunk_document(document: Document, splitter: TextSplitter, settings: Settings) -> list:
    """Split one document into a list of Chunk objects.

    Args:
        document: The cleaned document to split.
        splitter: A splitter made by create_splitter.
        settings: Used to record and check the chunking parameters.

    Returns:
        Chunks in document order. chunk_id is "<doc_id>::<position>",
        e.g. "papers/rag.pdf::0", "papers/rag.pdf::1", ...

    Raises:
        ChunkingError: If a chunk's recorded position does not match the
            document text, positions go backwards, or a chunk is longer
            than chunk_size.
    """
    langchain_chunks = splitter.create_documents([document.text])
    page_start_chars = document.metadata.get("page_start_chars")

    chunks = []
    previous_start = -1
    for position, langchain_chunk in enumerate(langchain_chunks):
        text = langchain_chunk.page_content
        start = langchain_chunk.metadata["start_index"]
        end = start + len(text)

        # Safety checks: never trust a position we have not verified.
        if start < 0 or document.text[start:end] != text:
            raise ChunkingError(
                f"{document.doc_id}: chunk {position} does not match the document "
                f"text at position {start}"
            )
        if start <= previous_start:
            raise ChunkingError(
                f"{document.doc_id}: chunk {position} starts at {start}, which is not "
                f"after the previous chunk's start {previous_start}"
            )
        if len(text) > settings.chunk_size:
            raise ChunkingError(
                f"{document.doc_id}: chunk {position} has {len(text)} characters, "
                f"more than chunk_size {settings.chunk_size}"
            )
        previous_start = start

        metadata = {
            "chunk_index": position,
            "chunk_strategy": settings.chunk_strategy,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "file_name": document.metadata.get("file_name"),
        }
        if page_start_chars is not None:
            metadata["page_start"] = find_page_number(page_start_chars, start)
            # end is exclusive, so the chunk's last character is at end - 1.
            metadata["page_end"] = find_page_number(page_start_chars, end - 1)

        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::{position}",
                doc_id=document.doc_id,
                text=text,
                start_char=start,
                end_char=end,
                metadata=metadata,
            )
        )
    return chunks


def chunk_documents(documents: list, settings: Settings) -> list:
    """Split many documents, creating the splitter only once.

    Returns:
        All chunks from all documents, in document order.
    """
    splitter = create_splitter(settings)
    all_chunks = []
    for document in documents:
        document_chunks = chunk_document(document, splitter, settings)
        all_chunks.extend(document_chunks)
    return all_chunks
