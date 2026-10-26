"""Tests for splitting documents into chunks."""

from pathlib import Path

import pytest
from langchain_core.documents import Document as LangChainDocument

from rag_app.chunking.chunker import (
    ChunkingError,
    chunk_document,
    chunk_documents,
    create_splitter,
    find_page_number,
)
from rag_app.config import Settings
from rag_app.models import Document

LONG_TEXT = (
    "Retrieval-Augmented Generation finds relevant passages before answering.\n\n"
    "Chunking decides how documents are cut into pieces. Small chunks are precise "
    "but can lose context. Large chunks keep context but can bury the answer.\n\n"
    "Overlap repeats a little text between neighbouring chunks, so a sentence "
    "that is cut at a boundary still appears whole in at least one chunk.\n"
    "Evaluation tells us which settings actually work best."
)


def make_settings(strategy: str, chunk_size: int, chunk_overlap: int) -> Settings:
    """Build Settings for a test without reading environment variables."""
    return Settings(
        raw_data_dir=Path("unused"),
        processed_data_dir=Path("unused"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=strategy,
    )


def split(text: str, strategy: str, chunk_size: int, chunk_overlap: int, metadata=None) -> list:
    """Chunk a single piece of text and return the chunks."""
    if metadata is None:
        metadata = {"file_name": "test.txt"}
    document = Document(doc_id="test.txt", source_path="test.txt", text=text, metadata=metadata)
    settings = make_settings(strategy, chunk_size, chunk_overlap)
    return chunk_document(document, create_splitter(settings), settings)


# --- rules every strategy must follow ---------------------------------------


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_chunk_positions_slice_back_to_chunk_text(strategy):
    chunks = split(LONG_TEXT, strategy, chunk_size=80, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert LONG_TEXT[chunk.start_char:chunk.end_char] == chunk.text


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_no_chunk_is_longer_than_chunk_size(strategy):
    chunks = split(LONG_TEXT, strategy, chunk_size=80, chunk_overlap=20)

    for chunk in chunks:
        assert len(chunk.text) <= 80


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_every_visible_character_is_in_some_chunk(strategy):
    # Nothing may be lost: each non-whitespace character must be covered.
    chunks = split(LONG_TEXT, strategy, chunk_size=80, chunk_overlap=20)

    for index in range(len(LONG_TEXT)):
        if LONG_TEXT[index].isspace():
            continue
        covered = False
        for chunk in chunks:
            if chunk.start_char <= index < chunk.end_char:
                covered = True
        assert covered, f"character {index} is in no chunk"


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_short_document_becomes_one_chunk(strategy):
    chunks = split("A short note.", strategy, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "A short note."
    assert chunks[0].start_char == 0


# --- differences between the strategies -------------------------------------


def test_fixed_strategy_cuts_at_exact_size_even_mid_word():
    chunks = split("abcdefghij", "fixed", chunk_size=4, chunk_overlap=0)

    texts = []
    for chunk in chunks:
        texts.append(chunk.text)
    assert texts == ["abcd", "efgh", "ij"]


def test_fixed_strategy_overlaps_neighbouring_chunks():
    chunks = split("abcdefghij", "fixed", chunk_size=4, chunk_overlap=2)

    texts = []
    for chunk in chunks:
        texts.append(chunk.text)
    assert texts == ["abcd", "cdef", "efgh", "ghij"]


def test_recursive_strategy_keeps_paragraphs_whole():
    text = "First paragraph here.\n\nSecond paragraph here."

    chunks = split(text, "recursive", chunk_size=30, chunk_overlap=0)

    texts = []
    for chunk in chunks:
        texts.append(chunk.text)
    assert texts == ["First paragraph here.", "Second paragraph here."]


# --- IDs and metadata ---------------------------------------------------------


def test_chunk_ids_and_metadata():
    chunks = split(LONG_TEXT, "recursive", chunk_size=80, chunk_overlap=20)

    assert chunks[0].chunk_id == "test.txt::0"
    assert chunks[1].chunk_id == "test.txt::1"
    assert chunks[1].doc_id == "test.txt"
    assert chunks[1].metadata["chunk_index"] == 1
    assert chunks[1].metadata["chunk_strategy"] == "recursive"
    assert chunks[1].metadata["chunk_size"] == 80
    assert chunks[1].metadata["chunk_overlap"] == 20
    assert chunks[1].metadata["file_name"] == "test.txt"
    assert "page_start" not in chunks[1].metadata


def test_pdf_chunks_record_their_pages():
    # Three "pages": page 1 starts at 0, page 2 at 14, page 3 at 27.
    text = "Page one AA.\n\nPage two B.\n\nPage three."
    page_starts = [0, 14, 27]
    metadata = {"file_name": "paper.pdf", "page_start_chars": page_starts}

    chunks = split(text, "fixed", chunk_size=20, chunk_overlap=0, metadata=metadata)

    # First chunk "Page one AA.\n\nPage t" spans pages 1-2.
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["page_end"] == 2
    assert chunks[-1].metadata["page_end"] == 3


def test_find_page_number():
    page_starts = [0, 100, 250]

    assert find_page_number(page_starts, 0) == 1
    assert find_page_number(page_starts, 99) == 1
    assert find_page_number(page_starts, 100) == 2
    assert find_page_number(page_starts, 400) == 3


def test_find_page_number_skips_empty_page():
    # Page 2 is empty, so it has the same start as page 3.
    page_starts = [0, 11, 11]

    assert find_page_number(page_starts, 11) == 3


# --- safety checks and errors --------------------------------------------------


class SplitterWithWrongPosition:
    """A fake splitter that reports a wrong start position."""

    def create_documents(self, texts):
        return [LangChainDocument(page_content="world", metadata={"start_index": 0})]


def test_wrong_position_from_splitter_raises_error():
    document = Document(doc_id="d", source_path="d.txt", text="hello world")
    settings = make_settings("recursive", chunk_size=100, chunk_overlap=0)

    with pytest.raises(ChunkingError, match="does not match the document text"):
        chunk_document(document, SplitterWithWrongPosition(), settings)


def test_unknown_strategy_raises_error():
    settings = make_settings("semantic", chunk_size=100, chunk_overlap=0)

    with pytest.raises(ValueError, match="unknown chunk strategy 'semantic'"):
        create_splitter(settings)


def test_chunk_documents_keeps_document_order():
    first = Document(doc_id="a.txt", source_path="a.txt", text="Alpha text.")
    second = Document(doc_id="b.txt", source_path="b.txt", text="Beta text.")
    settings = make_settings("recursive", chunk_size=100, chunk_overlap=0)

    chunks = chunk_documents([first, second], settings)

    ids = []
    for chunk in chunks:
        ids.append(chunk.chunk_id)
    assert ids == ["a.txt::0", "b.txt::0"]
