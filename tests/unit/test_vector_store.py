"""Tests for storing chunks in ChromaDB (uses tiny hand-made vectors, no model)."""

import pytest

from rag_app.models import Chunk
from rag_app.retrieval.vector_store import (
    IndexNotFoundError,
    build_index,
    chunk_from_stored,
    chunk_to_metadata,
    make_collection_name,
    open_chroma_client,
    open_collection,
)


def make_chunk(chunk_id: str, text: str, metadata=None) -> Chunk:
    if metadata is None:
        metadata = {"chunk_index": 0, "chunk_strategy": "recursive", "file_name": "a.txt"}
    return Chunk(chunk_id=chunk_id, doc_id="a.txt", text=text, start_char=0, end_char=len(text), metadata=metadata)


CHUNKS = [
    make_chunk("a.txt::0", "about retrieval"),
    make_chunk("a.txt::1", "about cooking"),
]
# Two-dimensional vectors chosen by hand; real ones have 384 numbers.
VECTORS = [[1.0, 0.0], [0.0, 1.0]]
INDEX_METADATA = {"embedding_model": "fake-model", "chunk_count": 2}


# --- names and metadata ---------------------------------------------------------


def test_collection_name_is_lowercase_and_safe():
    name = make_collection_name("recursive", "BAAI/bge-small-en-v1.5")

    assert name == "recursive__baai-bge-small-en-v1.5"


def test_metadata_leaves_out_missing_values():
    chunk = make_chunk("a.txt::0", "text", metadata={"chunk_index": 0, "file_name": None})

    metadata = chunk_to_metadata(chunk)

    assert "file_name" not in metadata
    assert "page_start" not in metadata
    assert metadata["doc_id"] == "a.txt"
    assert metadata["start_char"] == 0
    assert metadata["end_char"] == 4


def test_metadata_keeps_pdf_pages():
    chunk = make_chunk("p.pdf::3", "text", metadata={"chunk_index": 3, "page_start": 2, "page_end": 3})

    metadata = chunk_to_metadata(chunk)

    assert metadata["page_start"] == 2
    assert metadata["page_end"] == 3


# --- building the index ------------------------------------------------------------


def test_index_stores_chunks_vectors_and_metadata(tmp_path):
    client = open_chroma_client(tmp_path / "chroma")

    collection = build_index(client, "test_collection", CHUNKS, VECTORS, INDEX_METADATA)

    assert collection.count() == 2
    stored = collection.get(ids=["a.txt::1"], include=["documents", "metadatas"])
    assert stored["documents"] == ["about cooking"]
    assert stored["metadatas"][0]["doc_id"] == "a.txt"
    assert collection.metadata["embedding_model"] == "fake-model"


# Chroma 1.5 calls "no embedding function" a legacy setting and warns when the
# configuration is read. Choosing none is deliberate here (we supply vectors).
@pytest.mark.filterwarnings("ignore:legacy embedding function config")
def test_index_uses_cosine_and_no_chroma_embedder(tmp_path):
    client = open_chroma_client(tmp_path / "chroma")

    collection = build_index(client, "test_collection", CHUNKS, VECTORS, INDEX_METADATA)

    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert collection.configuration["embedding_function"] is None


def test_nearest_chunk_is_found_and_distance_is_one_minus_cosine(tmp_path):
    client = open_chroma_client(tmp_path / "chroma")
    collection = build_index(client, "test_collection", CHUNKS, VECTORS, INDEX_METADATA)

    # Cosine similarity with [1, 0] is 0.8 for chunk 0 and 0.6 for chunk 1.
    result = collection.query(query_embeddings=[[0.8, 0.6]], n_results=2, include=["distances"])

    assert result["ids"][0] == ["a.txt::0", "a.txt::1"]
    assert result["distances"][0][0] == pytest.approx(0.2, abs=1e-5)
    assert result["distances"][0][1] == pytest.approx(0.4, abs=1e-5)


def test_rebuilding_replaces_the_old_collection(tmp_path):
    client = open_chroma_client(tmp_path / "chroma")
    build_index(client, "test_collection", CHUNKS, VECTORS, INDEX_METADATA)

    collection = build_index(client, "test_collection", CHUNKS[:1], VECTORS[:1], INDEX_METADATA)

    assert collection.count() == 1


def test_index_survives_reopening_the_database(tmp_path):
    build_index(open_chroma_client(tmp_path / "chroma"), "test_collection", CHUNKS, VECTORS, INDEX_METADATA)

    reopened = open_chroma_client(tmp_path / "chroma").get_collection("test_collection")

    assert reopened.count() == 2


def test_chunk_and_vector_counts_must_match(tmp_path):
    client = open_chroma_client(tmp_path / "chroma")

    with pytest.raises(ValueError, match="2 chunks but 1 vectors"):
        build_index(client, "test_collection", CHUNKS, VECTORS[:1], INDEX_METADATA)


def test_empty_index_is_rejected(tmp_path):
    client = open_chroma_client(tmp_path / "chroma")

    with pytest.raises(ValueError, match="no chunks"):
        build_index(client, "test_collection", [], [], INDEX_METADATA)


# --- reading back --------------------------------------------------------------------


def test_stored_chunk_is_rebuilt_exactly():
    chunk = make_chunk("p.pdf::3", "text", metadata={"chunk_index": 3, "page_start": 2, "page_end": 3})

    rebuilt = chunk_from_stored(chunk.chunk_id, chunk.text, chunk_to_metadata(chunk))

    assert rebuilt == chunk


def test_opening_a_missing_collection_gives_clear_error(tmp_path):
    client = open_chroma_client(tmp_path / "chroma")

    with pytest.raises(IndexNotFoundError, match="not found"):
        open_collection(client, "no_such_collection")
