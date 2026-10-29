"""Store chunk vectors in ChromaDB, a local vector database.

This is the only module that talks to ChromaDB, so the rest of the app never
depends on Chroma's types and the vector store could be swapped later.

Design decisions (Milestone 2):
    - Local and persistent: Chroma saves its files in Settings.index_dir.
      No server, no account. The folder is generated and git-ignored.
    - We supply the vectors ourselves. Collections are created with
      embedding_function=None, so Chroma can never silently embed text
      with a different model; text queries are refused.
    - Cosine distance. Chroma returns distance = 1 - cosine similarity.
    - One collection per (chunking strategy, embedding model), e.g.
      "recursive__baai-bge-small-en-v1.5", so setups never mix.
    - Anonymous usage telemetry is switched off.
"""

import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from rag_app.models import Chunk

# Characters Chroma allows in collection names; everything else becomes "-".
NOT_ALLOWED_IN_NAME = re.compile(r"[^a-zA-Z0-9._-]")

# Metadata keys that hold Chunk fields rather than chunk.metadata entries.
CHUNK_FIELD_KEYS = ["doc_id", "start_char", "end_char"]


class IndexNotFoundError(Exception):
    """Raised when the requested collection has not been built yet."""


def open_chroma_client(index_dir) -> chromadb.ClientAPI:
    """Open (or create) the local ChromaDB database in index_dir.

    anonymized_telemetry=False stops Chroma from sending usage statistics
    to its analytics service.
    """
    return chromadb.PersistentClient(
        path=str(index_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def make_collection_name(chunk_strategy: str, embedding_model: str) -> str:
    """Build a collection name from the chunking strategy and model name.

    Example: ("recursive", "BAAI/bge-small-en-v1.5")
             -> "recursive__baai-bge-small-en-v1.5"
    """
    name = f"{chunk_strategy}__{embedding_model}".lower()
    return NOT_ALLOWED_IN_NAME.sub("-", name)


def chunk_to_metadata(chunk) -> dict:
    """Pick the chunk fields stored next to its vector in Chroma.

    Chroma only accepts strings, numbers and booleans as metadata values and
    rejects None, so missing values are left out instead of stored as None.
    Positions (start_char, end_char) are stored so retrieved chunks can be
    matched against evaluation evidence and turned into citations.
    """
    candidates = {
        "doc_id": chunk.doc_id,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "chunk_index": chunk.metadata.get("chunk_index"),
        "chunk_strategy": chunk.metadata.get("chunk_strategy"),
        "file_name": chunk.metadata.get("file_name"),
        "page_start": chunk.metadata.get("page_start"),
        "page_end": chunk.metadata.get("page_end"),
    }
    metadata = {}
    for key, value in candidates.items():
        if value is not None:
            metadata[key] = value
    return metadata


def build_index(
    client: chromadb.ClientAPI,
    collection_name: str,
    chunks: list,
    vectors: list,
    collection_metadata: dict,
):
    """Create a collection holding the given chunks and their vectors.

    If a collection with this name already exists it is deleted first, so
    re-running always gives a clean index (no duplicate or stale chunks).

    Args:
        client: An open Chroma client.
        collection_name: Name from make_collection_name.
        chunks: Chunk objects to store.
        vectors: One embedding vector per chunk, in the same order.
        collection_metadata: Facts about the whole index, e.g. which
            embedding model made the vectors. Values must not be None.

    Returns:
        The new Chroma collection.

    Raises:
        ValueError: If the number of chunks and vectors differ, or there
            are no chunks.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
    if len(chunks) == 0:
        raise ValueError("cannot build an index with no chunks")

    existing_names = []
    for collection in client.list_collections():
        existing_names.append(collection.name)
    if collection_name in existing_names:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata=collection_metadata,
        embedding_function=None,
    )

    # Chroma limits how many items one add() call may contain.
    batch_size = client.get_max_batch_size()
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = batch_start + batch_size
        batch_chunks = chunks[batch_start:batch_end]

        ids = []
        documents = []
        metadatas = []
        for chunk in batch_chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(chunk_to_metadata(chunk))

        collection.add(
            ids=ids,
            embeddings=vectors[batch_start:batch_end],
            documents=documents,
            metadatas=metadatas,
        )
    return collection


def open_collection(client: chromadb.ClientAPI, collection_name: str):
    """Open an existing collection.

    Raises:
        IndexNotFoundError: If no collection with this name exists.
    """
    try:
        return client.get_collection(collection_name)
    except NotFoundError:
        raise IndexNotFoundError(
            f"index '{collection_name}' not found. Run the build_index command first."
        ) from None


def chunk_from_stored(chunk_id: str, text: str, stored_metadata: dict) -> Chunk:
    """Rebuild a Chunk from what Chroma stored (the reverse of chunk_to_metadata)."""
    chunk_metadata = {}
    for key, value in stored_metadata.items():
        if key not in CHUNK_FIELD_KEYS:
            chunk_metadata[key] = value
    return Chunk(
        chunk_id=chunk_id,
        doc_id=stored_metadata["doc_id"],
        text=text,
        start_char=stored_metadata["start_char"],
        end_char=stored_metadata["end_char"],
        metadata=chunk_metadata,
    )


def query_collection(collection, query_vector: list, k: int) -> list:
    """Find the k stored chunks closest to a query vector.

    Chroma returns results for a LIST of queries; we send one query, so we
    read the first entry ([0]) of each result list.

    Returns:
        (Chunk, distance) pairs, closest first. Fewer than k pairs if the
        collection holds fewer than k chunks. distance = 1 - cosine similarity.
    """
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    matches = []
    for index in range(len(ids)):
        chunk = chunk_from_stored(ids[index], documents[index], metadatas[index])
        matches.append((chunk, distances[index]))
    return matches
