"""Vector retrieval: find the chunks whose meaning is closest to a question.

How a search works:
    1. Embed the question with the SAME model that embedded the chunks
       (plus the query instruction).
    2. Ask the vector store for the k closest chunk vectors.
    3. Return them as RetrievedChunk objects, best first, with a
       similarity score between -1 and 1 (higher = more similar).

The retriever must never search an index built with a different embedding
model. Both bge-small and all-MiniLM produce 384-number vectors, so a
mismatch would raise no error at all; it would just return meaningless
results. Two guards prevent that: the collection name contains the model
name, and the exact model name stored inside the index is checked too
(two names can map to the same collection name, e.g. "BAAI/x" and "baai/x").

Every retriever in this project (vector now, BM25 and hybrid later) offers
the same method, retrieve(query, k) -> list of RetrievedChunk, so they can
be swapped without changing the code that calls them.
"""

from dataclasses import dataclass

from rag_app.config import Settings
from rag_app.embeddings.embedder import embed_query
from rag_app.models import Chunk
from rag_app.retrieval.vector_store import (
    make_collection_name,
    open_chroma_client,
    open_collection,
    query_collection,
)


class IndexMismatchError(Exception):
    """Raised when an index was built with a different embedding model."""


@dataclass
class RetrievedChunk:
    """One search result.

    Attributes:
        chunk: The retrieved chunk, including its position in the document.
        score: Cosine similarity between question and chunk (higher is
            better). Scores are only comparable within one model and index.
        rank: Position in the results, starting at 1 for the best match.
    """

    chunk: Chunk
    score: float
    rank: int


class VectorRetriever:
    """Searches one Chroma collection with one embedding model."""

    def __init__(self, model, collection, query_instruction: str):
        """Store what a search needs. Use open_vector_retriever to create one."""
        self.model = model
        self.collection = collection
        self.query_instruction = query_instruction

    def retrieve(self, query: str, k: int) -> list:
        """Return the k chunks most similar to the query, best first.

        Raises:
            ValueError: If the query is empty or k is less than 1.
        """
        if query.strip() == "":
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

        query_vector = embed_query(self.model, query, self.query_instruction)
        matches = query_collection(self.collection, query_vector, k)

        results = []
        rank = 1
        for chunk, distance in matches:
            # Chroma reports cosine DISTANCE; similarity is 1 - distance.
            results.append(RetrievedChunk(chunk=chunk, score=1.0 - distance, rank=rank))
            rank = rank + 1
        return results


def open_vector_retriever(settings: Settings, model) -> VectorRetriever:
    """Open the index for the configured chunk strategy and embedding model.

    Args:
        settings: Provides index_dir, chunk_strategy, embedding_model and
            query_instruction.
        model: The loaded embedding model. Passed in so that callers
            running many searches load it only once.

    Raises:
        IndexNotFoundError: If the index has not been built.
        IndexMismatchError: If the index was built with another model.
    """
    client = open_chroma_client(settings.index_dir)
    collection_name = make_collection_name(settings.chunk_strategy, settings.embedding_model)
    collection = open_collection(client, collection_name)

    index_model = collection.metadata.get("embedding_model")
    if index_model != settings.embedding_model:
        raise IndexMismatchError(
            f"index '{collection_name}' was built with '{index_model}', "
            f"but the configured model is '{settings.embedding_model}'. Rebuild the index."
        )

    return VectorRetriever(model, collection, settings.query_instruction)
