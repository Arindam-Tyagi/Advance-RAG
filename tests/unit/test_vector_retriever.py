"""Tests for the vector retriever, with a fake model and a temporary index."""

from pathlib import Path

import numpy as np
import pytest

from rag_app.config import Settings
from rag_app.models import Chunk
from rag_app.retrieval.vector_retriever import IndexMismatchError, open_vector_retriever
from rag_app.retrieval.vector_store import (
    IndexNotFoundError,
    build_index,
    make_collection_name,
    open_chroma_client,
)


class FakeModel:
    """Texts mentioning "retrieval" point one way, all others point the other way."""

    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts, **options):
        vectors = []
        for text in texts:
            self.encoded_texts.append(text)
            if "retrieval" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors)


CHUNKS = [
    Chunk("paper.pdf::0", "paper.pdf", "dense retrieval basics", 0, 22,
          {"chunk_index": 0, "chunk_strategy": "recursive", "page_start": 1, "page_end": 2}),
    Chunk("paper.pdf::1", "paper.pdf", "baking bread at home", 20, 40,
          {"chunk_index": 1, "chunk_strategy": "recursive", "page_start": 2, "page_end": 2}),
]
# Vectors chosen to match what FakeModel would produce for these texts,
# except the second one leans slightly towards "retrieval".
VECTORS = [[1.0, 0.0], [0.6, 0.8]]


def make_settings(index_dir: Path, model_name: str = "fake/model") -> Settings:
    return Settings(
        raw_data_dir=Path("unused"),
        processed_data_dir=Path("unused"),
        chunk_size=100,
        chunk_overlap=0,
        chunk_strategy="recursive",
        embedding_model=model_name,
        query_instruction="Find: ",
        index_dir=index_dir,
    )


def build_test_index(index_dir: Path, model_name_in_index: str = "fake/model") -> None:
    client = open_chroma_client(index_dir)
    collection_name = make_collection_name("recursive", model_name_in_index)
    build_index(client, collection_name, CHUNKS, VECTORS, {"embedding_model": model_name_in_index})


# --- searching ------------------------------------------------------------------


def test_results_are_ranked_best_first_with_similarity_scores(tmp_path):
    build_test_index(tmp_path)
    retriever = open_vector_retriever(make_settings(tmp_path), FakeModel())

    results = retriever.retrieve("what is retrieval?", k=2)

    assert [result.chunk.chunk_id for result in results] == ["paper.pdf::0", "paper.pdf::1"]
    assert [result.rank for result in results] == [1, 2]
    # Query vector is [1, 0]: cosine similarity 1.0 with chunk 0, 0.6 with chunk 1.
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].score == pytest.approx(0.6, abs=1e-5)


def test_retrieved_chunk_keeps_position_and_metadata(tmp_path):
    build_test_index(tmp_path)
    retriever = open_vector_retriever(make_settings(tmp_path), FakeModel())

    chunk = retriever.retrieve("retrieval", k=1)[0].chunk

    assert chunk == CHUNKS[0]


def test_query_instruction_is_put_in_front_of_the_question(tmp_path):
    build_test_index(tmp_path)
    model = FakeModel()
    retriever = open_vector_retriever(make_settings(tmp_path), model)

    retriever.retrieve("what is retrieval?", k=1)

    assert model.encoded_texts == ["Find: what is retrieval?"]


def test_asking_for_more_results_than_chunks_returns_all_chunks(tmp_path):
    build_test_index(tmp_path)
    retriever = open_vector_retriever(make_settings(tmp_path), FakeModel())

    assert len(retriever.retrieve("retrieval", k=10)) == 2


def test_empty_query_and_bad_k_are_rejected(tmp_path):
    build_test_index(tmp_path)
    retriever = open_vector_retriever(make_settings(tmp_path), FakeModel())

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.retrieve("   ", k=3)
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.retrieve("retrieval", k=0)


# --- opening the index safely ------------------------------------------------------


def test_missing_index_gives_clear_error(tmp_path):
    with pytest.raises(IndexNotFoundError, match="Run the build_index command first"):
        open_vector_retriever(make_settings(tmp_path), FakeModel())


def test_index_built_with_another_model_is_refused(tmp_path):
    # "BAAI/x" and "baai/x" map to the same collection name, but they are
    # different model names, so the stored name check must catch it.
    build_test_index(tmp_path, model_name_in_index="BAAI/x")

    with pytest.raises(IndexMismatchError, match="was built with 'BAAI/x'"):
        open_vector_retriever(make_settings(tmp_path, model_name="baai/x"), FakeModel())
