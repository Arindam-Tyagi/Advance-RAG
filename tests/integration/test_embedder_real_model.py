"""Integration tests that load the real embedding model.

Not run by default (the first run downloads the model). Run them with:

    uv run pytest -m integration
"""

import math

import pytest

from rag_app.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_QUERY_INSTRUCTION
from rag_app.embeddings.embedder import (
    count_tokens,
    embed_documents,
    embed_query,
    get_token_limit,
    load_embedding_model,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests in this file (loading is slow)."""
    return load_embedding_model(DEFAULT_EMBEDDING_MODEL)


def dot_product(first: list, second: list) -> float:
    total = 0.0
    for index in range(len(first)):
        total = total + first[index] * second[index]
    return total


def test_model_facts_match_the_model_card(model):
    assert get_token_limit(model) == 512
    vector = embed_documents(model, ["hello"], batch_size=1)[0]
    assert len(vector) == 384


def test_vectors_are_normalized_to_length_one(model):
    vector = embed_documents(model, ["Dense retrieval uses embeddings."], batch_size=1)[0]

    length = math.sqrt(dot_product(vector, vector))

    assert abs(length - 1.0) < 1e-4


def test_related_passage_scores_higher_than_unrelated_one(model):
    passages = [
        "Dense passage retrieval encodes questions and passages with two BERT encoders.",
        "The pizza was baked in a wood-fired oven for ninety seconds.",
    ]
    passage_vectors = embed_documents(model, passages, batch_size=2)
    query_vector = embed_query(model, "How does dense passage retrieval work?", DEFAULT_QUERY_INSTRUCTION)

    related_score = dot_product(query_vector, passage_vectors[0])
    unrelated_score = dot_product(query_vector, passage_vectors[1])

    assert related_score > unrelated_score


def test_token_count_includes_start_and_end_tokens(model):
    # Two ordinary English words are two tokens, plus [CLS] and [SEP].
    assert count_tokens(model, "hello world") == 4


def test_unknown_model_name_gives_clear_error():
    with pytest.raises(RuntimeError, match="could not load embedding model"):
        load_embedding_model("this-org-does-not-exist/no-such-model-xyz")
