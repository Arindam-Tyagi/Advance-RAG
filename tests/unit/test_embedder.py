"""Unit tests for the embedder, using a fake model (fast, no download)."""

import numpy as np

from rag_app.embeddings.embedder import (
    count_tokens,
    embed_documents,
    embed_query,
    find_chunks_over_token_limit,
    get_token_limit,
)
from rag_app.models import Chunk


class FakeTokenizer:
    """Pretends each word is one token, plus a start and an end token."""

    def __call__(self, text, add_special_tokens=True, truncation=False):
        token_ids = []
        for _ in text.split():
            token_ids.append(1)
        if add_special_tokens:
            token_ids = [101] + token_ids + [102]
        return {"input_ids": token_ids}


class FakeModel:
    """Stands in for SentenceTransformer and records how it was called."""

    def __init__(self):
        self.max_seq_length = 5
        self.tokenizer = FakeTokenizer()
        self.encode_calls = []

    def encode(self, texts, **options):
        self.encode_calls.append({"texts": texts, "options": options})
        vectors = []
        for text in texts:
            vectors.append([float(len(text)), 0.0])
        return np.array(vectors)


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", text=text, start_char=0, end_char=len(text))


def test_token_limit_comes_from_the_model():
    assert get_token_limit(FakeModel()) == 5


def test_token_count_includes_special_tokens():
    # 3 words + start token + end token
    assert count_tokens(FakeModel(), "three short words") == 5


def test_only_chunks_over_the_limit_are_reported():
    chunks = [
        make_chunk("c0", "fits in limit"),  # 3 words + 2 = 5 tokens: fits
        make_chunk("c1", "this one is too long"),  # 5 words + 2 = 7 tokens
    ]

    over_limit = find_chunks_over_token_limit(FakeModel(), chunks)

    assert len(over_limit) == 1
    assert over_limit[0].chunk_id == "c1"
    assert over_limit[0].token_count == 7


def test_documents_are_embedded_in_batches_and_normalized():
    model = FakeModel()

    vectors = embed_documents(model, ["a", "bb"], batch_size=16)

    assert vectors == [[1.0, 0.0], [2.0, 0.0]]
    assert isinstance(vectors[0], list)
    options = model.encode_calls[0]["options"]
    assert options["batch_size"] == 16
    assert options["normalize_embeddings"] is True


def test_documents_get_no_instruction():
    model = FakeModel()

    embed_documents(model, ["chunk text"], batch_size=8)

    assert model.encode_calls[0]["texts"] == ["chunk text"]


def test_query_gets_the_instruction_in_front():
    model = FakeModel()

    embed_query(model, "What is RAG?", instruction="Find passages: ")

    assert model.encode_calls[0]["texts"] == ["Find passages: What is RAG?"]
    assert model.encode_calls[0]["options"]["normalize_embeddings"] is True


def test_empty_instruction_embeds_the_query_unchanged():
    model = FakeModel()

    vector = embed_query(model, "What is RAG?", instruction="")

    assert model.encode_calls[0]["texts"] == ["What is RAG?"]
    assert vector == [12.0, 0.0]
