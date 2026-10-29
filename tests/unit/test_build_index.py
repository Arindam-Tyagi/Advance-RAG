"""Tests for the build-index command, with a fake embedding model."""

import numpy as np

from rag_app import build_index as build_index_command
from rag_app.ingest import write_jsonl
from rag_app.models import Chunk
from rag_app.retrieval.vector_store import open_chroma_client


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": [0] * (len(text.split()) + 2)}


class FakeModel:
    """Makes 2-number vectors; treats each word as one token; limit 4 tokens."""

    max_seq_length = 4
    tokenizer = FakeTokenizer()

    def encode(self, texts, **options):
        vectors = []
        for text in texts:
            vectors.append([float(len(text)), 1.0])
        return np.array(vectors)


def fake_loader(model_name):
    return FakeModel()


def set_up_folders(tmp_path, monkeypatch):
    """Point every setting at temporary folders and use the fake model."""
    monkeypatch.setenv("RAG_PROCESSED_DATA_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RAG_CHUNK_STRATEGY", "recursive")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake/model")
    # Replace the real model loader inside the command module.
    monkeypatch.setattr(build_index_command, "load_embedding_model", fake_loader)


def test_command_builds_collection_and_reports_long_chunks(tmp_path, monkeypatch, capsys):
    set_up_folders(tmp_path, monkeypatch)
    chunks = [
        Chunk("a.txt::0", "a.txt", "short text", 0, 10, {"chunk_index": 0}),
        Chunk("a.txt::1", "a.txt", "this chunk has far too many words", 10, 43, {"chunk_index": 1}),
    ]
    write_jsonl(chunks, tmp_path / "processed" / "chunks_recursive.jsonl")

    exit_code = build_index_command.main()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Over limit: 1 chunk(s)" in output
    assert "a.txt::1" in output
    collection = open_chroma_client(tmp_path / "chroma").get_collection("recursive__fake-model")
    assert collection.count() == 2
    assert collection.metadata["embedding_model"] == "fake/model"
    assert collection.metadata["over_token_limit_count"] == 1


def test_command_fails_clearly_without_chunks_file(tmp_path, monkeypatch, capsys):
    set_up_folders(tmp_path, monkeypatch)

    exit_code = build_index_command.main()

    assert exit_code == 1
    assert "Run the ingest command first" in capsys.readouterr().err
