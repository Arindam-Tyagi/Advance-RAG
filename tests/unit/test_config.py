"""Tests for loading and validating settings."""

from pathlib import Path

import pytest

from rag_app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_STRATEGY,
    load_settings,
)

SETTING_VARIABLES = [
    "RAG_RAW_DATA_DIR",
    "RAG_PROCESSED_DATA_DIR",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_CHUNK_STRATEGY",
    "RAG_EVAL_DATASET_PATH",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_BATCH_SIZE",
    "RAG_QUERY_INSTRUCTION",
]


@pytest.fixture(autouse=True)
def clear_setting_variables(monkeypatch):
    """Remove RAG_* variables so the real shell environment can't affect tests."""
    for name in SETTING_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_used_when_no_variables_are_set():
    settings = load_settings()

    assert settings.raw_data_dir == Path("data/raw")
    assert settings.processed_data_dir == Path("data/processed")
    assert settings.chunk_size == DEFAULT_CHUNK_SIZE
    assert settings.chunk_overlap == DEFAULT_CHUNK_OVERLAP
    assert settings.chunk_strategy == DEFAULT_CHUNK_STRATEGY


def test_chunk_strategy_can_be_overridden(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_STRATEGY", "fixed")

    assert load_settings().chunk_strategy == "fixed"


def test_unknown_chunk_strategy_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_STRATEGY", "semantic")

    with pytest.raises(ValueError, match="chunk_strategy must be one of"):
        load_settings()


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "50")
    monkeypatch.setenv("RAG_RAW_DATA_DIR", "other/raw")

    settings = load_settings()

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.raw_data_dir == Path("other/raw")


def test_non_number_chunk_size_gives_clear_error(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "big")

    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE must be a whole number"):
        load_settings()


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")

    with pytest.raises(ValueError, match="must be smaller than"):
        load_settings()


def test_chunk_size_must_be_positive(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "0")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "0")

    with pytest.raises(ValueError, match="greater than 0"):
        load_settings()


def test_negative_overlap_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "-1")

    with pytest.raises(ValueError, match="cannot be negative"):
        load_settings()


def test_embedding_defaults():
    settings = load_settings()

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_batch_size == 32
    assert settings.query_instruction.startswith("Represent this sentence")


def test_empty_query_instruction_switches_it_off(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INSTRUCTION", "")

    assert load_settings().query_instruction == ""


def test_embedding_batch_size_must_be_positive(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "0")

    with pytest.raises(ValueError, match="embedding_batch_size must be greater than 0"):
        load_settings()
