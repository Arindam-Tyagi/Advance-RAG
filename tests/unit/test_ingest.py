"""Tests for the ingest command."""

import json
from pathlib import Path

from pdf_helper import write_simple_pdf

from rag_app.config import Settings
from rag_app.ingest import compute_chunk_stats, main, run_ingestion
from rag_app.models import Chunk


def make_settings(raw_folder: Path, processed_folder: Path) -> Settings:
    return Settings(
        raw_data_dir=raw_folder,
        processed_data_dir=processed_folder,
        chunk_size=50,
        chunk_overlap=10,
        chunk_strategy="recursive",
    )


def read_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def test_run_ingestion_writes_documents_and_chunks(tmp_path):
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    (raw_folder / "notes.txt").write_text(
        "A first paragraph about retrieval.\n\nA second paragraph about chunking.",
        encoding="utf-8",
    )
    write_simple_pdf(raw_folder / "paper.pdf", ["Page one of the paper.", "Page two."])
    (raw_folder / "ignore.docx").write_bytes(b"unsupported")
    settings = make_settings(raw_folder, tmp_path / "processed")

    result = run_ingestion(settings)

    assert len(result.documents) == 2
    assert len(result.skipped) == 1
    assert result.chunks_path.name == "chunks_recursive.jsonl"

    document_rows = read_jsonl(result.documents_path)
    chunk_rows = read_jsonl(result.chunks_path)
    assert len(document_rows) == 2
    assert len(chunk_rows) == len(result.chunks)

    # Every saved chunk must still slice back to its document's saved text.
    texts_by_id = {}
    for row in document_rows:
        texts_by_id[row["doc_id"]] = row["text"]
    for row in chunk_rows:
        document_text = texts_by_id[row["doc_id"]]
        assert document_text[row["start_char"]:row["end_char"]] == row["text"]


def test_compute_chunk_stats():
    chunks = [
        Chunk(chunk_id="a::0", doc_id="a", text="12345", start_char=0, end_char=5),
        Chunk(chunk_id="a::1", doc_id="a", text="123", start_char=5, end_char=8),
    ]

    stats = compute_chunk_stats(chunks)

    assert stats == {"chunk_count": 2, "min_chars": 3, "avg_chars": 4, "max_chars": 5}


def test_compute_chunk_stats_with_no_chunks():
    assert compute_chunk_stats([])["chunk_count"] == 0


def test_main_returns_error_code_when_folder_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAG_RAW_DATA_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("RAG_PROCESSED_DATA_DIR", str(tmp_path / "processed"))

    exit_code = main()

    assert exit_code == 1
    assert "data folder not found" in capsys.readouterr().err


def test_main_returns_error_code_when_no_documents_load(tmp_path, monkeypatch, capsys):
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    (raw_folder / "only.docx").write_bytes(b"unsupported")
    monkeypatch.setenv("RAG_RAW_DATA_DIR", str(raw_folder))
    monkeypatch.setenv("RAG_PROCESSED_DATA_DIR", str(tmp_path / "processed"))

    exit_code = main()

    assert exit_code == 1
    assert "No documents were loaded" in capsys.readouterr().err
