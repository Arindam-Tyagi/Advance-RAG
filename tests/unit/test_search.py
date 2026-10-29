"""Tests for the search command's small helpers."""

from rag_app import search
from rag_app.models import Chunk


def make_chunk(metadata: dict) -> Chunk:
    return Chunk("d::0", "d", "text", 0, 4, metadata)


def test_page_range_formats():
    assert search.format_pages(make_chunk({"page_start": 3, "page_end": 3})) == "p. 3"
    assert search.format_pages(make_chunk({"page_start": 3, "page_end": 4})) == "p. 3-4"
    assert search.format_pages(make_chunk({})) == ""


def test_missing_question_prints_usage(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["search"])

    exit_code = search.main()

    assert exit_code == 1
    assert "Usage" in capsys.readouterr().err
