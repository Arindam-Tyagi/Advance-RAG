"""Tests for loading files into Document objects."""

from pathlib import Path

import pytest
from pdf_helper import write_simple_pdf

from rag_app.ingestion.loaders import (
    join_pdf_pages,
    load_document,
    load_documents,
    make_doc_id,
)

FIXTURES_FOLDER = Path(__file__).parent.parent / "fixtures"


# --- make_doc_id -----------------------------------------------------------


def test_doc_id_is_relative_path_with_forward_slashes(tmp_path):
    path = tmp_path / "papers" / "rag.pdf"

    assert make_doc_id(path, tmp_path) == "papers/rag.pdf"


# --- text and markdown files ----------------------------------------------


def test_text_file_is_loaded_and_cleaned():
    document = load_document(FIXTURES_FOLDER / "sample.txt", FIXTURES_FOLDER)

    assert document.doc_id == "sample.txt"
    assert document.text == (
        "Retrieval-Augmented Generation combines search with a language model.\n\n"
        "The model answers using passages found by the retriever."
    )
    assert document.metadata["file_type"] == "text"
    assert document.metadata["char_count"] == len(document.text)


def test_markdown_file_keeps_heading_and_indentation():
    document = load_document(FIXTURES_FOLDER / "sample.md", FIXTURES_FOLDER)

    assert document.metadata["file_type"] == "markdown"
    assert document.text.startswith("# Chunking")
    assert "\n    indented code keeps its leading spaces" in document.text


def test_byte_order_mark_is_removed(tmp_path):
    path = tmp_path / "notepad.txt"
    path.write_bytes(b"\xef\xbb\xbfHello")

    document = load_document(path, tmp_path)

    assert document.text == "Hello"


def test_empty_text_file_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty after cleaning"):
        load_document(path, tmp_path)


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "report.docx"
    path.write_bytes(b"not really a docx")

    with pytest.raises(ValueError, match="unsupported file type '.docx'"):
        load_document(path, tmp_path)


# --- PDF files -------------------------------------------------------------


def test_pdf_pages_are_joined_and_page_starts_recorded(tmp_path):
    path = tmp_path / "paper.pdf"
    write_simple_pdf(path, ["First page text.", "Second page text."])

    document = load_document(path, tmp_path)

    assert document.metadata["file_type"] == "pdf"
    assert document.metadata["page_count"] == 2
    assert document.metadata["empty_page_count"] == 0
    starts = document.metadata["page_start_chars"]
    assert document.text[starts[0]:].startswith("First page text.")
    assert document.text[starts[1]:].startswith("Second page text.")


def test_pdf_with_no_text_is_rejected_with_ocr_hint(tmp_path):
    path = tmp_path / "scanned.pdf"
    write_simple_pdf(path, ["", ""])

    with pytest.raises(ValueError, match="may be a scanned PDF"):
        load_document(path, tmp_path)


def test_join_pdf_pages_handles_empty_middle_page():
    text, starts = join_pdf_pages(["Page one.", "", "Page three."])

    assert text == "Page one.\n\nPage three."
    # The empty page 2 starts where page 3 starts.
    assert starts == [0, 11, 11]
    assert text[starts[2]:] == "Page three."


# --- load_documents (whole folder) -----------------------------------------


def test_folder_loading_reports_skipped_files_and_continues(tmp_path):
    (tmp_path / "good.txt").write_text("Some content.", encoding="utf-8")
    (tmp_path / "notes.docx").write_bytes(b"unsupported")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf")
    subfolder = tmp_path / "more"
    subfolder.mkdir()
    (subfolder / "nested.md").write_text("# Nested", encoding="utf-8")

    result = load_documents(tmp_path)

    loaded_ids = []
    for document in result.documents:
        loaded_ids.append(document.doc_id)
    skipped_names = []
    for skipped in result.skipped:
        skipped_names.append(Path(skipped.path).name)

    assert loaded_ids == ["good.txt", "more/nested.md"]
    assert sorted(skipped_names) == ["broken.pdf", "notes.docx"]
    for skipped in result.skipped:
        assert skipped.reason != ""


def test_non_utf8_text_file_is_skipped_not_crashing(tmp_path):
    (tmp_path / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))

    result = load_documents(tmp_path)

    assert result.documents == []
    assert len(result.skipped) == 1


def test_missing_folder_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data folder not found"):
        load_documents(tmp_path / "does_not_exist")


def test_pdf_pages_share_one_vocabulary_for_hyphenation():
    # Page 1 writes "dual-encoder"; page 2 splits it across a line.
    pages = ["We train a dual-encoder.", "The dual-\nencoder is fast."]

    text, starts = join_pdf_pages(pages)

    assert text[starts[1]:] == "The dual-encoder is fast."
