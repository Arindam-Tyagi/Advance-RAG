"""Tests for the Document and Chunk data shapes."""

from rag_app.models import Chunk, Document


def test_document_stores_given_values():
    document = Document(doc_id="doc-1", source_path="data/raw/a.txt", text="Hello")

    assert document.doc_id == "doc-1"
    assert document.source_path == "data/raw/a.txt"
    assert document.text == "Hello"
    assert document.metadata == {}


def test_each_document_gets_its_own_metadata_dict():
    # Guards against the shared-mutable-default bug: changing one
    # document's metadata must not change another's.
    first = Document(doc_id="a", source_path="a.txt", text="A")
    second = Document(doc_id="b", source_path="b.txt", text="B")

    first.metadata["pages"] = 3

    assert second.metadata == {}


def test_chunk_offsets_point_back_into_document_text():
    document = Document(doc_id="doc-1", source_path="a.txt", text="Hello world")
    chunk = Chunk(chunk_id="doc-1-0", doc_id="doc-1", text="world", start_char=6, end_char=11)

    assert document.text[chunk.start_char:chunk.end_char] == chunk.text
