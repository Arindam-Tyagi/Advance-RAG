"""Core data shapes shared by every layer of the pipeline.

Ingestion produces Document objects. Chunking turns each Document into a list
of Chunk objects. Later layers (embeddings, retrieval, generation) only ever
pass these two shapes around, so each layer can change without breaking the
others.
"""

from dataclasses import dataclass, field


@dataclass
class Document:
    """One source file after it has been loaded and cleaned.

    Attributes:
        doc_id: Stable identifier for the document, used to link chunks
            and evaluation evidence back to their source.
        source_path: Path of the original file, so answers can be traced
            back to it.
        text: The full cleaned text of the document.
        metadata: Extra facts about the document, such as file type or
            page count.
    """

    doc_id: str
    source_path: str
    text: str
    # default_factory=dict gives every Document its own empty dict.
    # Writing "metadata: dict = {}" would make all documents share ONE dict.
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a Document's text that will be embedded and retrieved.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        doc_id: The doc_id of the Document this chunk came from.
        text: The chunk's text, an exact slice of the document text.
        start_char: Index in Document.text where this chunk starts.
        end_char: Index in Document.text where this chunk ends (exclusive),
            so document.text[start_char:end_char] == chunk.text.
        metadata: Extra facts about the chunk, such as which chunker made it.
    """

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)
