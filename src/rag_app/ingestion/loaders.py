"""Load source files from disk and turn them into cleaned Document objects.

Supported formats: .txt, .md and .pdf. Files that cannot be loaded are
reported with a reason instead of being silently ignored, so a missing
document never goes unnoticed.
"""

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from rag_app.ingestion.cleaning import build_vocabulary, clean_text
from rag_app.models import Document

# Maps each supported file extension to the file_type stored in metadata.
TEXT_FILE_TYPES = {".txt": "text", ".md": "markdown"}
PDF_EXTENSION = ".pdf"

# Placed between PDF pages when joining them into one document text.
# A blank line keeps pages apart without inventing any new words.
PAGE_SEPARATOR = "\n\n"


@dataclass
class SkippedFile:
    """A file that was not loaded, and why."""

    path: str
    reason: str


@dataclass
class LoadResult:
    """Everything load_documents found in a folder.

    Attributes:
        documents: Successfully loaded documents.
        skipped: Files that were not loaded, each with a reason.
    """

    documents: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def make_doc_id(path: Path, root_folder: Path) -> str:
    """Build a readable, stable document ID from the file's path.

    The ID is the path relative to the data folder, always written with
    forward slashes, e.g. "papers/rag_survey.pdf". It stays the same every
    time ingestion runs, which lets evaluation labels refer to documents.
    Renaming or moving a file changes its ID.
    """
    relative_path = path.relative_to(root_folder)
    return relative_path.as_posix()


def load_text_file(path: Path) -> str:
    """Read a .txt or .md file and return its raw text.

    "utf-8-sig" reads normal UTF-8 and also removes the invisible byte-order
    mark that some Windows editors (like Notepad) put at the start of a file.

    Raises:
        UnicodeDecodeError: If the file is not valid UTF-8.
        OSError: If the file cannot be opened.
    """
    with open(path, encoding="utf-8-sig") as file:
        return file.read()


def load_pdf_pages(path: Path) -> list:
    """Extract the raw text of each page of a PDF.

    Returns:
        A list with one string per page. A page with no extractable text
        (for example a scanned image) gives an empty string.

    Raises:
        ValueError: If the PDF is encrypted.
        PyPdfError: If pypdf cannot read the file.
        OSError: If the file cannot be opened.
    """
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("PDF is encrypted; a password would be needed to read it")

    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text is None:
            page_text = ""
        pages.append(page_text)
    return pages


def join_pdf_pages(pages: list) -> tuple:
    """Clean each page and join them into one text, remembering where pages start.

    Each page is cleaned on its own BEFORE joining (using a vocabulary
    built from all pages, for rejoining hyphenated words). That way the recorded
    start positions match the final text exactly, and a chunk's start_char
    can later be turned back into a page number for citations.

    Returns:
        A pair (full_text, page_start_chars). page_start_chars[0] is where
        page 1 starts in full_text, page_start_chars[1] where page 2 starts,
        and so on. An empty page starts where the next page starts.
    """
    # One vocabulary for the whole document: a word written "dual-encoder"
    # on page 2 tells us how to rejoin "dual-\nencoder" on page 7.
    vocabulary = build_vocabulary("\n".join(pages))

    full_text = ""
    page_start_chars = []
    for raw_page in pages:
        page_text = clean_text(raw_page, vocabulary)
        if page_text == "":
            # Position not known yet; filled in by the second loop below.
            page_start_chars.append(None)
            continue
        if full_text != "":
            full_text = full_text + PAGE_SEPARATOR
        page_start_chars.append(len(full_text))
        full_text = full_text + page_text

    # Second pass, from the last page backwards: give each empty page the
    # start position of the next page that has text (or the end of the text
    # if no later page has text). This way no character maps to an empty page.
    next_start = len(full_text)
    for index in range(len(page_start_chars) - 1, -1, -1):
        if page_start_chars[index] is None:
            page_start_chars[index] = next_start
        else:
            next_start = page_start_chars[index]
    return full_text, page_start_chars


def load_document(path: Path, root_folder: Path) -> Document:
    """Load one supported file into a cleaned Document.

    Args:
        path: The file to load.
        root_folder: The data folder, used to build the document ID.

    Returns:
        A Document with cleaned text and metadata.

    Raises:
        ValueError: If the file type is unsupported, the file has no text,
            or a PDF is encrypted. (UnicodeDecodeError is also a ValueError.)
        PyPdfError: If a PDF cannot be read.
        OSError: If the file cannot be opened.
    """
    extension = path.suffix.lower()
    metadata = {"file_name": path.name}

    if extension in TEXT_FILE_TYPES:
        text = clean_text(load_text_file(path))
        metadata["file_type"] = TEXT_FILE_TYPES[extension]
    elif extension == PDF_EXTENSION:
        pages = load_pdf_pages(path)
        text, page_start_chars = join_pdf_pages(pages)
        empty_page_count = 0
        for page in pages:
            if clean_text(page) == "":
                empty_page_count = empty_page_count + 1
        metadata["file_type"] = "pdf"
        metadata["page_count"] = len(pages)
        metadata["empty_page_count"] = empty_page_count
        metadata["page_start_chars"] = page_start_chars
    else:
        raise ValueError(f"unsupported file type '{extension}'")

    if text == "":
        if extension == PDF_EXTENSION:
            raise ValueError("no extractable text; it may be a scanned PDF that needs OCR")
        raise ValueError("file is empty after cleaning")

    metadata["char_count"] = len(text)
    return Document(
        doc_id=make_doc_id(path, root_folder),
        source_path=str(path),
        text=text,
        metadata=metadata,
    )


def load_documents(folder: Path) -> LoadResult:
    """Load every supported file in a folder and its subfolders.

    Files are processed in sorted order so results are the same on every run.
    A file that fails to load is recorded in LoadResult.skipped with the
    reason, and loading continues with the next file.

    Args:
        folder: The data folder to read.

    Returns:
        A LoadResult with the loaded documents and the skipped files.

    Raises:
        FileNotFoundError: If the folder does not exist.
        NotADirectoryError: If the path is a file, not a folder.
    """
    if not folder.exists():
        raise FileNotFoundError(f"data folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"not a folder: {folder}")

    result = LoadResult()
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        try:
            document = load_document(path, folder)
        except (ValueError, OSError, PyPdfError) as error:
            result.skipped.append(SkippedFile(path=str(path), reason=str(error)))
            continue
        result.documents.append(document)
    return result
