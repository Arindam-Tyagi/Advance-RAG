"""Search command: ask a question and see which chunks are retrieved.

Run from the project root, after build_index:

    uv run python -m rag_app.search "Which library does DPR use for indexing?"

Uses Settings.chunk_strategy, Settings.embedding_model and Settings.top_k
(RAG_TOP_K). This only RETRIEVES chunks; no answer is generated yet.
"""

import sys
import time

from rag_app.config import load_settings
from rag_app.embeddings.embedder import load_embedding_model
from rag_app.retrieval.vector_retriever import IndexMismatchError, open_vector_retriever
from rag_app.retrieval.vector_store import IndexNotFoundError

# How many characters of each chunk to show.
PREVIEW_LENGTH = 300


def format_pages(chunk) -> str:
    """Describe where a chunk is, e.g. "p. 3" or "p. 3-4"; "" if not a PDF."""
    page_start = chunk.metadata.get("page_start")
    page_end = chunk.metadata.get("page_end")
    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return f"p. {page_start}"
    return f"p. {page_start}-{page_end}"


def main() -> int:
    """Entry point for `python -m rag_app.search "<question>"`.

    Returns:
        0 on success, 1 if no question was given or the index cannot be used.
    """
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")

    query = " ".join(sys.argv[1:]).strip()
    if query == "":
        print('Usage: python -m rag_app.search "your question"', file=sys.stderr)
        return 1

    try:
        settings = load_settings()
        model = load_embedding_model(settings.embedding_model)
        retriever = open_vector_retriever(settings, model)
    except (ValueError, RuntimeError, IndexNotFoundError, IndexMismatchError) as error:
        print(f"Cannot search: {error}", file=sys.stderr)
        return 1

    start_time = time.perf_counter()
    results = retriever.retrieve(query, settings.top_k)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"Question: {query}")
    print(
        f"Index: {settings.chunk_strategy} chunks, {settings.embedding_model}, "
        f"top {settings.top_k}, {elapsed_ms:.0f} ms"
    )
    for result in results:
        chunk = result.chunk
        preview = chunk.text[:PREVIEW_LENGTH].replace("\n", " ")
        print()
        print(f"#{result.rank}  score={result.score:.3f}  {chunk.doc_id}  {format_pages(chunk)}")
        print(f"    {preview}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
