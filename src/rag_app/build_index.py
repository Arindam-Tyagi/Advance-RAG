"""Build-index command: embed the chunks and store them in ChromaDB.

Run from the project root, after the ingest command:

    uv run python -m rag_app.build_index

It uses the chunks file for Settings.chunk_strategy and the model in
Settings.embedding_model. To index the other strategy, set
RAG_CHUNK_STRATEGY and run it again; each setup gets its own collection.

Steps:
    1. Read the chunks written by the ingest command.
    2. Load the embedding model.
    3. Report chunks longer than the model's token limit (their ends are
       ignored by the model; we report them instead of hiding it).
    4. Embed every chunk.
    5. Store chunks and vectors in a fresh Chroma collection.
"""

import sys
import time
from datetime import datetime, timezone

from rag_app.config import load_settings
from rag_app.embeddings.embedder import (
    embed_documents,
    find_chunks_over_token_limit,
    get_token_limit,
    load_embedding_model,
)
from rag_app.ingest import chunks_file_path, read_chunks
from rag_app.retrieval.vector_store import build_index, make_collection_name, open_chroma_client


def main() -> int:
    """Entry point for `python -m rag_app.build_index`.

    Returns:
        0 on success, 1 if settings are invalid, the chunks file is missing,
        the model cannot be loaded, or there are no chunks.
    """
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")

    try:
        settings = load_settings()
    except ValueError as error:
        print(f"Invalid settings: {error}", file=sys.stderr)
        return 1

    chunks_path = chunks_file_path(settings)
    try:
        chunks = read_chunks(chunks_path)
    except FileNotFoundError:
        print(f"Chunks file not found: {chunks_path}. Run the ingest command first.", file=sys.stderr)
        return 1
    if len(chunks) == 0:
        print(f"No chunks in {chunks_path}.", file=sys.stderr)
        return 1
    print(f"Chunks:     {len(chunks)} from {chunks_path}")

    try:
        model = load_embedding_model(settings.embedding_model)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    token_limit = get_token_limit(model)
    print(f"Model:      {settings.embedding_model} (token limit {token_limit})")

    over_limit = find_chunks_over_token_limit(model, chunks)
    print(f"Over limit: {len(over_limit)} chunk(s) longer than {token_limit} tokens (their ends are ignored)")
    for item in over_limit:
        print(f"  - {item.chunk_id}: {item.token_count} tokens")

    texts = []
    for chunk in chunks:
        texts.append(chunk.text)
    start_time = time.perf_counter()
    vectors = embed_documents(model, texts, settings.embedding_batch_size)
    embed_seconds = time.perf_counter() - start_time
    print(f"Embedded:   {len(vectors)} chunks in {embed_seconds:.1f}s")

    collection_name = make_collection_name(settings.chunk_strategy, settings.embedding_model)
    # Stored with the index so a later query can check it uses the same model.
    collection_metadata = {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": len(vectors[0]),
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunk_count": len(chunks),
        "over_token_limit_count": len(over_limit),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    client = open_chroma_client(settings.index_dir)
    collection = build_index(client, collection_name, chunks, vectors, collection_metadata)
    print(f"Index:      collection '{collection_name}' with {collection.count()} chunks in {settings.index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
