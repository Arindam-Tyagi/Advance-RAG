"""Turn text into embedding vectors with a local sentence-transformers model.

An embedding is a list of numbers (384 for bge-small-en-v1.5) that captures
the meaning of a text: texts with similar meaning get similar vectors.
Retrieval later compares a question's vector with every chunk's vector.

All vectors are normalized to length 1. For normalized vectors, cosine
similarity equals the dot product, which is what the vector store uses.

The model runs on this computer. After a one-time download of the model
weights from Hugging Face, no text is sent anywhere.
"""

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer


@dataclass
class OverLimitChunk:
    """A chunk with more tokens than the model can read.

    The model silently ignores every token after its limit, so the end of
    this chunk's text does not affect its embedding.
    """

    chunk_id: str
    token_count: int


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load an embedding model, downloading it the first time.

    Downloaded models are cached in the Hugging Face cache folder in your
    user directory, so later loads work offline.

    Raises:
        RuntimeError: If the model cannot be downloaded or loaded, for
            example because the name is wrong or there is no internet.
    """
    try:
        return SentenceTransformer(model_name)
    except OSError as error:
        raise RuntimeError(
            f"could not load embedding model '{model_name}': {error}. "
            "Check the model name and your internet connection "
            "(the first load downloads the model)."
        ) from error


def get_token_limit(model: SentenceTransformer) -> int:
    """Return the maximum number of tokens the model reads from one text."""
    return model.max_seq_length


def count_tokens(model: SentenceTransformer, text: str) -> int:
    """Count the tokens the model would see for a text, without cutting it off.

    The count includes the special start and end tokens the model adds,
    because they also count towards the limit.
    """
    encoded = model.tokenizer(text, add_special_tokens=True, truncation=False)
    return len(encoded["input_ids"])


def find_chunks_over_token_limit(model: SentenceTransformer, chunks: list) -> list:
    """Find chunks that are too long for the model.

    Returns:
        OverLimitChunk objects, in chunk order. Empty if every chunk fits.
    """
    limit = get_token_limit(model)
    over_limit = []
    for chunk in chunks:
        token_count = count_tokens(model, chunk.text)
        if token_count > limit:
            over_limit.append(OverLimitChunk(chunk_id=chunk.chunk_id, token_count=token_count))
    return over_limit


def embed_documents(model: SentenceTransformer, texts: list, batch_size: int) -> list:
    """Embed document texts (chunks). No instruction is added to documents.

    Args:
        model: A loaded embedding model.
        texts: The texts to embed.
        batch_size: How many texts to process at once.

    Returns:
        One vector (a list of floats) per text, in the same order.
    """
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # tolist() turns the numpy array into plain Python lists of floats.
    return vectors.tolist()


def embed_query(model: SentenceTransformer, query: str, instruction: str) -> list:
    """Embed a user question.

    The instruction (e.g. "Represent this sentence for searching relevant
    passages: ") is put in front of the question. Documents are embedded
    without it; the instruction tells the model the two sides play
    different roles (a short question vs. a longer passage).

    Returns:
        The query vector as a list of floats.
    """
    text = instruction + query
    vectors = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors[0].tolist()
