"""Project settings: every tunable value lives here instead of in the code.

Each setting has a default. Any default can be overridden with an
environment variable, which makes experiments easy without editing code:

    PowerShell:  $env:RAG_CHUNK_SIZE = "800"

Note: values are read from the process environment only. A .env file is
NOT loaded automatically yet.

Relative paths are resolved from the current working directory, so run
commands from the project root (C:\\dev\\advanced-rag).
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Starting values for chunking. They are reasonable defaults, not proven
# best values; Milestone 2 tunes them using the evaluation dataset.
DEFAULT_CHUNK_SIZE = 1000  # maximum characters per chunk
DEFAULT_CHUNK_OVERLAP = 150  # characters shared between neighbouring chunks

# Names of the available chunking strategies (see rag_app.chunking.chunker).
# "fixed" is the simple baseline; "recursive" respects paragraphs and sentences.
CHUNK_STRATEGIES = ["fixed", "recursive"]
DEFAULT_CHUNK_STRATEGY = "recursive"

DEFAULT_RAW_DATA_DIR = "data/raw"
DEFAULT_PROCESSED_DATA_DIR = "data/processed"
DEFAULT_EVAL_DATASET_PATH = "evals/datasets/questions.jsonl"
# Where ChromaDB stores its files. Generated and git-ignored (under data/).
DEFAULT_INDEX_DIR = "data/indexes/chroma"
# How many chunks a search returns by default.
DEFAULT_TOP_K = 5

# Embeddings. bge-small-en-v1.5 runs locally (no API, documents never leave
# the machine) and accepts up to 512 tokens. Chosen in Milestone 2 after
# measuring the token counts of our chunks.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# How many texts are embedded at once. Larger is faster but uses more memory.
DEFAULT_EMBEDDING_BATCH_SIZE = 32
# BGE recommends putting this text in front of QUESTIONS (not documents) to
# say "find passages that answer this". Set to "" to switch it off.
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass
class Settings:
    """All tunable settings for the pipeline.

    Attributes:
        raw_data_dir: Folder containing the original source documents.
        processed_data_dir: Folder where generated outputs (chunks) are written.
        chunk_size: Maximum number of characters in one chunk.
        chunk_overlap: Number of characters repeated between neighbouring
            chunks, so a sentence cut at a boundary still appears whole
            in at least one chunk.
        chunk_strategy: Which chunking strategy to use; one of CHUNK_STRATEGIES.
        eval_dataset_path: The evaluation questions file (JSONL).
        embedding_model: Hugging Face name of the embedding model.
        embedding_batch_size: Number of texts embedded per batch.
        query_instruction: Text put in front of every query before embedding.
        index_dir: Folder where the vector index (ChromaDB) is stored.
        top_k: How many chunks a search returns by default.
    """

    raw_data_dir: Path
    processed_data_dir: Path
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str
    # These have defaults so code that does not need them can leave them out.
    eval_dataset_path: Path = Path(DEFAULT_EVAL_DATASET_PATH)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    index_dir: Path = Path(DEFAULT_INDEX_DIR)
    top_k: int = DEFAULT_TOP_K


def read_int_from_env(name: str, default: int) -> int:
    """Read an integer environment variable, or return the default.

    Raises:
        ValueError: If the variable is set but is not a whole number.
    """
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got '{value}'") from None


def validate_settings(settings: Settings) -> None:
    """Check that the settings make sense together.

    Raises:
        ValueError: If chunk_size is not positive, chunk_overlap is negative,
            chunk_overlap is not smaller than chunk_size, or chunk_strategy
            is not a known strategy name.
    """
    if settings.chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than 0, got {settings.chunk_size}")
    if settings.chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {settings.chunk_overlap}")
    # If overlap were equal to or larger than the chunk size, the splitter
    # could never move forward through the text.
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError(
            f"chunk_overlap ({settings.chunk_overlap}) must be smaller than "
            f"chunk_size ({settings.chunk_size})"
        )
    if settings.chunk_strategy not in CHUNK_STRATEGIES:
        raise ValueError(
            f"chunk_strategy must be one of {CHUNK_STRATEGIES}, "
            f"got '{settings.chunk_strategy}'"
        )
    if settings.embedding_batch_size <= 0:
        raise ValueError(
            f"embedding_batch_size must be greater than 0, got {settings.embedding_batch_size}"
        )
    if settings.top_k <= 0:
        raise ValueError(f"top_k must be greater than 0, got {settings.top_k}")
    if settings.embedding_model.strip() == "":
        raise ValueError("embedding_model cannot be empty")


def load_settings() -> Settings:
    """Build Settings from defaults plus any environment variable overrides.

    Returns:
        A validated Settings object.

    Raises:
        ValueError: If any value is invalid.
    """
    settings = Settings(
        raw_data_dir=Path(os.environ.get("RAG_RAW_DATA_DIR", DEFAULT_RAW_DATA_DIR)),
        processed_data_dir=Path(
            os.environ.get("RAG_PROCESSED_DATA_DIR", DEFAULT_PROCESSED_DATA_DIR)
        ),
        chunk_size=read_int_from_env("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=read_int_from_env("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        chunk_strategy=os.environ.get("RAG_CHUNK_STRATEGY", DEFAULT_CHUNK_STRATEGY),
        eval_dataset_path=Path(
            os.environ.get("RAG_EVAL_DATASET_PATH", DEFAULT_EVAL_DATASET_PATH)
        ),
        embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_batch_size=read_int_from_env(
            "RAG_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE
        ),
        # An empty value is allowed here: it switches the instruction off.
        query_instruction=os.environ.get("RAG_QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION),
        index_dir=Path(os.environ.get("RAG_INDEX_DIR", DEFAULT_INDEX_DIR)),
        top_k=read_int_from_env("RAG_TOP_K", DEFAULT_TOP_K),
    )
    validate_settings(settings)
    return settings
