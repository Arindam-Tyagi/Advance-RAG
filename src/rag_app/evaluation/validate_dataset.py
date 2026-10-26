"""Command that checks the evaluation dataset against the ingested documents.

Run from the project root, after the ingest command:

    uv run python -m rag_app.evaluation.validate_dataset

Exit code 0 means every question is well-formed and every evidence quote
was found exactly once in its document.
"""

import sys

from rag_app.config import load_settings
from rag_app.evaluation.dataset import load_documents_by_id, load_eval_questions, validate_dataset
from rag_app.ingest import DOCUMENTS_FILE_NAME


def count_by(questions: list, attribute: str) -> dict:
    """Count questions by the value of one attribute, e.g. "status"."""
    counts = {}
    for question in questions:
        value = getattr(question, attribute)
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> int:
    """Validate the dataset and print a summary.

    Returns:
        0 if valid, 1 if files are missing or any problem is found.
    """
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")

    try:
        settings = load_settings()
        documents_path = settings.processed_data_dir / DOCUMENTS_FILE_NAME
        questions = load_eval_questions(settings.eval_dataset_path)
        documents_by_id = load_documents_by_id(documents_path)
    except FileNotFoundError as error:
        print(f"Missing file: {error.filename}. Run the ingest command first?", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Invalid dataset: {error}", file=sys.stderr)
        return 1

    print(f"Dataset:   {settings.eval_dataset_path}")
    print(f"Questions: {len(questions)}")
    print(f"By type:   {count_by(questions, 'question_type')}")
    print(f"By status: {count_by(questions, 'status')}")

    problems = validate_dataset(questions, documents_by_id)
    if len(problems) > 0:
        print(f"\n{len(problems)} problem(s) found:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\nAll questions are valid and every evidence quote was found exactly once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
