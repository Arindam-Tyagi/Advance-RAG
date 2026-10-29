"""Load and validate the evaluation dataset.

The dataset is a JSONL file with one question per line. Each question lists
its evidence as exact quotes from the documents, NOT as chunk IDs. Chunk IDs
change whenever the chunking strategy changes; quotes do not. Later, a chunk
counts as relevant to a question if it overlaps one of the question's
evidence quotes, so the same labels work for every chunking strategy.

The format is documented in evals/README.md.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

QUESTION_TYPES = ["factual", "multi_doc", "unanswerable"]
STATUSES = ["draft", "verified"]
REQUIRED_FIELDS = ["question_id", "question", "reference_answer", "question_type", "evidence", "status"]


@dataclass
class EvidenceQuote:
    """A passage that supports the answer.

    Attributes:
        doc_id: The document containing the passage.
        quote: The passage, copied from documents.jsonl. Line breaks and
            repeated spaces do not need to match exactly.
    """

    doc_id: str
    quote: str


@dataclass
class EvalQuestion:
    """One evaluation question with its expected answer and evidence.

    Attributes:
        question_id: Unique ID such as "q001".
        question: The question as a user would ask it.
        reference_answer: A short correct answer, written by a person.
        question_type: One of QUESTION_TYPES.
        evidence: EvidenceQuote objects. Empty for unanswerable questions.
        status: "draft" until a person has checked it, then "verified".
        notes: Optional free text, e.g. why a question is unanswerable.
    """

    question_id: str
    question: str
    reference_answer: str
    question_type: str
    evidence: list
    status: str
    notes: str = ""


def parse_question(row: dict, line_number: int) -> EvalQuestion:
    """Turn one parsed JSON object into an EvalQuestion.

    Raises:
        ValueError: If a required field is missing or has the wrong form.
            The message includes the line number.
    """
    for field_name in REQUIRED_FIELDS:
        if field_name not in row:
            raise ValueError(f"line {line_number}: missing field '{field_name}'")

    if row["question_type"] not in QUESTION_TYPES:
        raise ValueError(
            f"line {line_number}: question_type must be one of {QUESTION_TYPES}, "
            f"got '{row['question_type']}'"
        )
    if row["status"] not in STATUSES:
        raise ValueError(
            f"line {line_number}: status must be one of {STATUSES}, got '{row['status']}'"
        )
    if not isinstance(row["evidence"], list):
        raise ValueError(f"line {line_number}: evidence must be a list")

    evidence = []
    for item in row["evidence"]:
        if not isinstance(item, dict) or "doc_id" not in item or "quote" not in item:
            raise ValueError(
                f"line {line_number}: each evidence item needs 'doc_id' and 'quote'"
            )
        evidence.append(EvidenceQuote(doc_id=item["doc_id"], quote=item["quote"]))

    return EvalQuestion(
        question_id=row["question_id"],
        question=row["question"],
        reference_answer=row["reference_answer"],
        question_type=row["question_type"],
        evidence=evidence,
        status=row["status"],
        notes=row.get("notes", ""),
    )


def load_eval_questions(path: Path) -> list:
    """Read every question from a JSONL dataset file. Blank lines are ignored.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is not valid JSON or not a valid question.
    """
    questions = []
    with open(path, encoding="utf-8") as file:
        line_number = 0
        for line in file:
            line_number = line_number + 1
            if line.strip() == "":
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON ({error})") from None
            questions.append(parse_question(row, line_number))
    return questions


def load_documents_by_id(path: Path) -> dict:
    """Read documents.jsonl (written by the ingest command) into {doc_id: text}."""
    documents = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            documents[row["doc_id"]] = row["text"]
    return documents


def find_quote_spans(text: str, quote: str) -> list:
    """Find every place a quote appears in a text.

    Whitespace is flexible: a single space in the quote matches any run of
    spaces or line breaks in the text. This matters because PDF text has a
    line break every ~45 characters, which people do not type when copying.

    Returns:
        A list of (start, end) pairs, one per match, as positions in text.
        An empty list means the quote was not found.
    """
    words = quote.split()
    if len(words) == 0:
        return []

    # Build a pattern like "dense\s+vector\s+index": each word is escaped so
    # characters like "(" or "." are matched literally, and words are joined
    # by \s+ ("one or more whitespace characters").
    escaped_words = []
    for word in words:
        escaped_words.append(re.escape(word))
    pattern = r"\s+".join(escaped_words)

    spans = []
    for match in re.finditer(pattern, text):
        spans.append((match.start(), match.end()))
    return spans


def validate_dataset(questions: list, documents_by_id: dict) -> list:
    """Check the dataset against the real document texts.

    Checks:
        - question IDs are unique;
        - unanswerable questions have no evidence; others have some;
        - multi_doc questions cite at least two different documents;
        - every evidence doc_id exists;
        - every quote appears in its document exactly once.

    Returns:
        A list of human-readable problems. An empty list means the dataset
        is valid.
    """
    problems = []
    seen_ids = set()

    for question in questions:
        qid = question.question_id
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate question_id")
        seen_ids.add(qid)

        if question.question_type == "unanswerable":
            if len(question.evidence) > 0:
                problems.append(f"{qid}: unanswerable questions must have no evidence")
        elif len(question.evidence) == 0:
            problems.append(f"{qid}: needs at least one evidence quote")

        if question.question_type == "multi_doc":
            cited_documents = set()
            for evidence in question.evidence:
                cited_documents.add(evidence.doc_id)
            if len(cited_documents) < 2:
                problems.append(f"{qid}: multi_doc questions must cite at least two documents")

        for evidence in question.evidence:
            if evidence.doc_id not in documents_by_id:
                problems.append(f"{qid}: unknown doc_id '{evidence.doc_id}'")
                continue
            spans = find_quote_spans(documents_by_id[evidence.doc_id], evidence.quote)
            if len(spans) == 0:
                problems.append(f"{qid}: quote not found in {evidence.doc_id}: '{evidence.quote[:60]}...'")
            elif len(spans) > 1:
                problems.append(
                    f"{qid}: quote appears {len(spans)} times in {evidence.doc_id}; "
                    "make it longer so it is unique"
                )

    return problems


@dataclass
class EvidenceSpan:
    """Where an evidence quote sits in its document.

    Attributes:
        doc_id: The document containing the quote.
        start: Index of the quote's first character in the document text.
        end: Index just after the quote's last character (exclusive).
    """

    doc_id: str
    start: int
    end: int


def locate_evidence(question: EvalQuestion, documents_by_id: dict) -> list:
    """Find the exact position of every evidence quote of a question.

    Returns:
        One EvidenceSpan per evidence quote, in the same order.
        Empty for unanswerable questions.

    Raises:
        ValueError: If a document is unknown or a quote is not found exactly
            once. (Running validate_dataset first reports all such problems.)
    """
    spans = []
    for evidence in question.evidence:
        if evidence.doc_id not in documents_by_id:
            raise ValueError(f"{question.question_id}: unknown doc_id '{evidence.doc_id}'")
        matches = find_quote_spans(documents_by_id[evidence.doc_id], evidence.quote)
        if len(matches) != 1:
            raise ValueError(
                f"{question.question_id}: quote found {len(matches)} times in "
                f"{evidence.doc_id}, expected exactly once"
            )
        start, end = matches[0]
        spans.append(EvidenceSpan(doc_id=evidence.doc_id, start=start, end=end))
    return spans
