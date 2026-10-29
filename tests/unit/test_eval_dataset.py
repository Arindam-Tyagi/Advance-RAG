"""Tests for loading and validating the evaluation dataset."""

import json
from pathlib import Path

import pytest

from rag_app.evaluation.dataset import (
    EvalQuestion,
    EvidenceQuote,
    find_quote_spans,
    load_eval_questions,
    locate_evidence,
    validate_dataset,
)

DOCUMENTS = {
    "a.pdf": "Dense retrieval uses\nembeddings. BM25 uses\nkeywords. BM25 is fast.",
    "b.pdf": "Rerankers reorder the retrieved passages.",
}


def make_question(question_id="q1", question_type="factual", evidence=None) -> EvalQuestion:
    if evidence is None:
        evidence = [EvidenceQuote(doc_id="a.pdf", quote="Dense retrieval uses embeddings.")]
    return EvalQuestion(
        question_id=question_id,
        question="A question?",
        reference_answer="An answer.",
        question_type=question_type,
        evidence=evidence,
        status="draft",
    )


def write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


VALID_ROW = {
    "question_id": "q1",
    "question": "What does dense retrieval use?",
    "reference_answer": "Embeddings.",
    "question_type": "factual",
    "evidence": [{"doc_id": "a.pdf", "quote": "Dense retrieval uses embeddings."}],
    "status": "draft",
}


# --- find_quote_spans ---------------------------------------------------------


def test_quote_matches_across_line_breaks():
    spans = find_quote_spans(DOCUMENTS["a.pdf"], "Dense retrieval uses embeddings.")

    assert spans == [(0, 32)]
    assert DOCUMENTS["a.pdf"][0:32] == "Dense retrieval uses\nembeddings."


def test_quote_with_special_characters_is_matched_literally():
    text = "accuracy (top-20) improved by 9%-19%."

    assert len(find_quote_spans(text, "(top-20) improved by 9%-19%.")) == 1


def test_quote_not_in_text_gives_no_spans():
    assert find_quote_spans(DOCUMENTS["a.pdf"], "sparse vectors") == []


def test_repeated_quote_gives_several_spans():
    assert len(find_quote_spans(DOCUMENTS["a.pdf"], "BM25")) == 2


# --- load_eval_questions ------------------------------------------------------


def test_valid_file_is_loaded(tmp_path):
    path = tmp_path / "questions.jsonl"
    write_jsonl(path, [VALID_ROW])

    questions = load_eval_questions(path)

    assert len(questions) == 1
    assert questions[0].evidence[0].doc_id == "a.pdf"
    assert questions[0].notes == ""


def test_missing_field_reports_line_number(tmp_path):
    row = dict(VALID_ROW)
    del row["reference_answer"]
    path = tmp_path / "questions.jsonl"
    write_jsonl(path, [VALID_ROW, row])

    with pytest.raises(ValueError, match="line 2: missing field 'reference_answer'"):
        load_eval_questions(path)


def test_unknown_question_type_is_rejected(tmp_path):
    row = dict(VALID_ROW)
    row["question_type"] = "opinion"
    path = tmp_path / "questions.jsonl"
    write_jsonl(path, [row])

    with pytest.raises(ValueError, match="question_type must be one of"):
        load_eval_questions(path)


def test_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1: invalid JSON"):
        load_eval_questions(path)


# --- validate_dataset ---------------------------------------------------------


def test_valid_dataset_has_no_problems():
    questions = [
        make_question("q1"),
        make_question(
            "q2",
            "multi_doc",
            [
                EvidenceQuote("a.pdf", "BM25 uses keywords."),
                EvidenceQuote("b.pdf", "Rerankers reorder the retrieved passages."),
            ],
        ),
        make_question("q3", "unanswerable", []),
    ]

    assert validate_dataset(questions, DOCUMENTS) == []


def test_quote_that_is_not_in_document_is_reported():
    question = make_question(evidence=[EvidenceQuote("a.pdf", "made-up sentence")])

    problems = validate_dataset([question], DOCUMENTS)

    assert len(problems) == 1
    assert "quote not found" in problems[0]


def test_ambiguous_quote_is_reported():
    question = make_question(evidence=[EvidenceQuote("a.pdf", "BM25")])

    problems = validate_dataset([question], DOCUMENTS)

    assert "appears 2 times" in problems[0]


def test_unknown_document_is_reported():
    question = make_question(evidence=[EvidenceQuote("missing.pdf", "anything")])

    assert "unknown doc_id 'missing.pdf'" in validate_dataset([question], DOCUMENTS)[0]


def test_duplicate_question_ids_are_reported():
    problems = validate_dataset([make_question("q1"), make_question("q1")], DOCUMENTS)

    assert problems == ["q1: duplicate question_id"]


def test_type_rules_are_enforced():
    unanswerable_with_evidence = make_question("q1", "unanswerable")
    factual_without_evidence = make_question("q2", "factual", [])
    multi_doc_with_one_document = make_question("q3", "multi_doc")

    problems = validate_dataset(
        [unanswerable_with_evidence, factual_without_evidence, multi_doc_with_one_document],
        DOCUMENTS,
    )

    assert len(problems) == 3
    assert "must have no evidence" in problems[0]
    assert "needs at least one evidence quote" in problems[1]
    assert "at least two documents" in problems[2]


def test_project_dataset_file_is_well_formed():
    # Checks the real committed dataset's format. Quote checks need the
    # ingested documents, so they run in the validate_dataset command instead.
    dataset_path = Path(__file__).parents[2] / "evals" / "datasets" / "questions.jsonl"

    questions = load_eval_questions(dataset_path)

    assert len(questions) > 0
    ids = set()
    for question in questions:
        ids.add(question.question_id)
    assert len(ids) == len(questions)


# --- locate_evidence ----------------------------------------------------------------


def test_evidence_is_located_at_exact_positions():
    question = make_question(evidence=[EvidenceQuote("a.pdf", "BM25 uses keywords.")])

    spans = locate_evidence(question, DOCUMENTS)

    assert len(spans) == 1
    assert spans[0].doc_id == "a.pdf"
    assert DOCUMENTS["a.pdf"][spans[0].start:spans[0].end] == "BM25 uses\nkeywords."


def test_unanswerable_question_has_no_spans():
    assert locate_evidence(make_question("q9", "unanswerable", []), DOCUMENTS) == []


def test_ambiguous_quote_cannot_be_located():
    question = make_question(evidence=[EvidenceQuote("a.pdf", "BM25")])

    with pytest.raises(ValueError, match="found 2 times"):
        locate_evidence(question, DOCUMENTS)
