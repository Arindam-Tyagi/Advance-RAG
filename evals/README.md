# Evaluation dataset

`datasets/questions.jsonl` holds the fixed set of questions used to compare
every retrieval and generation configuration. Using the **same** questions
for every experiment is what makes comparisons fair.

> **Status:** all questions are `draft`. A question becomes `verified` only
> after a person has checked the question, answer and evidence against the
> paper. No evaluation results exist yet; metrics arrive in Milestone 2.

## Format

One JSON object per line:

```json
{
  "question_id": "q005",
  "question": "By how much does Dense Passage Retrieval outperform BM25 ...?",
  "reference_answer": "By 9% to 19% absolute.",
  "question_type": "factual",
  "evidence": [
    {"doc_id": "arxiv/2004.04906v3_dense_passage_retrieval.pdf",
     "quote": "greatly by 9%-19% absolute in terms of top-20 passage retrieval accuracy"}
  ],
  "status": "draft",
  "notes": "optional"
}
```

| Field | Meaning |
|---|---|
| `question_id` | Unique ID, never reused. |
| `question` | Phrased the way a user would ask it. |
| `reference_answer` | Short correct answer written by a person. Used later to judge answer correctness. |
| `question_type` | `factual` (one document), `multi_doc` (needs two or more documents), or `unanswerable` (the corpus does not contain the answer). |
| `evidence` | Exact quotes that support the answer, each with its `doc_id`. Empty for `unanswerable`. |
| `status` | `draft` or `verified`. |
| `notes` | Optional, e.g. how unanswerability was checked. |

## Why evidence is a quote, not a chunk ID

Chunk IDs change every time the chunking strategy or size changes. A quote
from the document text does not. In Milestone 2, a retrieved chunk will count
as **relevant** when its character range overlaps an evidence quote's range in
the document. The same labels then work for fixed chunks, recursive chunks,
and any future strategy.

## Rules for writing evidence quotes

- Copy quotes from `data/processed/documents.jsonl` (the cleaned text the
  pipeline actually uses), not from the PDF viewer. PDF extraction can drop
  spaces (e.g. `encoderEP`), so the PDF and the extracted text can differ.
- Line breaks and repeated spaces do not need to match: one space in a quote
  matches any whitespace in the document.
- Each quote must appear **exactly once** in its document. Make it longer if
  the validator reports it appears several times.
- Quote only what supports the answer; a full sentence is usually right.

## Why unanswerable questions are included

A good RAG system should say "I don't know" when the documents do not
contain the answer, instead of making one up. Unanswerable questions test
that behavior. Each one records in `notes` how its absence was checked.
A text search is evidence of absence, not proof; a person should confirm it.

## Validating the dataset

After running the ingest command:

```powershell
uv run python -m rag_app.evaluation.validate_dataset
```

It checks the format, the type rules, and that every quote is found exactly
once in its document. Exit code 0 means the dataset is valid.

## Verifying a draft question

1. Open the paper (see `docs/corpus.md`) at the quoted passage.
2. Confirm the question is clear and has one correct answer.
3. Confirm `reference_answer` is correct and complete.
4. Change `status` to `verified`.
