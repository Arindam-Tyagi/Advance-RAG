"""Tests for retrieval metrics. Every expected value is worked out by hand."""

import pytest

from rag_app.evaluation.dataset import EvidenceSpan
from rag_app.evaluation.metrics import (
    chunk_is_relevant,
    chunk_overlaps_span,
    count_relevant_chunks,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
)
from rag_app.models import Chunk


def make_chunk(doc_id: str, start: int, end: int) -> Chunk:
    return Chunk(f"{doc_id}::{start}", doc_id, "x" * (end - start), start, end, {})


# --- overlap rule -------------------------------------------------------------------


def test_overlapping_ranges_in_same_document():
    span = EvidenceSpan("a", 10, 20)

    assert chunk_overlaps_span(make_chunk("a", 0, 11), span)  # shares character 10
    assert chunk_overlaps_span(make_chunk("a", 19, 30), span)  # shares character 19
    assert chunk_overlaps_span(make_chunk("a", 12, 15), span)  # inside the span
    assert chunk_overlaps_span(make_chunk("a", 0, 50), span)  # contains the span


def test_touching_ranges_do_not_overlap():
    # End positions are exclusive: 0-10 covers characters 0..9.
    span = EvidenceSpan("a", 10, 20)

    assert not chunk_overlaps_span(make_chunk("a", 0, 10), span)
    assert not chunk_overlaps_span(make_chunk("a", 20, 30), span)


def test_same_positions_in_another_document_do_not_overlap():
    assert not chunk_overlaps_span(make_chunk("b", 10, 20), EvidenceSpan("a", 10, 20))


def test_relevance_flags_and_counting():
    spans = [EvidenceSpan("a", 10, 20)]
    ranked = [make_chunk("a", 30, 40), make_chunk("a", 5, 15), make_chunk("b", 10, 20)]

    assert relevance_flags(ranked, spans) == [0, 1, 0]
    assert chunk_is_relevant(ranked[1], spans)
    assert count_relevant_chunks(ranked, spans) == 1


# --- recall@k (per evidence span) ------------------------------------------------------


SPANS = [EvidenceSpan("a", 10, 20), EvidenceSpan("b", 0, 5)]
RANKED = [
    make_chunk("a", 0, 15),  # rank 1: overlaps span in "a"
    make_chunk("a", 30, 40),  # rank 2: not relevant
    make_chunk("b", 3, 8),  # rank 3: overlaps span in "b"
]


def test_recall_counts_evidence_spans_found():
    assert recall_at_k(RANKED, SPANS, k=1) == 0.5  # 1 of 2 spans
    assert recall_at_k(RANKED, SPANS, k=2) == 0.5
    assert recall_at_k(RANKED, SPANS, k=3) == 1.0  # both spans


def test_recall_counts_a_span_once_even_if_two_chunks_overlap_it():
    ranked = [make_chunk("a", 0, 12), make_chunk("a", 12, 25)]  # both overlap "a" 10-20

    assert recall_at_k(ranked, SPANS, k=2) == 0.5  # span "b" still missing


def test_recall_is_undefined_without_evidence():
    with pytest.raises(ValueError, match="without evidence"):
        recall_at_k(RANKED, [], k=3)


# --- precision@k ------------------------------------------------------------------------


def test_precision_at_k():
    flags = [0, 1, 0, 1, 0]

    assert precision_at_k(flags, 1) == 0.0
    assert precision_at_k(flags, 2) == 0.5
    assert precision_at_k(flags, 5) == 0.4


def test_precision_divides_by_k_even_with_fewer_results():
    assert precision_at_k([1], 5) == 0.2


# --- reciprocal rank -----------------------------------------------------------------------


def test_reciprocal_rank():
    assert reciprocal_rank([1, 0, 0], k=10) == 1.0
    assert reciprocal_rank([0, 1, 1], k=10) == 0.5
    assert reciprocal_rank([0, 0, 0, 1], k=10) == 0.25
    assert reciprocal_rank([0, 0, 0], k=10) == 0.0


def test_reciprocal_rank_ignores_hits_after_k():
    assert reciprocal_rank([0, 0, 1], k=2) == 0.0


# --- nDCG@k ----------------------------------------------------------------------------------


def test_ndcg_worked_example():
    # DCG  = 1/log2(3) + 1/log2(5) = 0.63093 + 0.43068 = 1.06161
    # IDCG = 1/log2(2) + 1/log2(3) = 1.00000 + 0.63093 = 1.63093
    # nDCG = 1.06161 / 1.63093 = 0.65092
    assert ndcg_at_k([0, 1, 0, 1, 0], k=5, total_relevant=2) == pytest.approx(0.65092, abs=1e-5)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([1, 1, 0, 0], k=4, total_relevant=2) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k([0, 0, 0], k=3, total_relevant=4) == 0.0


def test_ndcg_ideal_is_limited_to_k_positions():
    # 5 relevant chunks exist but k=2, so the ideal is two hits at ranks 1-2.
    # Actual: one hit at rank 1 -> DCG = 1.0; IDCG = 1 + 0.63093 = 1.63093.
    assert ndcg_at_k([1, 0], k=2, total_relevant=5) == pytest.approx(1.0 / 1.63093, abs=1e-5)


def test_ndcg_needs_at_least_one_relevant_chunk():
    with pytest.raises(ValueError, match="no chunk in the index is relevant"):
        ndcg_at_k([0, 0], k=2, total_relevant=0)


# --- shared input check ------------------------------------------------------------------------


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be at least 1"):
        precision_at_k([1], 0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        reciprocal_rank([1], 0)
