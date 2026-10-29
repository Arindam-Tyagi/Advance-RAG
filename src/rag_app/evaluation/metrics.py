"""Retrieval metrics: how good is a ranked list of retrieved chunks?

Relevance rule (decided in Milestone 2):
    A retrieved chunk is RELEVANT to a question if its character range
    overlaps at least one of the question's evidence spans in the same
    document. Even a one-character overlap counts.

Because relevance is defined by character positions, the same evaluation
labels work for every chunking strategy and chunk size.

All functions take chunks in rank order (best first) and are "pure":
they only compute numbers from their inputs, with no model or database.
"""

import math


def check_k(k: int) -> None:
    """Raise ValueError if k is not a positive whole number."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def chunk_overlaps_span(chunk, span) -> bool:
    """True if a chunk and an evidence span share at least one character.

    Both ranges are "end-exclusive": a chunk covering 0-10 and a span
    starting at 10 touch but do not overlap.
    """
    if chunk.doc_id != span.doc_id:
        return False
    return chunk.start_char < span.end and span.start < chunk.end_char


def chunk_is_relevant(chunk, spans: list) -> bool:
    """True if the chunk overlaps any of the evidence spans."""
    for span in spans:
        if chunk_overlaps_span(chunk, span):
            return True
    return False


def relevance_flags(ranked_chunks: list, spans: list) -> list:
    """Mark each retrieved chunk as relevant (1) or not (0), in rank order.

    Example: [0, 1, 0] means only the second-ranked chunk is relevant.
    """
    flags = []
    for chunk in ranked_chunks:
        if chunk_is_relevant(chunk, spans):
            flags.append(1)
        else:
            flags.append(0)
    return flags


def count_relevant_chunks(all_chunks: list, spans: list) -> int:
    """Count how many chunks in the WHOLE index are relevant to a question.

    Needed by nDCG to know what a perfect ranking would look like.
    """
    count = 0
    for chunk in all_chunks:
        if chunk_is_relevant(chunk, spans):
            count = count + 1
    return count


def recall_at_k(ranked_chunks: list, spans: list, k: int) -> float:
    """Share of the evidence spans found in the top k chunks.

    Recall is counted per EVIDENCE SPAN, not per chunk: a span counts as
    found if any top-k chunk overlaps it. Counting chunks instead would
    reward strategies that cut one quote into many pieces.

    Example: 2 evidence spans, top 5 chunks overlap only one -> 0.5.

    Raises:
        ValueError: If k < 1 or there are no spans (an unanswerable question
            has nothing to recall).
    """
    check_k(k)
    if len(spans) == 0:
        raise ValueError("recall is undefined for a question without evidence")

    top_chunks = ranked_chunks[:k]
    found = 0
    for span in spans:
        for chunk in top_chunks:
            if chunk_overlaps_span(chunk, span):
                found = found + 1
                break
    return found / len(spans)


def precision_at_k(flags: list, k: int) -> float:
    """Share of the top k positions that hold a relevant chunk.

    Divides by k even if fewer than k chunks were returned, so returning
    fewer results is not rewarded.

    Example: flags [0, 1, 0, 1, 0] -> precision@5 = 2 / 5 = 0.4.
    """
    check_k(k)
    relevant_in_top_k = 0
    for flag in flags[:k]:
        relevant_in_top_k = relevant_in_top_k + flag
    return relevant_in_top_k / k


def reciprocal_rank(flags: list, k: int) -> float:
    """1 divided by the rank of the first relevant chunk in the top k.

    Rank 1 -> 1.0, rank 2 -> 0.5, rank 4 -> 0.25, none in top k -> 0.0.
    Averaging this over all questions gives MRR (mean reciprocal rank).
    """
    check_k(k)
    for position in range(min(k, len(flags))):
        if flags[position] == 1:
            rank = position + 1
            return 1.0 / rank
    return 0.0


def dcg(flags: list, k: int) -> float:
    """Discounted cumulative gain of the top k flags.

    Each relevant chunk adds 1 / log2(rank + 1): rank 1 adds 1.0,
    rank 2 adds 0.63, rank 3 adds 0.5, ... so relevant chunks near the top
    are worth more than relevant chunks further down.
    """
    total = 0.0
    for position in range(min(k, len(flags))):
        rank = position + 1
        total = total + flags[position] / math.log2(rank + 1)
    return total


def ndcg_at_k(flags: list, k: int, total_relevant: int) -> float:
    """Normalized DCG: actual DCG divided by the best possible DCG.

    The best possible ranking puts every relevant chunk at the top, so its
    DCG uses min(k, total_relevant) ones. The result is between 0 and 1,
    where 1 means no better ordering of the top k exists.

    Example: flags [0, 1, 0, 1, 0], k=5, 2 relevant chunks in the index:
        DCG  = 1/log2(3) + 1/log2(5) = 0.631 + 0.431 = 1.062
        IDCG = 1/log2(2) + 1/log2(3) = 1.000 + 0.631 = 1.631
        nDCG = 1.062 / 1.631 = 0.651

    Raises:
        ValueError: If k < 1 or total_relevant < 1.
    """
    check_k(k)
    if total_relevant < 1:
        raise ValueError("nDCG is undefined when no chunk in the index is relevant")

    ideal_flags = []
    for _ in range(min(k, total_relevant)):
        ideal_flags.append(1)
    return dcg(flags, k) / dcg(ideal_flags, k)
