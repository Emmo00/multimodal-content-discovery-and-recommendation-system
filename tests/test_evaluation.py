"""Tests for the evaluation metrics.

Metric implementations are checked against hand-computed values, because a
silently wrong metric would corrupt every number reported in Chapter Four.
"""

from __future__ import annotations

import math

import pytest

from intelliverse.evaluation import (
    Evaluator,
    Query,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"a", "c", "e"}


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 1) == 1.0        # a
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5        # a, b
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(3 / 5)


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0


def test_recall_is_zero_when_nothing_is_relevant():
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_precision_handles_short_result_lists():
    assert precision_at_k(["a"], RELEVANT, 10) == 1.0


def test_dcg_matches_manual_calculation():
    grades = {"a": 3.0, "b": 0.0, "c": 2.0}
    expected = 3.0 / math.log2(2) + 0.0 / math.log2(3) + 2.0 / math.log2(4)
    assert dcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(expected)


def test_ndcg_is_one_for_ideal_ordering():
    grades = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)


def test_ndcg_penalises_reversed_ordering():
    grades = {"a": 3.0, "b": 2.0, "c": 1.0}
    ideal = ndcg_at_k(["a", "b", "c"], grades, 3)
    reversed_order = ndcg_at_k(["c", "b", "a"], grades, 3)
    assert reversed_order < ideal


def test_ndcg_is_zero_when_no_grades_are_positive():
    assert ndcg_at_k(["a", "b"], {"a": 0.0}, 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], RELEVANT) == 0.5
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0


def test_evaluator_produces_both_conditions_over_same_queries():
    from tests.test_intelliverse import CORPUS, make_item
    from intelliverse import IntelliVerse, InteractionType, Settings
    from intelliverse.embedding import HashingEmbeddingService
    from intelliverse.storage.memory import (
        InMemoryInteractionRepository,
        InMemoryUserRepository,
        InMemoryVectorIndex,
    )

    service = HashingEmbeddingService(dimension=256)
    engine = IntelliVerse(
        embedding_service=service,
        index=InMemoryVectorIndex(service.dimension),
        user_repository=InMemoryUserRepository(),
        interaction_repository=InMemoryInteractionRepository(),
        settings=Settings(min_top_similarity=0.05),
    )
    engine.ingest(CORPUS)

    user = engine.register_user("Test", "t@example.test")
    cooking = [i for i in engine.index.all_items() if "cooking" in i.topics]
    for item in cooking:
        for signal in (InteractionType.LIKE, InteractionType.SAVE, InteractionType.COMPLETION):
            engine.record_interaction(user.user_id, item.content_id, signal)

    queries = [
        Query(
            query_id="q1",
            text="learning techniques",
            user_id=user.user_id,
            relevance={item.content_id: 1.0 for item in cooking},
        )
    ]

    reports = Evaluator(engine, cutoffs=(1, 3, 5)).compare(queries)

    assert set(reports) == {"baseline", "personalised"}
    assert reports["baseline"].per_query[0].personalised is False
    assert reports["personalised"].per_query[0].personalised is True
    # Both summaries must expose the same metric keys for a valid comparison.
    assert reports["baseline"].summary((1, 3, 5)).keys() == reports[
        "personalised"
    ].summary((1, 3, 5)).keys()