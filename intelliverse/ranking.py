"""Personalised ranking.

Implements step 4 of the algorithm design (Chapter Three, Section 3.2.2.5). The
final score of a candidate is

    score = w_sem * sim(query, item)
          + w_prof * sim(interest, item)
          + w_fresh * freshness(item)
          + w_type * typePreference(item)

Keeping ranking separate from retrieval is deliberate and is the distinction
Chapter One insists on: retrieval decides which content is *about* the query,
ranking decides which of that content is right *for this user*. The two are
computed by different components against different inputs, and the evaluation in
Chapter Four compares them by disabling the second.

All similarity terms are clamped to [0, 1]. Cosine similarity is defined on
[-1, 1], but a negative similarity means "unrelated", not "worse than
unrelated", and allowing negative contributions would let a strongly disliked
topic drag a candidate below content the user has no opinion about at all.
"""

from __future__ import annotations

import math

import numpy as np

from .config import RankingWeights
from .embedding import cosine_similarity
from .models import ContentItem, ScoredContent, UserProfile


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def freshness_score(item: ContentItem, half_life_days: float) -> float:
    """Exponential decay in the item's age, in [0, 1].

    Freshness is weighted lightly by default: IntelliVerse is a discovery
    system, not a news feed, and a well-matched explanation from three years ago
    is usually more useful than a poorly matched one from yesterday.
    """
    if half_life_days <= 0:
        return 0.0
    return float(2.0 ** (-item.age_days() / half_life_days))


class RecommendationEngine:
    def __init__(
        self,
        weights: RankingWeights | None = None,
        *,
        freshness_half_life_days: float = 90.0,
    ) -> None:
        self.weights = weights or RankingWeights()
        self.freshness_half_life_days = freshness_half_life_days

    def compute_score(
        self,
        item: ContentItem,
        semantic_similarity: float,
        profile: UserProfile | None,
        *,
        personalise: bool,
    ) -> ScoredContent:
        interest = profile.get_interest_embedding() if profile else None
        active = personalise and interest is not None and item.embedding is not None
        weights = self.weights if active else self.weights.for_cold_start()

        semantic = _clamp01(semantic_similarity)
        freshness = freshness_score(item, self.freshness_half_life_days)

        profile_similarity = 0.0
        type_preference = 0.0
        if active:
            profile_similarity = _clamp01(
                cosine_similarity(interest, np.asarray(item.embedding, dtype=np.float32))
            )
            type_preference = _clamp01(profile.content_type_preference(item.content_type))

        score = (
            weights.semantic * semantic
            + weights.profile * profile_similarity
            + weights.freshness * freshness
            + weights.content_type * type_preference
        )

        return ScoredContent(
            item=item,
            score=score,
            semantic_similarity=semantic,
            profile_similarity=profile_similarity,
            freshness=freshness,
            type_preference=type_preference,
            personalised=active,
        )

    def rank_results(
        self,
        candidates: list[tuple[ContentItem, float]],
        profile: UserProfile | None,
        *,
        personalise: bool = True,
        limit: int | None = None,
    ) -> list[ScoredContent]:
        """Score and order candidates.

        Setting ``personalise=False`` yields the non-personalised semantic
        baseline that Chapter Four's comparison requires — the same retrieval
        set, ordered without any user signal.
        """
        scored = [
            self.compute_score(item, similarity, profile, personalise=personalise)
            for item, similarity in candidates
        ]
        # Tie-break on semantic similarity so ordering is deterministic when
        # scores coincide, which matters for reproducible evaluation runs.
        scored.sort(key=lambda s: (s.score, s.semantic_similarity), reverse=True)
        return scored[:limit] if limit else scored
