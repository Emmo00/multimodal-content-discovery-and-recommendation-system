"""Configuration for IntelliVerse.

Every value here that affects retrieval or ranking quality is an *evaluation
parameter*: Chapter Three deliberately left the coverage threshold and the
ranking weights undetermined, to be fixed empirically. They are gathered in one
place so that Chapter Four can sweep them and report the values actually used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RankingWeights:
    """Relative contribution of each ranking signal.

    The four weights are normalised to sum to 1 on construction. On cold start
    the profile and content-type terms are inapplicable, and the ranker
    redistributes their weight to the semantic term (see
    :meth:`for_cold_start`), so that ranking degrades gracefully to pure
    semantic relevance rather than to an arbitrary partial score.
    """

    semantic: float = 0.60
    profile: float = 0.25
    freshness: float = 0.05
    content_type: float = 0.10

    def __post_init__(self) -> None:
        total = self.semantic + self.profile + self.freshness + self.content_type
        if total <= 0:
            raise ValueError("Ranking weights must sum to a positive value")
        self.semantic /= total
        self.profile /= total
        self.freshness /= total
        self.content_type /= total

    def for_cold_start(self) -> "RankingWeights":
        """Weights for a user with no usable interest profile."""
        return RankingWeights(
            semantic=self.semantic + self.profile + self.content_type,
            profile=0.0,
            freshness=self.freshness,
            content_type=0.0,
        )


@dataclass
class Settings:
    # --- Embedding -------------------------------------------------------
    #: Sentence-BERT model selected in Chapter Three, Section 3.2.2.2.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # --- Retrieval -------------------------------------------------------
    #: Number of candidates drawn from the vector index before ranking.
    #: Kept above the display count so the ranker has room to reorder.
    top_k: int = 50
    #: Results returned to the user.
    result_limit: int = 10

    # --- Coverage check (triggers external discovery) --------------------
    #: Discovery fires if fewer than this many candidates come back...
    min_candidates: int = 5
    #: ...or if the best candidate is less similar than this.
    #: Cosine similarity on MiniLM embeddings; an initial value to be tuned.
    min_top_similarity: float = 0.35
    #: Minimum similarity required for a candidate to be returned.
    min_result_similarity: float = 0.15
    #: Ceiling on items ingested per discovery run, to bound query latency.
    discovery_limit: int = 25

    # --- Ranking ---------------------------------------------------------
    weights: RankingWeights = field(default_factory=RankingWeights)
    #: Days after which a freshness score halves.
    freshness_half_life_days: float = 90.0
    #: Days after which a past interaction's contribution to the interest
    #: vector halves. Governs how quickly the profile follows changing
    #: interests: shorter is more responsive, longer is more stable.
    interest_half_life_days: float = 30.0

    # --- Storage ---------------------------------------------------------
    database_url: str | None = field(
        default_factory=lambda: os.environ.get("INTELLIVERSE_DATABASE_URL")
    )

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls()
        if v := os.environ.get("INTELLIVERSE_TOP_K"):
            settings.top_k = int(v)
        if v := os.environ.get("INTELLIVERSE_MIN_TOP_SIMILARITY"):
            settings.min_top_similarity = float(v)
        if v := os.environ.get("INTELLIVERSE_MIN_RESULT_SIMILARITY"):
            settings.min_result_similarity = float(v)
        if v := os.environ.get("INTELLIVERSE_EMBEDDING_MODEL"):
            settings.embedding_model = v
        return settings