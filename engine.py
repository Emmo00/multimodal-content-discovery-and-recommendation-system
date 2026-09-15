"""The IntelliVerse engine.

Corresponds to ``QueryInterface`` in the Chapter Three class diagram and
implements the workflow of the activity diagram (Figure 3.2): embed the query,
retrieve candidates, check coverage, discover and re-retrieve if coverage is
insufficient, then rank against the user's profile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from .config import Settings
from .discovery import DiscoveryService
from .embedding import EmbeddingService
from .models import (
    ContentItem,
    ContentType,
    Interaction,
    InteractionType,
    ScoredContent,
    User,
)
from .profile import UserProfileService
from .ranking import RecommendationEngine
from .sources.base import ContentSource
from .storage.base import InteractionRepository, UserRepository, VectorIndexService

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    """Results plus the provenance needed to explain and evaluate them."""

    query: str
    results: list[ScoredContent] = field(default_factory=list)
    personalised: bool = False
    discovery_triggered: bool = False
    newly_indexed: int = 0
    candidates_considered: int = 0

    def __len__(self) -> int:
        return len(self.results)


class IntelliVerse:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        index: VectorIndexService,
        user_repository: UserRepository,
        interaction_repository: InteractionRepository,
        *,
        sources: Sequence[ContentSource] = (),
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.embedding_service = embedding_service
        self.index = index
        self.users = user_repository
        self.interactions = interaction_repository

        self.discovery = DiscoveryService(sources, embedding_service, index)
        self.profiles = UserProfileService(
            user_repository, index, half_life_days=self.settings.interest_half_life_days
        )
        self.ranker = RecommendationEngine(
            self.settings.weights,
            freshness_half_life_days=self.settings.freshness_half_life_days,
        )

    # -- users ------------------------------------------------------------

    def register_user(self, name: str, email: str) -> User:
        user = User(name=name, email=email)
        self.users.save_user(user)
        self.profiles.get_profile(user.user_id)  # materialise an empty profile
        return user

    # -- coverage ---------------------------------------------------------

    def _coverage_is_sufficient(self, candidates: list[tuple[ContentItem, float]]) -> bool:
        """Decide whether the index already answers this query well enough.

        Two conditions, either of which forces discovery: too few candidates, or
        a best candidate too weak to be plausibly relevant. The second matters
        more than the first — a large index will almost always return *some*
        results, so count alone would never trigger discovery once the corpus
        had grown.
        """
        if len(candidates) < self.settings.min_candidates:
            return False
        return candidates[0][1] >= self.settings.min_top_similarity

    # -- search -----------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: str | None = None,
        *,
        content_types: Sequence[ContentType] | None = None,
        limit: int | None = None,
        personalise: bool = True,
        allow_discovery: bool = True,
    ) -> SearchResponse:
        """Run the full query-to-recommendation workflow.

        ``personalise=False`` produces the non-personalised semantic baseline
        used for comparison in evaluation; ``allow_discovery=False`` pins the
        corpus, which a controlled evaluation requires so that the index does
        not change beneath successive queries.
        """
        limit = limit or self.settings.result_limit
        response = SearchResponse(query=query)

        query_vector = self.embedding_service.embed(query)
        candidates = self.index.query_similar(
            query_vector, self.settings.top_k, content_types=content_types
        )

        if allow_discovery and not self._coverage_is_sufficient(candidates):
            logger.info("Coverage insufficient for %r; triggering discovery", query)
            response.discovery_triggered = True
            result = self.discovery.discover_and_index(
                query, limit=self.settings.discovery_limit
            )
            response.newly_indexed = result.indexed
            if result.indexed:
                candidates = self.index.query_similar(
                    query_vector, self.settings.top_k, content_types=content_types
                )

        # Discard candidates with no meaningful semantic relation to the query,
        # so that ranking reorders relevant content rather than promoting
        # irrelevant content on freshness or profile fit alone.
        candidates = [
            (item, similarity)
            for item, similarity in candidates
            if similarity >= self.settings.min_result_similarity
        ]

        response.candidates_considered = len(candidates)
        if not candidates:
            return response

        profile = self.profiles.get_profile(user_id) if user_id else None
        personalise_now = (
            personalise and profile is not None and self.profiles.is_personalisable(profile)
        )

        response.results = self.ranker.rank_results(
            candidates, profile, personalise=personalise_now, limit=limit
        )
        response.personalised = personalise_now
        return response

    # -- feedback ---------------------------------------------------------

    def record_interaction(
        self,
        user_id: str,
        content_id: str,
        interaction_type: InteractionType | str,
    ) -> Interaction:
        """Record a signal and fold it into the user's profile immediately.

        The profile is updated synchronously so that the effect of feedback is
        visible on the user's very next query, which is the continuous-
        improvement behaviour of Chapter One stage 8.
        """
        if isinstance(interaction_type, str):
            interaction_type = InteractionType(interaction_type)

        interaction = Interaction(
            user_id=user_id, content_id=content_id, interaction_type=interaction_type
        )
        self.interactions.record(interaction)
        self.profiles.update_from_interaction(interaction)
        return interaction

    # -- corpus -----------------------------------------------------------

    def ingest(self, items: Sequence[ContentItem]) -> int:
        """Add a supplied collection to the index. Returns the number indexed."""
        return self.discovery.ingest(items).indexed

    def corpus_size(self) -> int:
        return self.index.count()


def build_default_engine(
    *,
    settings: Settings | None = None,
    sources: Sequence[ContentSource] = (),
    allow_embedding_fallback: bool = True,
) -> IntelliVerse:
    """Assemble an engine with in-memory storage.

    Convenience wiring for development, demonstration and tests. Production
    deployment substitutes the Postgres backends; see ``README.md``.
    """
    from .embedding import build_embedding_service
    from .storage.memory import (
        InMemoryInteractionRepository,
        InMemoryUserRepository,
        InMemoryVectorIndex,
    )

    settings = settings or Settings()
    embedding_service = build_embedding_service(
        settings.embedding_model, allow_fallback=allow_embedding_fallback
    )
    return IntelliVerse(
        embedding_service=embedding_service,
        index=InMemoryVectorIndex(embedding_service.dimension),
        user_repository=InMemoryUserRepository(),
        interaction_repository=InMemoryInteractionRepository(),
        sources=sources,
        settings=settings,
    )
