"""Storage interfaces.

The vector index is reachable only through :class:`VectorIndexService`, so the
storage technology chosen in Chapter Three can be substituted without touching
retrieval, ranking or profile code (Chapter Three, Section 3.2.3).
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..models import ContentItem, ContentType, Interaction, User, UserProfile


class VectorIndexService(Protocol):
    """Stores content embeddings and retrieves by vector similarity."""

    def index_content(self, item: ContentItem) -> None:
        """Insert or update a single item and its embedding."""
        ...

    def index_many(self, items: Sequence[ContentItem]) -> int:
        """Bulk insert. Returns the number of items newly indexed."""
        ...

    def query_similar(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        content_types: Sequence[ContentType] | None = None,
        exclude_ids: Sequence[str] | None = None,
    ) -> list[tuple[ContentItem, float]]:
        """Return up to ``top_k`` items with cosine similarity, most similar first.

        ``content_types`` is the metadata filter that motivated the choice of a
        store supporting filtering alongside similarity search.
        """
        ...

    def get(self, content_id: str) -> ContentItem | None: ...

    def exists_by_url(self, source_url: str) -> bool:
        """Deduplication check: the same URL must not be indexed twice."""
        ...

    def count(self) -> int: ...


class UserRepository(Protocol):
    def save_user(self, user: User) -> None: ...

    def get_user(self, user_id: str) -> User | None: ...

    def get_profile(self, user_id: str) -> UserProfile | None: ...

    def save_profile(self, profile: UserProfile) -> None: ...


class InteractionRepository(Protocol):
    def record(self, interaction: Interaction) -> None: ...

    def for_user(self, user_id: str, limit: int = 100) -> list[Interaction]: ...

    def count_for_user(self, user_id: str) -> int: ...