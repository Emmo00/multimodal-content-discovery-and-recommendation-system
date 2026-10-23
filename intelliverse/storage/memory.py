"""In-memory storage backend.

Exact brute-force search over a NumPy matrix. At the corpus sizes this project
evaluates, exact search is fast enough and has a useful property the approximate
index lacks: it establishes the ground-truth ranking that pgvector's HNSW index
is an approximation of, so the two can be compared directly if recall of the
approximate index ever comes into question.
"""

from __future__ import annotations

import numpy as np

from ..models import ContentItem, ContentType, Interaction, User, UserProfile


class InMemoryVectorIndex:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self._items: dict[str, ContentItem] = {}
        self._urls: set[str] = set()
        self._matrix: np.ndarray | None = None
        self._order: list[str] = []
        self._dirty = True

    def index_content(self, item: ContentItem) -> None:
        if item.embedding is None:
            raise ValueError(f"Item '{item.title}' has no embedding; embed before indexing")
        self._items[item.content_id] = item
        self._urls.add(item.source_url)
        self._dirty = True

    def index_many(self, items) -> int:
        added = 0
        for item in items:
            if item.content_id not in self._items:
                added += 1
            self.index_content(item)
        return added

    def _rebuild(self) -> None:
        if not self._items:
            self._matrix, self._order = None, []
        else:
            self._order = list(self._items)
            self._matrix = np.vstack(
                [np.asarray(self._items[cid].embedding, dtype=np.float32) for cid in self._order]
            )
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            self._matrix = self._matrix / np.maximum(norms, 1e-9)
        self._dirty = False

    def query_similar(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        content_types=None,
        exclude_ids=None,
    ) -> list[tuple[ContentItem, float]]:
        if self._dirty:
            self._rebuild()
        if self._matrix is None:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return []
        similarities = self._matrix @ (q / norm)

        allowed_types = {ct.value for ct in content_types} if content_types else None
        excluded = set(exclude_ids or ())

        results: list[tuple[ContentItem, float]] = []
        # Sort descending, then filter, so that filtering never silently
        # truncates the candidate pool below top_k when matches exist deeper in
        # the ranking.
        for idx in np.argsort(-similarities):
            content_id = self._order[idx]
            if content_id in excluded:
                continue
            item = self._items[content_id]
            if allowed_types and item.content_type.value not in allowed_types:
                continue
            results.append((item, float(similarities[idx])))
            if len(results) >= top_k:
                break
        return results

    def get(self, content_id: str) -> ContentItem | None:
        return self._items.get(content_id)

    def exists_by_url(self, source_url: str) -> bool:
        return source_url in self._urls

    def count(self) -> int:
        return len(self._items)

    def all_items(self) -> list[ContentItem]:
        return list(self._items.values())


class InMemoryUserRepository:
    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._profiles: dict[str, UserProfile] = {}

    def save_user(self, user: User) -> None:
        self._users[user.user_id] = user

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def get_profile(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    def save_profile(self, profile: UserProfile) -> None:
        self._profiles[profile.user_id] = profile


class InMemoryInteractionRepository:
    def __init__(self) -> None:
        self._interactions: list[Interaction] = []

    def record(self, interaction: Interaction) -> None:
        self._interactions.append(interaction)

    def for_user(self, user_id: str, limit: int = 100) -> list[Interaction]:
        matching = [i for i in self._interactions if i.user_id == user_id]
        matching.sort(key=lambda i: i.timestamp, reverse=True)
        return matching[:limit]

    def count_for_user(self, user_id: str) -> int:
        return sum(1 for i in self._interactions if i.user_id == user_id)