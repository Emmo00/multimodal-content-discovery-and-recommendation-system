"""Content discovery and ingestion.

Implements the pipeline of Chapter One's stages 1-3: search external sources,
normalise what comes back, embed it, and add it to the index.

Deduplication happens on ``source_url`` before embedding rather than after,
because embedding is the expensive step and rediscovering already-indexed
content is the common case once a corpus has been running for a while.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from .embedding import EmbeddingService
from .models import ContentItem
from .sources.base import ContentSource
from .storage.base import VectorIndexService

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Outcome of one discovery run, for logging and evaluation."""

    retrieved: int = 0
    indexed: int = 0
    duplicates: int = 0
    failed_sources: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failed_sources is None:
            self.failed_sources = []


class DiscoveryService:
    def __init__(
        self,
        sources: Sequence[ContentSource],
        embedding_service: EmbeddingService,
        index: VectorIndexService,
    ) -> None:
        self.sources = list(sources)
        self.embedding_service = embedding_service
        self.index = index

    def search_external_sources(self, query: str, limit: int = 10) -> list[ContentItem]:
        """Query every configured source, tolerating individual failures."""
        discovered: list[ContentItem] = []
        per_source = max(1, limit // max(1, len(self.sources)))

        for source in self.sources:
            if hasattr(source, "available") and not source.available:
                logger.debug("Skipping unavailable source: %s", source.name)
                continue
            try:
                results = source.search(query, limit=per_source)
                discovered.extend(results)
                logger.info("Source %s returned %d items", source.name, len(results))
            except Exception:
                # One source failing must not deny the user their results.
                logger.exception("Source %s failed for query %r", source.name, query)
        return discovered

    def discover_and_index(self, query: str, limit: int = 25) -> DiscoveryResult:
        """Full ingestion pass: search, deduplicate, embed, index."""
        result = DiscoveryResult()

        candidates = self.search_external_sources(query, limit=limit)
        result.retrieved = len(candidates)
        if not candidates:
            return result

        fresh: list[ContentItem] = []
        seen_urls: set[str] = set()
        for item in candidates:
            if item.source_url in seen_urls or self.index.exists_by_url(item.source_url):
                result.duplicates += 1
                continue
            seen_urls.add(item.source_url)
            fresh.append(item)

        if not fresh:
            return result

        # Batch the embedding call: one forward pass over N texts is markedly
        # cheaper than N passes over one.
        vectors = self.embedding_service.embed_batch(
            [item.build_semantic_text() for item in fresh]
        )
        for item, vector in zip(fresh, vectors):
            item.embedding = vector

        result.indexed = self.index.index_many(fresh)
        logger.info(
            "Discovery for %r: retrieved=%d indexed=%d duplicates=%d",
            query, result.retrieved, result.indexed, result.duplicates,
        )
        return result

    def ingest(self, items: Sequence[ContentItem]) -> DiscoveryResult:
        """Embed and index a supplied collection.

        Used to seed a corpus from a dataset rather than a live source, which is
        how the controlled evaluation collection of Chapter Four is loaded.
        """
        result = DiscoveryResult(retrieved=len(items))
        fresh = [i for i in items if not self.index.exists_by_url(i.source_url)]
        result.duplicates = len(items) - len(fresh)
        if not fresh:
            return result

        vectors = self.embedding_service.embed_batch(
            [item.build_semantic_text() for item in fresh]
        )
        for item, vector in zip(fresh, vectors):
            item.embedding = vector
        result.indexed = self.index.index_many(fresh)
        return result