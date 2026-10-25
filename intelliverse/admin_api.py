"""Administrative REST interface for the IntelliVerse demonstration UI."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Query

from .engine import IntelliVerse
from .models import ContentType

admin_app = FastAPI(
    title="IntelliVerse Admin API",
    description="Operational and evaluation endpoints for the IntelliVerse demo.",
    version="0.1.0",
)


def _get_engine() -> IntelliVerse:
    from .api import get_engine

    return get_engine()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


@admin_app.get("/health")
def admin_health(engine: IntelliVerse = Depends(_get_engine)) -> dict[str, Any]:
    settings = engine.settings
    return {
        "status": "ok",
        "corpus_size": engine.corpus_size(),
        "embedding_model": getattr(engine.embedding_service, "model_name", "unknown"),
        "embedding_dimension": engine.embedding_service.dimension,
        "storage_backend": type(engine.index).__name__,
        "source_count": len(engine.discovery.sources),
        "in_memory": type(engine.index).__name__.startswith("InMemory"),
        "settings": {
            "top_k": settings.top_k,
            "result_limit": settings.result_limit,
            "min_top_similarity": settings.min_top_similarity,
            "min_result_similarity": settings.min_result_similarity,
            "discovery_limit": settings.discovery_limit,
        },
    }


@admin_app.post("/discover")
def admin_discover(
    q: str = Query(min_length=1),
    limit: int = Query(default=25, ge=1, le=100),
    engine: IntelliVerse = Depends(_get_engine),
) -> dict[str, int]:
    result = engine.discovery.discover_and_index(q, limit=limit)
    return {
        "retrieved": result.retrieved,
        "indexed": result.indexed,
        "duplicates": result.duplicates,
        "corpus_size": engine.corpus_size(),
    }


@admin_app.get("/content")
def admin_content(
    query: str | None = None,
    content_type: ContentType | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    engine: IntelliVerse = Depends(_get_engine),
) -> dict[str, Any]:
    items = engine.index.all_items()
    query_terms = set(query.lower().split()) if query else set()
    filtered = []
    for item in items:
        if content_type and item.content_type != content_type:
            continue
        haystack = f"{item.title} {item.description} {' '.join(item.topics)}".lower()
        if query_terms and not query_terms.intersection(haystack.split()):
            continue
        filtered.append(
            {
                "content_id": item.content_id,
                "title": item.title,
                "description": item.description,
                "content_type": item.content_type.value,
                "source_platform": item.source_platform,
                "source_url": item.source_url,
                "topics": list(item.topics),
                "published_at": _iso(item.published_at),
                "discovered_at": _iso(item.discovered_at),
            }
        )
        if len(filtered) >= limit:
            break
    return {"count": len(filtered), "items": filtered}


@admin_app.get("/sources")
def admin_sources(engine: IntelliVerse = Depends(_get_engine)) -> dict[str, Any]:
    sources = []
    for source in engine.discovery.sources:
        available = getattr(source, "available", True)
        sources.append(
            {
                "name": source.name,
                "type": type(source).__name__,
                "available": bool(available),
                "feed_count": len(source.feeds) if hasattr(source, "feeds") else None,
            }
        )
    return {"sources": sources}


@admin_app.get("/config")
def admin_config(engine: IntelliVerse = Depends(_get_engine)) -> dict[str, Any]:
    settings = engine.settings
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "top_k": settings.top_k,
        "result_limit": settings.result_limit,
        "min_candidates": settings.min_candidates,
        "min_top_similarity": settings.min_top_similarity,
        "min_result_similarity": settings.min_result_similarity,
        "discovery_limit": settings.discovery_limit,
        "freshness_half_life_days": settings.freshness_half_life_days,
        "interest_half_life_days": settings.interest_half_life_days,
        "storage_backend": type(engine.index).__name__,
    }


@admin_app.get("/evaluation")
def admin_evaluation(
    q: str = Query(min_length=1),
    user_id: str | None = None,
    limit: int = Query(default=5, ge=1, le=50),
    engine: IntelliVerse = Depends(_get_engine),
) -> dict[str, Any]:
    baseline = engine.search(
        q, user_id=user_id, limit=limit, personalise=False, allow_discovery=False
    )
    personalised = engine.search(
        q, user_id=user_id, limit=limit, personalise=True, allow_discovery=False
    )

    def titles(response) -> list[dict[str, Any]]:
        return [
            {
                "content_id": result.item.content_id,
                "title": result.item.title,
                "score": round(result.score, 4),
                "semantic_similarity": round(result.semantic_similarity, 4),
                "profile_similarity": round(result.profile_similarity, 4),
            }
            for result in response.results
        ]

    return {
        "query": q,
        "semantic_only": titles(baseline),
        "personalised": titles(personalised),
    }


app = admin_app
