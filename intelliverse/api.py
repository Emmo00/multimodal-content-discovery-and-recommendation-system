"""REST interface.

The HTTP surface of ``QueryInterface``. Deliberately thin: it validates input,
delegates, and shapes the response. No retrieval, ranking or profile logic lives
here, so the engine remains testable without a running server.

Run with:  uvicorn intelliverse.api:app --reload
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query as QueryParam
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field

from .config import Settings
from .engine import IntelliVerse, build_default_engine
from .models import ContentType, InteractionType
from .sources import DEFAULT_FEEDS, RSSContentSource, YouTubeContentSource


# --- schemas -------------------------------------------------------------

class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr


class UserOut(BaseModel):
    user_id: str
    name: str
    email: str
    joined_at: datetime


class ResultOut(BaseModel):
    content_id: str
    title: str
    description: str
    content_type: str
    source_platform: str
    source_url: str
    topics: list[str]
    score: float
    # The breakdown is returned, not hidden: a discovery system that cannot say
    # why it ranked something is hard to trust and impossible to debug.
    semantic_similarity: float
    profile_similarity: float
    freshness: float


class SearchOut(BaseModel):
    query: str
    personalised: bool
    discovery_triggered: bool
    newly_indexed: int
    candidates_considered: int
    results: list[ResultOut]


class InteractionIn(BaseModel):
    user_id: str
    content_id: str
    interaction_type: Literal[
        "click", "view", "completion", "save", "like", "dislike", "skip"
    ]


class ProfileOut(BaseModel):
    user_id: str
    interaction_count: int
    personalisable: bool
    content_type_preferences: dict[str, float]
    last_updated: datetime


# --- application ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_engine() -> IntelliVerse:
    """Build the engine once and reuse it.

    Cached because loading the embedding model is expensive and must not happen
    per request. Swap the in-memory backends for the Postgres ones here when
    deploying against a real database.
    """
    settings = Settings.from_env()
    sources = [RSSContentSource(DEFAULT_FEEDS)]
    youtube = YouTubeContentSource()
    if youtube.available:
        sources.append(youtube)
    return build_default_engine(settings=settings, sources=sources)


app = FastAPI(
    title="IntelliVerse",
    description=(
        "AI-powered multimodal content discovery and recommendation. Indexes "
        "metadata from external platforms and ranks it semantically and "
        "personally; it does not host media."
    ),
    version="0.1.0",
)

FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/api/health")
@app.get("/health")
def health(engine: IntelliVerse = Depends(get_engine)) -> dict:
    return {
        "status": "ok",
        "corpus_size": engine.corpus_size(),
        "embedding_model": getattr(engine.embedding_service, "model_name", "unknown"),
        "embedding_dimension": engine.embedding_service.dimension,
    }


@app.post("/api/users", response_model=UserOut, status_code=201)
@app.post("/users", response_model=UserOut, status_code=201)
def create_user(payload: UserCreate, engine: IntelliVerse = Depends(get_engine)) -> UserOut:
    user = engine.register_user(payload.name, str(payload.email))
    return UserOut(**user.__dict__)


@app.get("/api/search", response_model=SearchOut)
@app.get("/search", response_model=SearchOut)
def search(
    q: str = QueryParam(min_length=1, description="Natural-language query"),
    user_id: str | None = None,
    content_type: list[Literal["video", "audio", "text"]] | None = QueryParam(default=None),
    limit: int = QueryParam(default=10, ge=1, le=50),
    personalise: bool = True,
    allow_discovery: bool = True,
    engine: IntelliVerse = Depends(get_engine),
) -> SearchOut:
    types = [ContentType(c) for c in content_type] if content_type else None
    response = engine.search(
        q,
        user_id=user_id,
        content_types=types,
        limit=limit,
        personalise=personalise,
        allow_discovery=allow_discovery,
    )
    return SearchOut(
        query=response.query,
        personalised=response.personalised,
        discovery_triggered=response.discovery_triggered,
        newly_indexed=response.newly_indexed,
        candidates_considered=response.candidates_considered,
        results=[
            ResultOut(
                content_id=r.item.content_id,
                title=r.item.title,
                description=r.item.description,
                content_type=r.item.content_type.value,
                source_platform=r.item.source_platform,
                source_url=r.item.source_url,
                topics=list(r.item.topics),
                score=round(r.score, 4),
                semantic_similarity=round(r.semantic_similarity, 4),
                profile_similarity=round(r.profile_similarity, 4),
                freshness=round(r.freshness, 4),
            )
            for r in response.results
        ],
    )


@app.post("/api/interactions", status_code=201)
@app.post("/interactions", status_code=201)
def record_interaction(
    payload: InteractionIn, engine: IntelliVerse = Depends(get_engine)
) -> dict:
    if engine.index.get(payload.content_id) is None:
        raise HTTPException(status_code=404, detail="Content item not found")
    if engine.users.get_user(payload.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    interaction = engine.record_interaction(
        payload.user_id, payload.content_id, InteractionType(payload.interaction_type)
    )
    profile = engine.profiles.get_profile(payload.user_id)
    return {
        "interaction_id": interaction.interaction_id,
        "strength": interaction.strength,
        "interaction_count": profile.interaction_count,
        "personalisation_active": engine.profiles.is_personalisable(profile),
    }


@app.get("/api/users/{user_id}/profile", response_model=ProfileOut)
@app.get("/users/{user_id}/profile", response_model=ProfileOut)
def get_profile(user_id: str, engine: IntelliVerse = Depends(get_engine)) -> ProfileOut:
    if engine.users.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    profile = engine.profiles.get_profile(user_id)
    return ProfileOut(
        user_id=profile.user_id,
        interaction_count=profile.interaction_count,
        personalisable=engine.profiles.is_personalisable(profile),
        content_type_preferences={
            ct.value: round(profile.content_type_preference(ct), 4) for ct in ContentType
        },
        last_updated=profile.last_updated,
    )


@app.post("/admin/api/discover")
@app.post("/discover")
def discover(
    q: str = QueryParam(min_length=1),
    limit: int = QueryParam(default=25, ge=1, le=100),
    engine: IntelliVerse = Depends(get_engine),
) -> dict:
    """Ingest content for a topic without performing a search.

    Used to seed the corpus ahead of evaluation, so that measurement runs
    against a fixed index.
    """
    result = engine.discovery.discover_and_index(q, limit=limit)
    return {
        "retrieved": result.retrieved,
        "indexed": result.indexed,
        "duplicates": result.duplicates,
        "corpus_size": engine.corpus_size(),
    }


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def user_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "user" / "index.html")


@app.get("/admin/", include_in_schema=False)
def admin_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "admin" / "index.html")


from .admin_api import admin_app  # noqa: E402

app.mount("/admin/api", admin_app)
