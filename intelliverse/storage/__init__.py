"""Storage backends for IntelliVerse."""

from .base import InteractionRepository, UserRepository, VectorIndexService
from .memory import (
    InMemoryInteractionRepository,
    InMemoryUserRepository,
    InMemoryVectorIndex,
)

__all__ = [
    "VectorIndexService",
    "UserRepository",
    "InteractionRepository",
    "InMemoryVectorIndex",
    "InMemoryUserRepository",
    "InMemoryInteractionRepository",
]


def __getattr__(name: str):
    # Postgres backends are imported lazily so that the package remains usable
    # without psycopg installed.
    if name in {
        "PostgresVectorIndex",
        "PostgresUserRepository",
        "PostgresInteractionRepository",
        "connect",
    }:
        from . import postgres

        return getattr(postgres, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")