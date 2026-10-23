"""PostgreSQL + pgvector storage backend.

This is the production backend selected in Chapter Three, Section 3.2.2.2. It
implements the same protocols as the in-memory backend and maps onto the schema
in ``schema.sql``.

Similarity is computed with pgvector's ``<=>`` cosine-distance operator, and
cosine similarity is recovered as ``1 - distance``. Because embeddings are
L2-normalised at generation time, this is exact rather than an approximation of
cosine.

NOTE ON VERIFICATION: this module was written against the psycopg 3 API and the
pgvector SQL surface but could not be executed in the environment where it was
authored, which had neither PostgreSQL nor network access to install it. It
should be run against a live database before being relied upon. The in-memory
backend is fully exercised by the test suite.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from ..models import (
    ContentItem,
    ContentType,
    Interaction,
    InteractionType,
    User,
    UserProfile,
)


def _require_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg, dict_row
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The PostgreSQL backend requires psycopg and pgvector. Install with "
            "`pip install 'psycopg[binary]' pgvector`."
        ) from exc


def connect(database_url: str):
    """Open a connection with pgvector's type adapters registered."""
    psycopg, dict_row = _require_psycopg()
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(database_url, row_factory=dict_row)
    register_vector(conn)
    return conn


def _as_datetime(value) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _row_to_item(row: dict) -> ContentItem:
    return ContentItem(
        content_id=str(row["content_id"]),
        title=row["title"],
        description=row["description"] or "",
        content_type=ContentType(row["content_type"]),
        source_platform=row["source_platform"],
        source_url=row["source_url"],
        topics=list(row["topics"] or []),
        embedding=None if row.get("embedding") is None else np.asarray(row["embedding"], np.float32),
        published_at=_as_datetime(row.get("published_at")) if row.get("published_at") else None,
        discovered_at=_as_datetime(row.get("discovered_at")),
    )


class PostgresVectorIndex:
    """Content store and vector index backed by PostgreSQL with pgvector."""

    def __init__(self, connection) -> None:
        self._conn = connection

    def index_content(self, item: ContentItem) -> None:
        self.index_many([item])

    def index_many(self, items: Sequence[ContentItem]) -> int:
        if not items:
            return 0
        for item in items:
            if item.embedding is None:
                raise ValueError(f"Item '{item.title}' has no embedding")

        # ON CONFLICT on source_url makes ingestion idempotent: rediscovering
        # the same URL refreshes its metadata instead of creating a duplicate.
        sql = """
            INSERT INTO content_items (
                content_id, title, description, content_type, source_platform,
                source_url, topics, embedding, published_at, discovered_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_url) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                topics = EXCLUDED.topics,
                embedding = EXCLUDED.embedding
            RETURNING (xmax = 0) AS inserted
        """
        added = 0
        with self._conn.cursor() as cur:
            for item in items:
                cur.execute(
                    sql,
                    (
                        item.content_id,
                        item.title,
                        item.description,
                        item.content_type.value,
                        item.source_platform,
                        item.source_url,
                        list(item.topics),
                        np.asarray(item.embedding, dtype=np.float32),
                        item.published_at,
                        item.discovered_at,
                    ),
                )
                row = cur.fetchone()
                if row and row["inserted"]:
                    added += 1
        self._conn.commit()
        return added

    def query_similar(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        content_types: Sequence[ContentType] | None = None,
        exclude_ids: Sequence[str] | None = None,
    ) -> list[tuple[ContentItem, float]]:
        clauses, params = [], [np.asarray(query_vector, dtype=np.float32)]
        if content_types:
            clauses.append("content_type = ANY(%s)")
            params.append([ct.value for ct in content_types])
        if exclude_ids:
            clauses.append("content_id <> ALL(%s)")
            params.append(list(exclude_ids))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(top_k)

        sql = f"""
            SELECT content_id, title, description, content_type, source_platform,
                   source_url, topics, published_at, discovered_at,
                   1 - (embedding <=> %s) AS similarity
            FROM content_items
            {where}
            ORDER BY similarity DESC
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [(_row_to_item(r), float(r["similarity"])) for r in cur.fetchall()]

    def get(self, content_id: str) -> ContentItem | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM content_items WHERE content_id = %s", (content_id,))
            row = cur.fetchone()
        return _row_to_item(row) if row else None

    def exists_by_url(self, source_url: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1 FROM content_items WHERE source_url = %s LIMIT 1", (source_url,))
            return cur.fetchone() is not None

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM content_items")
            return int(cur.fetchone()["n"])


class PostgresUserRepository:
    def __init__(self, connection) -> None:
        self._conn = connection

    def save_user(self, user: User) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, name, email, joined_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                    SET name = EXCLUDED.name, email = EXCLUDED.email
                """,
                (user.user_id, user.name, user.email, user.joined_at),
            )
        self._conn.commit()

    def get_user(self, user_id: str) -> User | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
        if not row:
            return None
        return User(
            user_id=str(row["user_id"]),
            name=row["name"],
            email=row["email"],
            joined_at=_as_datetime(row["joined_at"]),
        )

    def get_profile(self, user_id: str) -> UserProfile | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
        if not row:
            return None
        embedding = row.get("interest_embedding")
        return UserProfile(
            user_id=str(row["user_id"]),
            interest_vector=None if embedding is None else np.asarray(embedding, np.float32),
            content_type_weights=dict(row["content_type_preferences"] or {}),
            interaction_count=int(row.get("interaction_count") or 0),
            last_updated=_as_datetime(row["last_updated"]),
        )

    def save_profile(self, profile: UserProfile) -> None:
        from psycopg.types.json import Jsonb

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_profiles (
                    user_id, interest_embedding, content_type_preferences,
                    interaction_count, last_updated
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    interest_embedding = EXCLUDED.interest_embedding,
                    content_type_preferences = EXCLUDED.content_type_preferences,
                    interaction_count = EXCLUDED.interaction_count,
                    last_updated = EXCLUDED.last_updated
                """,
                (
                    profile.user_id,
                    None if profile.interest_vector is None
                    else np.asarray(profile.interest_vector, dtype=np.float32),
                    Jsonb(profile.content_type_weights),
                    profile.interaction_count,
                    profile.last_updated,
                ),
            )
        self._conn.commit()


class PostgresInteractionRepository:
    def __init__(self, connection) -> None:
        self._conn = connection

    def record(self, interaction: Interaction) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interactions (
                    interaction_id, user_id, content_id, interaction_type,
                    strength, timestamp
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    interaction.interaction_id,
                    interaction.user_id,
                    interaction.content_id,
                    interaction.interaction_type.value,
                    interaction.strength,
                    interaction.timestamp,
                ),
            )
        self._conn.commit()

    def for_user(self, user_id: str, limit: int = 100) -> list[Interaction]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM interactions WHERE user_id = %s
                ORDER BY timestamp DESC LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [
            Interaction(
                interaction_id=str(r["interaction_id"]),
                user_id=str(r["user_id"]),
                content_id=str(r["content_id"]),
                interaction_type=InteractionType(r["interaction_type"]),
                strength=float(r["strength"]),
                timestamp=_as_datetime(r["timestamp"]),
            )
            for r in rows
        ]

    def count_for_user(self, user_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM interactions WHERE user_id = %s", (user_id,))
            return int(cur.fetchone()["n"])