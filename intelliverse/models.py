"""Domain models for IntelliVerse.

These correspond directly to the classes in the Chapter Three class diagram
(Figure 3.4) and to the database tables in Tables 3.3-3.6.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class ContentType(str, Enum):
    """The three media types IntelliVerse spans (Chapter One, scope)."""

    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class InteractionType(str, Enum):
    """Interaction signals recorded against returned results.

    The set is fixed by Chapter One's user-feedback stage. Each type carries a
    weight (see ``INTERACTION_STRENGTH``) reflecting how strongly it evidences
    interest.
    """

    CLICK = "click"
    VIEW = "view"
    COMPLETION = "completion"
    SAVE = "save"
    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"


#: Evidence weight per interaction type.
#:
#: Deliberate actions (save, like) outrank passive ones (click, view), and
#: negative signals carry negative weight so that they subtract from the user's
#: interest vector rather than merely failing to add to it. These values are an
#: initial, defensible ordering rather than an empirically tuned result; they
#: are an evaluation parameter (Chapter Three, Section 3.2.2.5).
INTERACTION_STRENGTH: Mapping[InteractionType, float] = {
    InteractionType.CLICK: 0.3,
    InteractionType.VIEW: 0.5,
    InteractionType.COMPLETION: 1.0,
    InteractionType.SAVE: 1.2,
    InteractionType.LIKE: 1.0,
    InteractionType.DISLIKE: -1.0,
    InteractionType.SKIP: -0.3,
}


@dataclass
class User:
    """A registered user of the system (Table 3.3)."""

    name: str
    email: str
    user_id: str = field(default_factory=_new_id)
    joined_at: datetime = field(default_factory=_now)


@dataclass
class ContentItem:
    """A single piece of content discovered from an external source.

    IntelliVerse stores only the metadata and a link; the media itself remains
    on the source platform (Chapter One, scope).
    """

    title: str
    description: str
    content_type: ContentType
    source_platform: str
    source_url: str
    topics: Sequence[str] = field(default_factory=list)
    content_id: str = field(default_factory=_new_id)
    embedding: np.ndarray | None = None
    published_at: datetime | None = None
    discovered_at: datetime = field(default_factory=_now)

    def build_semantic_text(self) -> str:
        """Compose the textual representation submitted to the embedding model.

        This is the point at which content of any medium is reduced to a uniform
        textual form, which is what allows one embedding model to span video,
        audio and text (Chapter Three, Section 3.3.1).

        The content type is included in the text deliberately: a query such as
        "podcast about macroeconomics" carries medium intent that would
        otherwise be invisible to a metadata-only representation.
        """
        parts = [self.title.strip()]
        if self.description:
            parts.append(self.description.strip())
        if self.topics:
            parts.append("Topics: " + ", ".join(self.topics))
        parts.append(f"Content type: {self.content_type.value}")
        return ". ".join(p for p in parts if p)

    def age_days(self, *, at: datetime | None = None) -> float:
        """Age in days, measured from publication where known.

        Falls back to discovery time for sources that expose no publication
        date, which is the best available proxy.
        """
        reference = self.published_at or self.discovered_at
        at = at or _now()
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        return max(0.0, (at - reference).total_seconds() / 86400.0)


@dataclass
class Interaction:
    """A timestamped record of a user acting on a content item (Table 3.6)."""

    user_id: str
    content_id: str
    interaction_type: InteractionType
    interaction_id: str = field(default_factory=_new_id)
    strength: float | None = None
    timestamp: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.strength is None:
            self.strength = INTERACTION_STRENGTH[self.interaction_type]


@dataclass
class UserProfile:
    """Dynamic representation of a user's interests (Table 3.4).

    ``interest_vector`` holds a *decayed accumulator*, not a unit vector: each
    interaction adds ``strength * embedding`` to it, and prior contributions are
    exponentially decayed by elapsed time before the addition. Normalisation is
    applied on read, by :meth:`get_interest_embedding`.

    Keeping the accumulator rather than a normalised average makes the update
    O(1) in the number of past interactions, so the profile can be maintained
    incrementally as feedback arrives instead of being recomputed from the full
    interaction history on every query.
    """

    user_id: str
    interest_vector: np.ndarray | None = None
    content_type_weights: dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    last_updated: datetime = field(default_factory=_now)

    def get_interest_embedding(self) -> np.ndarray | None:
        """Return the L2-normalised interest vector, or ``None`` on cold start."""
        if self.interest_vector is None:
            return None
        norm = float(np.linalg.norm(self.interest_vector))
        if norm < 1e-9:
            # Interactions have cancelled out (e.g. a like and a dislike on
            # near-identical content). Treat as no usable signal.
            return None
        return self.interest_vector / norm

    def content_type_preference(self, content_type: ContentType) -> float:
        """Share of this user's accumulated positive interest in one medium.

        Returns 0.0 when nothing is known, so that an unknown medium is neither
        rewarded nor penalised beyond its semantic merit.
        """
        total = sum(w for w in self.content_type_weights.values() if w > 0)
        if total <= 0:
            return 0.0
        return max(0.0, self.content_type_weights.get(content_type.value, 0.0)) / total

    def to_row(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "interest_embedding": (
                None if self.interest_vector is None else self.interest_vector.tolist()
            ),
            "content_type_preferences": dict(self.content_type_weights),
            "interaction_count": self.interaction_count,
            "last_updated": self.last_updated,
        }


@dataclass
class ScoredContent:
    """A candidate together with the score components that ranked it.

    The breakdown is retained rather than discarded because the evaluation in
    Chapter Four compares personalised against non-personalised ranking, which
    requires the semantic term to remain separately inspectable.
    """

    item: ContentItem
    score: float
    semantic_similarity: float
    profile_similarity: float = 0.0
    freshness: float = 0.0
    type_preference: float = 0.0
    personalised: bool = False
