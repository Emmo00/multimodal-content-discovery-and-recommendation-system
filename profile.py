"""User interest modelling.

Implements the user model of Chapter One stage 7: a representation of interests
built from the embeddings of content the user has engaged with, weighted by
interaction strength and recency.

The update rule is

    v <- v * 2^(-dt / H) + s * e

where ``v`` is the stored accumulator, ``dt`` the days since the last update,
``H`` the interest half-life, ``s`` the interaction strength, and ``e`` the
embedding of the content acted upon.

Two properties of this rule matter. It is O(1) per interaction, so the profile
is maintained incrementally rather than rebuilt from the full history. And
because ``s`` is negative for dislikes and skips, negative feedback actively
pushes the profile away from disliked content instead of merely leaving it
unchanged — without which a user could never correct a profile that had drifted.

This is content-based user modelling: it uses only the acting user's own
history. No signal from other users enters, so the approach carries none of the
cold-start dependency on a populated interaction matrix that Chapter Two
identifies in collaborative filtering.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .models import ContentItem, Interaction, UserProfile
from .storage.base import UserRepository, VectorIndexService


class UserProfileService:
    def __init__(
        self,
        user_repository: UserRepository,
        index: VectorIndexService,
        *,
        half_life_days: float = 30.0,
        min_interactions_for_personalisation: int = 3,
    ) -> None:
        self.users = user_repository
        self.index = index
        self.half_life_days = half_life_days
        # Below this many interactions a profile is too noisy to trust, and the
        # ranker treats the user as cold-start. A single click should not
        # reshape a user's results.
        self.min_interactions = min_interactions_for_personalisation

    # -- reading ----------------------------------------------------------

    def get_profile(self, user_id: str) -> UserProfile:
        profile = self.users.get_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            self.users.save_profile(profile)
        return profile

    def is_personalisable(self, profile: UserProfile) -> bool:
        """Whether this profile carries enough signal to rank with."""
        return (
            profile.interaction_count >= self.min_interactions
            and profile.get_interest_embedding() is not None
        )

    # -- updating ---------------------------------------------------------

    def _decay_factor(self, since: datetime, until: datetime) -> float:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        elapsed_days = max(0.0, (until - since).total_seconds() / 86400.0)
        return float(2.0 ** (-elapsed_days / self.half_life_days))

    def update_from_interaction(
        self,
        interaction: Interaction,
        *,
        item: ContentItem | None = None,
    ) -> UserProfile:
        """Fold one interaction into the user's profile."""
        item = item or self.index.get(interaction.content_id)
        if item is None or item.embedding is None:
            # Nothing to learn from content we cannot represent.
            return self.get_profile(interaction.user_id)

        profile = self.get_profile(interaction.user_id)
        now = interaction.timestamp
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        decay = self._decay_factor(profile.last_updated, now)
        strength = float(interaction.strength or 0.0)
        embedding = np.asarray(item.embedding, dtype=np.float32)

        if profile.interest_vector is None:
            profile.interest_vector = strength * embedding
        else:
            profile.interest_vector = profile.interest_vector * decay + strength * embedding

        # Content-type preference is decayed on the same schedule, so that a
        # user who shifts from video to text is followed rather than averaged.
        key = item.content_type.value
        for existing_key in list(profile.content_type_weights):
            profile.content_type_weights[existing_key] *= decay
        profile.content_type_weights[key] = (
            profile.content_type_weights.get(key, 0.0) + strength
        )

        profile.interaction_count += 1
        profile.last_updated = now
        self.users.save_profile(profile)
        return profile

    def rebuild_from_history(
        self, user_id: str, interactions: list[Interaction]
    ) -> UserProfile:
        """Recompute a profile from scratch over an interaction history.

        Not needed in normal operation, where updates are incremental. It exists
        for two cases: recovering a profile after a schema change, and
        constructing profiles deterministically from a fixed interaction log
        during evaluation.
        """
        profile = UserProfile(user_id=user_id)
        self.users.save_profile(profile)
        for interaction in sorted(interactions, key=lambda i: i.timestamp):
            profile = self.update_from_interaction(interaction)
        return profile
