"""Content source adapters.

Chapter One deliberately left the specific external platforms unfixed. This
module therefore defines a narrow interface that any source must satisfy, so
that adding a platform means writing one adapter rather than modifying the
discovery pipeline.

An adapter's sole responsibility is to return normalised :class:`ContentItem`
objects. Embedding and indexing happen downstream, in the discovery service.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from ..models import ContentItem


class ContentSource(Protocol):
    """An external platform from which content references can be obtained."""

    name: str

    def search(self, query: str, limit: int = 10) -> Sequence[ContentItem]:
        """Return content matching ``query``.

        Adapters should fail soft: a source that is unreachable or rate-limited
        returns an empty sequence rather than raising, so that one failing
        source cannot take down a user's query.
        """
        ...