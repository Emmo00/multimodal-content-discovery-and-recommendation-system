"""YouTube Data API v3 source.

Complements the RSS adapter by providing genuine keyword *search* across a large
video corpus rather than filtering a fixed set of feeds.

Requires an API key (``YOUTUBE_API_KEY``). The adapter reports itself as
unavailable when no key is configured, and :class:`~intelliverse.discovery.DiscoveryService`
skips unavailable sources, so the system degrades to whatever sources are
configured rather than failing.

NOTE ON VERIFICATION: the request construction and response mapping below follow
the documented ``search.list`` response shape but were not executed against the
live API in the environment where this was authored, which had no outbound
access to googleapis.com and no key. Verify before relying on it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Sequence

from ..models import ContentItem, ContentType

API_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"


class YouTubeContentSource:
    def __init__(self, api_key: str | None = None, *, timeout: int = 10) -> None:
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        self.name = "youtube"
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, limit: int = 10) -> Sequence[ContentItem]:
        if not self.available:
            return []

        params = urllib.parse.urlencode(
            {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": min(limit, 50),  # API ceiling
                "key": self.api_key,
            }
        )
        try:
            with urllib.request.urlopen(f"{API_ENDPOINT}?{params}", timeout=self.timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, json.JSONDecodeError):
            return []

        items = []
        for entry in payload.get("items", []):
            snippet = entry.get("snippet", {})
            video_id = entry.get("id", {}).get("videoId")
            title = snippet.get("title")
            if not video_id or not title:
                continue
            published = snippet.get("publishedAt")
            try:
                published_at = (
                    datetime.fromisoformat(published.replace("Z", "+00:00"))
                    if published
                    else None
                )
            except ValueError:
                published_at = None
            items.append(
                ContentItem(
                    title=title,
                    description=snippet.get("description", "")[:2000],
                    content_type=ContentType.VIDEO,
                    source_platform=snippet.get("channelTitle") or "YouTube",
                    source_url=f"https://www.youtube.com/watch?v={video_id}",
                    topics=[],
                    published_at=published_at,
                )
            )
        return items