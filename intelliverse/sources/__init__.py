"""External content source adapters."""

from .base import ContentSource
from .rss import DEFAULT_FEEDS, RSSContentSource
from .youtube import YouTubeContentSource

__all__ = ["ContentSource", "RSSContentSource", "YouTubeContentSource", "DEFAULT_FEEDS"]