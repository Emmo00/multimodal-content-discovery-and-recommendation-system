"""RSS and Atom feed source.

Chosen as the first working adapter because it needs no API key, no
registration and no per-request quota, and because the three media types this
project spans are all published over it: articles via RSS, podcasts via RSS with
enclosures, and video via YouTube's per-channel Atom feeds.

Feeds are not searchable, so this adapter retrieves entries and filters them
locally against the query. That is a genuine limitation: it can only surface
content from feeds the system already subscribes to, and its filtering is
lexical. A keyed search API (see :mod:`.youtube`) is the natural complement.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, Sequence
from xml.etree import ElementTree as ET

from ..models import ContentItem, ContentType

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "media": "http://search.yahoo.com/mrss/",
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "content": "http://purl.org/rss/1.0/modules/content/",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")

USER_AGENT = "IntelliVerse/0.1 (final-year project; content discovery)"


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)  # RFC 822, used by RSS
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # ISO, Atom
        except ValueError:
            return None
    if parsed and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _text(element, path: str) -> str:
    found = element.find(path, _NS)
    return (found.text or "").strip() if found is not None and found.text else ""


class RSSContentSource:
    """Retrieves and normalises entries from a fixed set of feeds.

    Parameters
    ----------
    feeds:
        Mapping of feed URL to the :class:`ContentType` its entries represent.
        The type is declared per feed rather than inferred, because a podcast
        feed and a blog feed are structurally identical apart from their
        enclosures.
    """

    def __init__(
        self,
        feeds: dict[str, ContentType] | None = None,
        *,
        name: str = "rss",
        timeout: int = 10,
    ) -> None:
        self.feeds = feeds or {}
        self.name = name
        self.timeout = timeout

    # -- fetching ---------------------------------------------------------

    def _fetch(self, url: str) -> str | None:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            # Fail soft: an unreachable feed must not break the user's query.
            return None

    # -- parsing ----------------------------------------------------------

    def parse_feed(self, xml_text: str, content_type: ContentType) -> list[ContentItem]:
        """Parse feed XML into content items. Supports both RSS 2.0 and Atom."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        channel = root.find("channel")
        if channel is not None:
            platform = _text(channel, "title") or self.name
            return self._parse_rss(channel, platform, content_type)
        if root.tag.endswith("feed"):
            platform = _text(root, "atom:title") or self.name
            return self._parse_atom(root, platform, content_type)
        return []

    def _parse_rss(self, channel, platform: str, content_type: ContentType) -> list[ContentItem]:
        items = []
        for entry in channel.findall("item"):
            title = _text(entry, "title")
            link = _text(entry, "link")
            if not title or not link:
                continue
            description = strip_html(
                _text(entry, "description") or _text(entry, "content:encoded")
            )
            topics = [c.text.strip() for c in entry.findall("category", _NS) if c.text]
            items.append(
                ContentItem(
                    title=title,
                    description=description[:2000],
                    content_type=content_type,
                    source_platform=platform,
                    source_url=link,
                    topics=topics,
                    published_at=_parse_date(_text(entry, "pubDate")),
                )
            )
        return items

    def _parse_atom(self, root, platform: str, content_type: ContentType) -> list[ContentItem]:
        items = []
        for entry in root.findall("atom:entry", _NS):
            title = _text(entry, "atom:title")
            link_el = entry.find("atom:link", _NS)
            link = link_el.get("href") if link_el is not None else ""
            if not title or not link:
                continue
            description = strip_html(
                _text(entry, "media:group/media:description")
                or _text(entry, "atom:summary")
                or _text(entry, "atom:content")
            )
            topics = [
                c.get("term") for c in entry.findall("atom:category", _NS) if c.get("term")
            ]
            items.append(
                ContentItem(
                    title=title,
                    description=description[:2000],
                    content_type=content_type,
                    source_platform=platform,
                    source_url=link,
                    topics=topics,
                    published_at=_parse_date(
                        _text(entry, "atom:published") or _text(entry, "atom:updated")
                    ),
                )
            )
        return items

    # -- searching --------------------------------------------------------

    @staticmethod
    def _relevance(item: ContentItem, query_tokens: set[str]) -> float:
        """Fraction of query tokens appearing in the item's text.

        Lexical by necessity: feeds expose no search endpoint. Semantic ranking
        is applied later, after these candidates are embedded and indexed.
        """
        if not query_tokens:
            return 0.0
        haystack = set(_TOKEN_RE.findall(f"{item.title} {item.description}".lower()))
        haystack |= {t.lower() for t in item.topics}
        return len(query_tokens & haystack) / len(query_tokens)

    def search(self, query: str, limit: int = 10) -> Sequence[ContentItem]:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        scored: list[tuple[float, ContentItem]] = []

        for url, content_type in self.feeds.items():
            xml_text = self._fetch(url)
            if not xml_text:
                continue
            for item in self.parse_feed(xml_text, content_type):
                score = self._relevance(item, query_tokens)
                if score > 0:
                    scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def fetch_all(self, limit_per_feed: int = 50) -> Iterable[ContentItem]:
        """Retrieve every entry from every feed, ignoring relevance.

        Used to seed an initial corpus, and to build the evaluation collection
        for Chapter Four.
        """
        for url, content_type in self.feeds.items():
            xml_text = self._fetch(url)
            if xml_text:
                yield from self.parse_feed(xml_text, content_type)[:limit_per_feed]


#: A starting set of feeds spanning all three media types. These are examples,
#: not a committed dependency of the system — Chapter One treats specific
#: platforms as illustrative.
DEFAULT_FEEDS: dict[str, ContentType] = {
    "https://feeds.arstechnica.com/arstechnica/technology-lab": ContentType.TEXT,
    "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml": ContentType.TEXT,
    "https://feeds.megaphone.fm/hubermanlab": ContentType.AUDIO,
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg": ContentType.VIDEO,
}