"""Command-line interface.

Provides the operations needed to work with a corpus outside the HTTP API:
seeding from feeds, ad-hoc querying, and running the evaluation comparison.

Usage:
    python -m intelliverse.cli seed --feed URL:text --feed URL:video
    python -m intelliverse.cli search "how do neural networks learn"
    python -m intelliverse.cli serve
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv
load_dotenv()  # Load .env file

from .config import Settings
from .engine import build_default_engine
from .models import ContentType
from .sources import DEFAULT_FEEDS, RSSContentSource


def _parse_feed(spec: str) -> tuple[str, ContentType]:
    """Parse ``URL:content_type``, splitting on the final colon only.

    Splitting on the last colon rather than the first keeps the scheme
    (``https://``) intact.
    """
    url, _, type_name = spec.rpartition(":")
    if not url:
        raise argparse.ArgumentTypeError(
            f"Expected 'URL:content_type', got {spec!r}"
        )
    try:
        return url, ContentType(type_name.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Unknown content type {type_name!r}; use video, audio or text"
        ) from None


def cmd_seed(args) -> int:
    feeds = dict(args.feed) if args.feed else DEFAULT_FEEDS
    source = RSSContentSource(feeds)
    engine = build_default_engine(sources=[source])

    print(f"Fetching {len(feeds)} feed(s)...", file=sys.stderr)
    items = list(source.fetch_all(limit_per_feed=args.limit))
    if not items:
        print("No items retrieved. Check the feed URLs and network access.",
              file=sys.stderr)
        return 1

    indexed = engine.ingest(items)
    print(f"Retrieved {len(items)} items; indexed {indexed}.")
    return 0


def cmd_search(args) -> int:
    engine = build_default_engine(
        settings=Settings.from_env(), sources=[RSSContentSource(DEFAULT_FEEDS)]
    )
    types = [ContentType(t) for t in args.type] if args.type else None
    response = engine.search(
        args.query,
        user_id=args.user,
        content_types=types,
        limit=args.limit,
        personalise=not args.no_personalise,
        allow_discovery=not args.no_discovery,
    )

    if args.json:
        print(json.dumps({
            "query": response.query,
            "personalised": response.personalised,
            "discovery_triggered": response.discovery_triggered,
            "results": [
                {
                    "title": r.item.title,
                    "content_type": r.item.content_type.value,
                    "url": r.item.source_url,
                    "score": round(r.score, 4),
                    "semantic_similarity": round(r.semantic_similarity, 4),
                }
                for r in response.results
            ],
        }, indent=2))
        return 0

    if not response.results:
        print("No results.")
        return 0
    for rank, r in enumerate(response.results, 1):
        print(f"{rank}. [{r.item.content_type.value}] {r.item.title}")
        print(f"   {r.item.source_url}")
        print(f"   score={r.score:.3f} semantic={r.semantic_similarity:.3f}")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install uvicorn", file=sys.stderr)
        return 1
    uvicorn.run("intelliverse.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="intelliverse", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="Fetch feeds and index their content")
    seed.add_argument("--feed", action="append", type=_parse_feed,
                      metavar="URL:TYPE", help="Repeatable; defaults to DEFAULT_FEEDS")
    seed.add_argument("--limit", type=int, default=50, help="Max items per feed")
    seed.set_defaults(func=cmd_seed)

    search = subparsers.add_parser("search", help="Run a query")
    search.add_argument("query")
    search.add_argument("--user", help="User ID for personalised ranking")
    search.add_argument("--type", action="append",
                        choices=[t.value for t in ContentType])
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--no-personalise", action="store_true")
    search.add_argument("--no-discovery", action="store_true")
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=cmd_search)

    serve = subparsers.add_parser("serve", help="Start the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())