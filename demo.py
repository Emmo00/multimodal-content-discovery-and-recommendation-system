#!/usr/bin/env python3
"""End-to-end demonstration of IntelliVerse.

Walks through the workflow of Chapter Three's activity diagram using a small
built-in corpus spanning all three media types:

  1. Index content and run an unpersonalised semantic query
  2. Record interactions that build a user profile
  3. Re-run the same query and observe the reordering
  4. Trigger discovery for a query the corpus cannot answer
  5. Evaluate personalised against non-personalised ranking

Run:  python demo.py
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone

from intelliverse import (
    ContentItem,
    ContentType,
    InteractionType,
    IntelliVerse,
    Settings,
)
from intelliverse.embedding import build_embedding_service
from intelliverse.evaluation import Evaluator, Query, format_comparison
from intelliverse.storage.memory import (
    InMemoryInteractionRepository,
    InMemoryUserRepository,
    InMemoryVectorIndex,
)

NOW = datetime.now(timezone.utc)


def item(title, description, content_type, days_old, topics):
    return ContentItem(
        title=title,
        description=description,
        content_type=content_type,
        source_platform="demo-corpus",
        source_url=f"https://example.test/{title.lower().replace(' ', '-')}",
        topics=topics,
        published_at=NOW - timedelta(days=days_old),
    )


CORPUS = [
    item("How neural networks learn", "A visual walkthrough of backpropagation and "
         "gradient descent in deep neural networks", ContentType.VIDEO, 20,
         ["machine learning", "education"]),
    item("Backpropagation explained simply", "Why the chain rule lets a network "
         "adjust its weights from its errors", ContentType.TEXT, 200,
         ["machine learning"]),
    item("The deep learning podcast: optimisers", "Researchers discuss Adam, SGD "
         "and why optimiser choice matters", ContentType.AUDIO, 5,
         ["machine learning"]),
    item("Transformers and attention", "Self-attention, positional encoding and "
         "the architecture behind modern language models", ContentType.VIDEO, 45,
         ["machine learning", "nlp"]),
    item("Fermentation in sourdough", "How wild yeast and bacteria develop flavour "
         "in naturally leavened bread", ContentType.TEXT, 300, ["cooking", "baking"]),
    item("Handmade pasta masterclass", "Rolling, cutting and shaping fresh egg "
         "pasta dough by hand", ContentType.VIDEO, 90, ["cooking"]),
    item("The food science podcast: bread", "A conversation on gluten development "
         "and hydration ratios in baking", ContentType.AUDIO, 15, ["cooking", "baking"]),
    item("Roman aqueducts", "How Roman engineers moved water hundreds of miles "
         "using only gravity", ContentType.TEXT, 500, ["history", "engineering"]),
]


def show(response, title):
    print(f"\n  {title}")
    print(f"  {'-' * 72}")
    if not response.results:
        print("  (no results)")
        return
    for rank, r in enumerate(response.results, 1):
        marker = "*" if r.personalised else " "
        print(f"  {rank}.{marker} [{r.item.content_type.value:<5}] {r.item.title}")
        print(f"       score={r.score:.3f}  semantic={r.semantic_similarity:.3f}  "
              f"profile={r.profile_similarity:.3f}")


def main() -> None:
    print("=" * 78)
    print("  INTELLIVERSE — demonstration")
    print("=" * 78)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        embedding_service = build_embedding_service()
        fell_back = any(issubclass(w.category, RuntimeWarning) for w in caught)

    model = getattr(embedding_service, "model_name", "unknown")
    print(f"\n  Embedding backend : {model} ({embedding_service.dimension}d)")
    if fell_back:
        print("  WARNING: Sentence-BERT unavailable; using the LEXICAL fallback.")
        print("           Rankings below exercise the pipeline but are not semantic.")
        print("           Install sentence-transformers for real results.")

    engine = IntelliVerse(
        embedding_service=embedding_service,
        index=InMemoryVectorIndex(embedding_service.dimension),
        user_repository=InMemoryUserRepository(),
        interaction_repository=InMemoryInteractionRepository(),
        sources=(),
        settings=Settings(result_limit=5, min_top_similarity=0.05),
    )

    # -- 1. index ---------------------------------------------------------
    indexed = engine.ingest(CORPUS)
    print(f"\n[1] Indexed {indexed} items across video, audio and text.")

    query = "how do machines learn from data"
    show(engine.search(query, allow_discovery=False), f"Query: '{query}' — no user")

    # -- 2. build a profile ----------------------------------------------
    user = engine.register_user("Demo User", "demo@example.test")
    print(f"\n[2] Registered user; recording interactions with cooking content.")
    for content in engine.index.all_items():
        if "cooking" in content.topics:
            for signal in (InteractionType.LIKE, InteractionType.COMPLETION,
                           InteractionType.SAVE):
                engine.record_interaction(user.user_id, content.content_id, signal)

    profile = engine.profiles.get_profile(user.user_id)
    print(f"    interactions={profile.interaction_count}  "
          f"personalisation active={engine.profiles.is_personalisable(profile)}")
    preferences = {
        ct.value: round(profile.content_type_preference(ct), 2) for ct in ContentType
    }
    print(f"    learned content-type preference: {preferences}")

    # -- 3. same query, now personalised ---------------------------------
    print("\n[3] Re-running an ambiguous query for this user.")
    ambiguous = "techniques and methods explained"
    show(engine.search(ambiguous, user_id=user.user_id, personalise=False,
                       allow_discovery=False), f"'{ambiguous}' — semantic only")
    show(engine.search(ambiguous, user_id=user.user_id, personalise=True,
                       allow_discovery=False), f"'{ambiguous}' — personalised")

    # -- 4. discovery -----------------------------------------------------
    print("\n[4] Querying a topic absent from the corpus.")

    class DemoSource:
        name = "demo-external"

        def search(self, q, limit=10):
            return [item("Quantum error correction", "Surface codes, logical qubits "
                         "and fault tolerance in quantum computers",
                         ContentType.TEXT, 3, ["quantum computing"])]

    engine.discovery.sources = [DemoSource()]
    engine.settings.min_top_similarity = 0.30
    response = engine.search("quantum error correction")
    print(f"    discovery triggered : {response.discovery_triggered}")
    print(f"    newly indexed       : {response.newly_indexed}")
    print(f"    corpus size         : {engine.corpus_size()}")
    show(response, "Results after discovery")

    # -- 5. evaluation ----------------------------------------------------
    print("\n[5] Comparing personalised against non-personalised ranking.")
    cooking_ids = {
        c.content_id: 1.0 for c in engine.index.all_items() if "cooking" in c.topics
    }
    queries = [
        Query(query_id="q1", text="techniques and methods explained",
              user_id=user.user_id, relevance=cooking_ids),
        Query(query_id="q2", text="how things are made", user_id=user.user_id,
              relevance=cooking_ids),
    ]
    reports = Evaluator(engine, cutoffs=(1, 3, 5)).compare(queries)
    print()
    print(format_comparison(reports, cutoffs=(1, 3, 5)))

    print("\n  These figures come from a 9-item toy corpus and two queries with")
    print("  relevance judgements written for this demonstration. They show the")
    print("  harness works; they are not evaluation results and must not be")
    print("  reported as such.")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
