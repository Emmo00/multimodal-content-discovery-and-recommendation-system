"""Test suite for IntelliVerse.

Runs against the in-memory backends and the hashing embedding service, so the
whole suite executes offline. The behaviour under test is the system's logic —
profile accumulation, decay, negative feedback, coverage checking, cold-start
degradation — none of which depends on which embedding model supplies the
vectors.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from intelliverse import (
    ContentItem,
    ContentType,
    Interaction,
    InteractionType,
    IntelliVerse,
    RankingWeights,
    Settings,
)
from intelliverse.embedding import HashingEmbeddingService, cosine_similarity
from intelliverse.models import UserProfile
from intelliverse.ranking import RecommendationEngine, freshness_score
from intelliverse.storage.memory import (
    InMemoryInteractionRepository,
    InMemoryUserRepository,
    InMemoryVectorIndex,
)

NOW = datetime.now(timezone.utc)


def make_item(title, description, content_type=ContentType.TEXT, days_old=10, topics=()):
    return ContentItem(
        title=title,
        description=description,
        content_type=content_type,
        source_platform="test",
        source_url=f"https://example.test/{abs(hash(title))}",
        topics=list(topics),
        published_at=NOW - timedelta(days=days_old),
    )


CORPUS = [
    make_item("Neural networks explained", "How backpropagation trains deep networks",
              ContentType.VIDEO, 5, ["machine learning"]),
    make_item("Gradient descent tutorial", "Optimising loss functions with gradient descent",
              ContentType.TEXT, 30, ["machine learning"]),
    make_item("Deep learning podcast", "A conversation about neural network architectures",
              ContentType.AUDIO, 2, ["machine learning"]),
    make_item("Sourdough bread baking", "Fermentation and hydration in artisan bread",
              ContentType.TEXT, 100, ["cooking"]),
    make_item("Italian pasta from scratch", "Making fresh pasta dough by hand",
              ContentType.VIDEO, 60, ["cooking"]),
    make_item("Roman aqueduct engineering", "How Roman engineers moved water across valleys",
              ContentType.TEXT, 400, ["history"]),
]


@pytest.fixture
def engine():
    embedding_service = HashingEmbeddingService(dimension=256)
    settings = Settings(result_limit=5, min_candidates=2, min_top_similarity=0.05)
    engine = IntelliVerse(
        embedding_service=embedding_service,
        index=InMemoryVectorIndex(embedding_service.dimension),
        user_repository=InMemoryUserRepository(),
        interaction_repository=InMemoryInteractionRepository(),
        sources=(),
        settings=settings,
    )
    engine.ingest([make_item(i.title, i.description, i.content_type,
                             i.age_days(), i.topics) for i in CORPUS])
    return engine


# --- models --------------------------------------------------------------

def test_semantic_text_includes_all_metadata_fields():
    item = make_item("Title here", "Description here", ContentType.AUDIO, topics=["alpha"])
    text = item.build_semantic_text()
    assert "Title here" in text
    assert "Description here" in text
    assert "alpha" in text
    assert "audio" in text


def test_interaction_strength_defaults_by_type():
    like = Interaction(user_id="u", content_id="c", interaction_type=InteractionType.LIKE)
    dislike = Interaction(user_id="u", content_id="c", interaction_type=InteractionType.DISLIKE)
    assert like.strength > 0
    assert dislike.strength < 0


def test_age_prefers_publication_over_discovery_date():
    item = make_item("t", "d", days_old=50)
    assert 49 < item.age_days() < 51


# --- embedding -----------------------------------------------------------

def test_embeddings_are_normalised_and_deterministic():
    service = HashingEmbeddingService(dimension=128)
    first = service.embed("machine learning tutorial")
    second = service.embed("machine learning tutorial")
    np.testing.assert_allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0, atol=1e-6)


def test_shared_vocabulary_scores_higher_than_unrelated_text():
    service = HashingEmbeddingService(dimension=256)
    a = service.embed("neural networks and deep learning")
    b = service.embed("deep learning with neural networks")
    c = service.embed("sourdough bread fermentation")
    assert cosine_similarity(a, b) > cosine_similarity(a, c)


def test_empty_text_yields_zero_vector_without_error():
    service = HashingEmbeddingService(dimension=64)
    assert np.linalg.norm(service.embed("")) == 0.0


# --- vector index --------------------------------------------------------

def test_retrieval_orders_by_similarity(engine):
    vector = engine.embedding_service.embed("neural networks")
    results = engine.index.query_similar(vector, top_k=3)
    similarities = [s for _, s in results]
    assert similarities == sorted(similarities, reverse=True)


def test_metadata_filter_restricts_content_type(engine):
    vector = engine.embedding_service.embed("neural networks")
    results = engine.index.query_similar(vector, top_k=10,
                                         content_types=[ContentType.AUDIO])
    assert results
    assert all(item.content_type == ContentType.AUDIO for item, _ in results)


def test_duplicate_urls_are_not_indexed_twice(engine):
    before = engine.corpus_size()
    duplicate = ContentItem(
        title="Neural networks explained", description="dup", content_type=ContentType.VIDEO,
        source_platform="test",
        source_url=engine.index.all_items()[0].source_url,
    )
    engine.ingest([duplicate])
    assert engine.corpus_size() == before


# --- profile modelling ---------------------------------------------------

def test_profile_moves_toward_liked_content(engine):
    user = engine.register_user("Ada", "ada@example.test")
    ml_item = next(i for i in engine.index.all_items() if "Neural" in i.title)

    engine.record_interaction(user.user_id, ml_item.content_id, InteractionType.LIKE)
    profile = engine.profiles.get_profile(user.user_id)
    interest = profile.get_interest_embedding()

    assert interest is not None
    assert cosine_similarity(interest, ml_item.embedding) > 0.9


def test_dislike_pushes_profile_away(engine):
    user = engine.register_user("Bob", "bob@example.test")
    items = engine.index.all_items()
    ml_item = next(i for i in items if "Neural" in i.title)
    food_item = next(i for i in items if "Sourdough" in i.title)

    engine.record_interaction(user.user_id, ml_item.content_id, InteractionType.LIKE)
    before = engine.profiles.get_profile(user.user_id).get_interest_embedding()
    similarity_before = cosine_similarity(before, food_item.embedding)

    engine.record_interaction(user.user_id, food_item.content_id, InteractionType.DISLIKE)
    after = engine.profiles.get_profile(user.user_id).get_interest_embedding()
    similarity_after = cosine_similarity(after, food_item.embedding)

    assert similarity_after < similarity_before


def test_older_interactions_decay(engine):
    user = engine.register_user("Cleo", "cleo@example.test")
    items = engine.index.all_items()
    old_item, new_item = items[0], items[3]

    old = Interaction(user_id=user.user_id, content_id=old_item.content_id,
                      interaction_type=InteractionType.LIKE,
                      timestamp=NOW - timedelta(days=365))
    engine.profiles.update_from_interaction(old, item=old_item)

    recent = Interaction(user_id=user.user_id, content_id=new_item.content_id,
                         interaction_type=InteractionType.LIKE, timestamp=NOW)
    engine.profiles.update_from_interaction(recent, item=new_item)

    interest = engine.profiles.get_profile(user.user_id).get_interest_embedding()
    # A year of decay at a 30-day half-life should leave the recent interaction
    # overwhelmingly dominant.
    assert cosine_similarity(interest, new_item.embedding) > cosine_similarity(
        interest, old_item.embedding
    )


def test_content_type_preference_is_a_normalised_share(engine):
    user = engine.register_user("Dee", "dee@example.test")
    for item in engine.index.all_items():
        if item.content_type == ContentType.VIDEO:
            engine.record_interaction(user.user_id, item.content_id, InteractionType.LIKE)

    profile = engine.profiles.get_profile(user.user_id)
    assert profile.content_type_preference(ContentType.VIDEO) == pytest.approx(1.0)
    assert profile.content_type_preference(ContentType.TEXT) == 0.0


def test_cancelling_feedback_yields_no_usable_profile():
    profile = UserProfile(user_id="u", interest_vector=np.zeros(8, dtype=np.float32))
    assert profile.get_interest_embedding() is None


# --- ranking -------------------------------------------------------------

def test_cold_start_weights_redistribute_to_semantic():
    weights = RankingWeights(semantic=0.6, profile=0.25, freshness=0.05, content_type=0.10)
    cold = weights.for_cold_start()
    assert cold.profile == 0.0
    assert cold.content_type == 0.0
    assert cold.semantic == pytest.approx(0.95)
    assert cold.semantic + cold.freshness == pytest.approx(1.0)


def test_weights_are_normalised_on_construction():
    weights = RankingWeights(semantic=6, profile=2, freshness=1, content_type=1)
    total = weights.semantic + weights.profile + weights.freshness + weights.content_type
    assert total == pytest.approx(1.0)


def test_freshness_halves_at_half_life():
    item = make_item("t", "d", days_old=90)
    assert freshness_score(item, half_life_days=90) == pytest.approx(0.5, abs=0.01)


def test_negative_similarity_is_clamped_not_propagated():
    engine_ranker = RecommendationEngine()
    item = make_item("t", "d")
    item.embedding = np.array([1.0, 0.0], dtype=np.float32)
    scored = engine_ranker.compute_score(item, -0.8, None, personalise=False)
    assert scored.semantic_similarity == 0.0
    assert scored.score >= 0.0


def test_ranking_without_profile_is_not_personalised(engine):
    response = engine.search("neural networks", user_id=None)
    assert response.personalised is False
    assert all(not r.personalised for r in response.results)


# --- end-to-end workflow -------------------------------------------------

def test_search_returns_topically_relevant_results(engine):
    response = engine.search("neural networks deep learning", allow_discovery=False)
    assert response.results
    top_titles = [r.item.title for r in response.results[:3]]
    assert any("Neural" in t or "Deep" in t or "Gradient" in t for t in top_titles)


def test_personalisation_reorders_the_same_candidate_set(engine):
    """The core claim: same retrieval, different order once a profile exists."""
    user = engine.register_user("Eve", "eve@example.test")
    items = engine.index.all_items()

    # Establish a clear cooking preference.
    for item in items:
        if "cooking" in item.topics:
            for signal in (InteractionType.LIKE, InteractionType.COMPLETION,
                           InteractionType.SAVE):
                engine.record_interaction(user.user_id, item.content_id, signal)

    query = "learning techniques"
    baseline = engine.search(query, user_id=user.user_id, personalise=False,
                             allow_discovery=False)
    personalised = engine.search(query, user_id=user.user_id, personalise=True,
                                 allow_discovery=False)

    assert personalised.personalised is True
    assert baseline.personalised is False
    # Same candidate pool considered by both.
    assert baseline.candidates_considered == personalised.candidates_considered

    baseline_order = [r.item.content_id for r in baseline.results]
    personalised_order = [r.item.content_id for r in personalised.results]
    assert baseline_order != personalised_order, "profile should change the ordering"


def test_few_interactions_do_not_trigger_personalisation(engine):
    user = engine.register_user("Fay", "fay@example.test")
    item = engine.index.all_items()[0]
    engine.record_interaction(user.user_id, item.content_id, InteractionType.CLICK)

    response = engine.search("learning", user_id=user.user_id, allow_discovery=False)
    assert response.personalised is False, "one click is not enough evidence"


def test_discovery_fires_when_coverage_is_insufficient():
    """A query the corpus cannot answer should reach for external sources."""

    class StubSource:
        name = "stub"

        def __init__(self):
            self.calls = []

        def search(self, query, limit=10):
            self.calls.append(query)
            return [make_item("Quantum error correction",
                              "Surface codes and logical qubits in quantum computing")]

    embedding_service = HashingEmbeddingService(dimension=256)
    source = StubSource()
    engine = IntelliVerse(
        embedding_service=embedding_service,
        index=InMemoryVectorIndex(embedding_service.dimension),
        user_repository=InMemoryUserRepository(),
        interaction_repository=InMemoryInteractionRepository(),
        sources=[source],
        settings=Settings(min_candidates=3, min_top_similarity=0.5),
    )
    engine.ingest([make_item("Sourdough bread baking", "Fermentation in bread")])

    response = engine.search("quantum error correction surface codes")

    assert response.discovery_triggered is True
    assert source.calls == ["quantum error correction surface codes"]
    assert response.newly_indexed == 1
    assert any("Quantum" in r.item.title for r in response.results)


def test_discovery_is_skipped_when_corpus_already_covers_query(engine):
    class FailingSource:
        name = "should-not-be-called"

        def search(self, query, limit=10):
            raise AssertionError("discovery should not have been triggered")

    engine.discovery.sources = [FailingSource()]
    response = engine.search("neural networks deep learning backpropagation")
    assert response.discovery_triggered is False


def test_failing_source_does_not_break_search():
    class BrokenSource:
        name = "broken"

        def search(self, query, limit=10):
            raise ConnectionError("network down")

    embedding_service = HashingEmbeddingService(dimension=128)
    engine = IntelliVerse(
        embedding_service=embedding_service,
        index=InMemoryVectorIndex(embedding_service.dimension),
        user_repository=InMemoryUserRepository(),
        interaction_repository=InMemoryInteractionRepository(),
        sources=[BrokenSource()],
        settings=Settings(),
    )
    engine.ingest([make_item("Anything", "Some content")])
    response = engine.search("unrelated query")  # forces discovery, which fails
    assert response.discovery_triggered is True
    assert response.newly_indexed == 0  # degraded, not crashed


def test_irrelevant_candidates_are_filtered_out(engine):
    """Freshness and profile fit must not float unrelated content into results."""
    response = engine.search("quantum chromodynamics gluon confinement",
                             allow_discovery=False)
    assert all(r.semantic_similarity >= engine.settings.min_result_similarity
               for r in response.results)


def test_relevance_floor_can_empty_the_result_set(engine):
    engine.settings.min_result_similarity = 0.99
    response = engine.search("neural networks", allow_discovery=False)
    assert response.results == []


def test_empty_corpus_returns_no_results_gracefully():
    embedding_service = HashingEmbeddingService(dimension=64)
    engine = IntelliVerse(
        embedding_service=embedding_service,
        index=InMemoryVectorIndex(embedding_service.dimension),
        user_repository=InMemoryUserRepository(),
        interaction_repository=InMemoryInteractionRepository(),
        settings=Settings(),
    )
    response = engine.search("anything at all")
    assert response.results == []


# --- RSS parsing ---------------------------------------------------------

RSS_SAMPLE = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Test Feed</title>
  <item>
    <title>Understanding transformers</title>
    <description>&lt;p&gt;Attention mechanisms &amp;amp; self-attention&lt;/p&gt;</description>
    <link>https://example.test/transformers</link>
    <pubDate>Wed, 02 Oct 2024 13:00:00 GMT</pubDate>
    <category>machine learning</category>
  </item>
</channel></rss>"""

ATOM_SAMPLE = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Video Channel</title>
  <entry>
    <title>Building a compiler</title>
    <link href="https://example.test/compiler"/>
    <summary>Lexing, parsing and code generation</summary>
    <published>2024-06-01T10:00:00Z</published>
  </entry>
</feed>"""


def test_rss_parsing_strips_html_and_reads_metadata():
    from intelliverse.sources.rss import RSSContentSource

    items = RSSContentSource().parse_feed(RSS_SAMPLE, ContentType.TEXT)
    assert len(items) == 1
    item = items[0]
    assert item.title == "Understanding transformers"
    assert "<p>" not in item.description
    assert "Attention mechanisms" in item.description
    assert item.topics == ["machine learning"]
    assert item.published_at.year == 2024
    assert item.source_platform == "Test Feed"


def test_atom_parsing_reads_link_href():
    from intelliverse.sources.rss import RSSContentSource

    items = RSSContentSource().parse_feed(ATOM_SAMPLE, ContentType.VIDEO)
    assert len(items) == 1
    assert items[0].source_url == "https://example.test/compiler"
    assert items[0].content_type == ContentType.VIDEO


def test_malformed_feed_returns_empty_not_exception():
    from intelliverse.sources.rss import RSSContentSource

    assert RSSContentSource().parse_feed("not xml at all", ContentType.TEXT) == []
