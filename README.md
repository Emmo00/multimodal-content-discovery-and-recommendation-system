# IntelliVerse

AI-powered multimodal content discovery and recommendation.

IntelliVerse is a discovery and recommendation **layer** over external content
platforms. It indexes metadata for video, audio and text content, represents it
semantically, and ranks it for the individual user. It does not host, stream or
redistribute media — the source platform remains where content lives.

This is version 0.1, implementing the design in Chapters One to Three.

---

## What it does

Two things are kept deliberately separate, because conflating them is the
weakness the project exists to address:

- **Semantic retrieval** decides which content is *about* the query, by
  comparing embeddings rather than matching keywords.
- **Personalised ranking** decides which of that content is right *for this
  user*, using a profile built from their own interaction history.

The evaluation harness compares the two by disabling the second, which is how
the project's central claim is measured rather than asserted.

---

## Installation

```bash
pip install -r requirements.txt
```

The most important dependency is `sentence-transformers`. Without it the system
falls back to lexical matching and emits a `RuntimeWarning`. See
[Embedding backends](#embedding-backends) — this matters for correctness, not
just performance.

## Quick start

```bash
python demo.py                                    # full walkthrough
python -m intelliverse.cli seed                   # index default feeds
python -m intelliverse.cli search "how neural networks learn"
python -m intelliverse.cli serve                  # REST API on :8000
pytest                                            # 41 tests
```

```python
from intelliverse import build_default_engine, InteractionType

engine = build_default_engine()
engine.ingest(my_items)

user = engine.register_user("Ada", "ada@example.com")
response = engine.search("how do neural networks learn", user_id=user.user_id)

for result in response.results:
    print(result.score, result.item.title, result.item.source_url)

engine.record_interaction(user.user_id, response.results[0].item.content_id,
                          InteractionType.LIKE)
```

---

## Architecture

Module layout follows the Chapter Three class diagram directly.

| Module | Responsibility | Chapter Three component |
|---|---|---|
| `models.py` | Domain entities | `ContentItem`, `User`, `UserProfile`, `Interaction` |
| `embedding.py` | Text → vector | `EmbeddingService` |
| `storage/` | Persistence and similarity search | `VectorIndexService` |
| `sources/` | External platform adapters | (feeds the Discovery Service) |
| `discovery.py` | Search, normalise, embed, index | `DiscoveryService` |
| `profile.py` | Interest modelling | `UserProfileService` |
| `ranking.py` | Personalised scoring | `RecommendationEngine` |
| `engine.py` | Workflow orchestration | `QueryInterface` |
| `evaluation.py` | Offline metrics | (supports Chapter Four) |

Every component is reachable only through an interface, so the storage backend
or embedding model can be replaced without touching the rest — the low coupling
argued for in Section 3.2.3.

### Query workflow

```
query → embed → similarity search → coverage sufficient?
                                      ├─ no  → discover → normalise → embed
                                      │         → index → re-retrieve
                                      └─ yes ↓
        relevance floor → profile exists? ─ no → rank by semantics alone
                                          └ yes → rank with profile
                                                    ↓
                                          results → interactions → profile update
```

### How the profile works

Each interaction folds into a decayed accumulator:

```
v ← v · 2^(−Δt/H) + s · e
```

where `s` is the interaction strength (negative for dislikes and skips) and `H`
is the interest half-life. This is O(1) per interaction, so profiles update
incrementally as feedback arrives. Because `s` can be negative, negative
feedback actively moves the profile away from disliked content rather than
merely failing to reinforce it.

Only the acting user's own history is used. No cross-user signal enters, so the
approach avoids the cold-start dependency on a populated interaction matrix that
collaborative filtering carries.

### Ranking

```
score = w_sem·sim(query, item) + w_prof·sim(interest, item)
      + w_fresh·freshness(item) + w_type·typePreference(item)
```

On cold start the profile and content-type weights are redistributed to the
semantic term, so ranking degrades to pure semantic relevance rather than to a
partial score. A relevance floor is applied before ranking so that freshness and
profile fit reorder relevant content instead of promoting irrelevant content.

---

## Embedding backends

| Backend | Behaviour | Use |
|---|---|---|
| `SentenceTransformerEmbeddingService` | Semantic (all-MiniLM-L6-v2, 384d) | Production, **all reported results** |
| `HashingEmbeddingService` | Lexical only | Offline dev and CI |

The fallback exists because model weights cannot always be downloaded. It is
**not** a substitute: it reproduces exactly the vocabulary-mismatch failure the
project sets out to solve. Any metric produced with it measures the opposite of
what the project claims. Pass `allow_fallback=False` wherever results are being
measured, so a silent downgrade cannot be mistaken for semantic retrieval.

---

## PostgreSQL deployment

```bash
createdb intelliverse
psql intelliverse -f schema.sql
export INTELLIVERSE_DATABASE_URL="postgresql://user:pass@localhost/intelliverse"
```

```python
from intelliverse import IntelliVerse, Settings
from intelliverse.embedding import build_embedding_service
from intelliverse.storage import (connect, PostgresVectorIndex,
                                  PostgresUserRepository, PostgresInteractionRepository)

conn = connect(Settings().database_url)
engine = IntelliVerse(
    embedding_service=build_embedding_service(allow_fallback=False),
    index=PostgresVectorIndex(conn),
    user_repository=PostgresUserRepository(conn),
    interaction_repository=PostgresInteractionRepository(conn),
)
```

Content embeddings and relational data share one database, so similarity search
and metadata filtering compose in a single SQL query — the reason `pgvector` was
chosen over a filtering-less vector library.

---

## Evaluation

```python
from intelliverse.evaluation import Evaluator, Query, format_comparison

queries = [Query(query_id="q1", text="...", user_id=uid,
                 relevance={content_id: 1.0, ...})]
reports = Evaluator(engine).compare(queries)
print(format_comparison(reports))
```

`compare()` runs both conditions over the same queries with discovery disabled,
so the corpus is identical across conditions and any difference is attributable
to personalisation alone. Metrics: Precision@K, Recall@K, NDCG@K, MRR.

The harness computes metrics. It ships with no results, and the figures printed
by `demo.py` come from a nine-item toy corpus with judgements written for the
demonstration. They show the harness works. They are not evaluation results.

---

## Verification status

Be aware of what has and has not been executed.

**Tested — 41 passing tests:** domain models, both embedding backends, in-memory
vector index and metadata filtering, deduplication, profile accumulation, decay,
negative feedback, cold-start handling, ranking weights, relevance floor,
coverage checking, discovery triggering, source-failure tolerance, RSS and Atom
parsing, and all evaluation metrics against hand-computed values. The REST API
was smoke-tested across every endpoint including error paths.

**Written but not executed:**

- `storage/postgres.py` — no PostgreSQL available in the authoring environment.
  Written against the psycopg 3 and pgvector APIs; run against a live database
  before relying on it.
- `sources/youtube.py` — no API key and no outbound access to googleapis.com.
- `SentenceTransformerEmbeddingService` — the model could not be downloaded, so
  the test suite ran on the hashing backend. The code path is standard
  `sentence-transformers` usage, but **run `demo.py` with the real model
  installed before reporting anything**, since no result in this repository has
  yet been produced by a semantic model.

## Known limitations

- **RSS cannot be searched.** Feeds expose no search endpoint, so that adapter
  filters locally and can only surface content from feeds already subscribed to.
  A keyed search API is the natural complement.
- **Tuning parameters are placeholders.** The coverage threshold, relevance
  floor, ranking weights and half-lives are defensible starting values, not
  empirical results. Chapter Four must report the values actually used and how
  they were arrived at.
- **No authentication.** The API has no auth layer and is not deployment-ready
  as it stands.
- **Single-process in-memory backend.** The default backend does not persist or
  share state across processes; use PostgreSQL for anything beyond a demo.
- **Exact search does not scale indefinitely.** The in-memory index is
  brute-force. This is fine at project scale and gives an exact ground truth for
  checking HNSW recall, but production scale needs the indexed backend.

## Schema note

`user_profiles.interaction_count` is an addition to Table 3.4 as printed in
Chapter Three. It lets the ranker distinguish a cold-start user from an
established one without counting interaction rows on every query. Chapter Three
should be amended to include it.
