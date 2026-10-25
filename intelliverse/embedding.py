"""Embedding services.

Chapter Three, Section 3.2.2.2 selects ``all-MiniLM-L6-v2`` via
``sentence-transformers``. That is :class:`SentenceTransformerEmbeddingService`
below and is what the deployed system uses.

:class:`HashingEmbeddingService` exists because the model weights cannot always
be downloaded (restricted networks, CI runners, air-gapped marking
environments). It is a *lexical* fallback: it maps overlapping vocabulary to
similar vectors. That is enough to exercise the pipeline end to end, but it is
emphatically **not** a semantic model, and any evaluation figure produced with
it would measure the opposite of what this project claims. Results reported in
Chapter Four must come from the Sentence-BERT backend.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable, Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe against zero vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbeddingService(Protocol):
    """Converts text into a vector in the system's shared semantic space."""

    dimension: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbeddingService:
    """Sentence-BERT backend — the production implementation.

    Embeddings are L2-normalised at generation time so that cosine similarity
    reduces to a dot product, which is what both pgvector's ``<=>`` operator and
    the in-memory index rely on.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "sentence-transformers is required for the production embedding "
                "backend. Install it with `pip install sentence-transformers`, "
                "or use HashingEmbeddingService for offline development."
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimension = int(self._model.get_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


class HashingEmbeddingService:
    """Deterministic lexical fallback for offline development and tests.

    Uses the hashing trick: each token is hashed to a bucket and accumulated
    with sub-linear term weighting. Texts sharing vocabulary land near each
    other; texts expressing the same idea in different words do not. This is
    precisely the vocabulary-mismatch failure Chapter One identifies, reproduced
    here on purpose so that the difference between the two backends is visible
    rather than hidden.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.model_name = f"hashing-{dimension}d"

    @staticmethod
    def _tokenise(text: str) -> Iterable[str]:
        return _TOKEN_RE.findall(text.lower())

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        counts = Counter(self._tokenise(text))
        if not counts:
            return vector
        for token, count in counts.items():
            # Sub-linear term weighting, as in classical tf weighting, so that
            # a repeated word does not dominate the representation.
            vector[self._bucket(token)] += 1.0 + np.log(count)
        norm = float(np.linalg.norm(vector))
        return vector if norm < 1e-9 else vector / norm

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


def build_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    allow_fallback: bool = True,
) -> EmbeddingService:
    """Return the Sentence-BERT service, falling back only if permitted.

    ``allow_fallback`` defaults to True for developer convenience but should be
    set False anywhere a result is being measured, so that a silent downgrade to
    lexical matching cannot be mistaken for semantic retrieval.
    """
    try:
        return SentenceTransformerEmbeddingService(model_name)
    except Exception as exc:
        if not allow_fallback:
            raise
        import warnings

        warnings.warn(
            f"Could not load '{model_name}' ({exc.__class__.__name__}). Falling "
            "back to HashingEmbeddingService, which performs LEXICAL matching "
            "only. Do not report evaluation results obtained this way.",
            RuntimeWarning,
            stacklevel=2,
        )
        return HashingEmbeddingService()