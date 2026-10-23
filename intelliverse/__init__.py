"""IntelliVerse — AI-powered multimodal content discovery and recommendation.

A discovery and recommendation layer over external content platforms. It does
not host media: it indexes metadata, represents it semantically, and ranks it
for the individual user.
"""

from .config import RankingWeights, Settings
from .discovery import DiscoveryService
from .embedding import (
    HashingEmbeddingService,
    SentenceTransformerEmbeddingService,
    build_embedding_service,
)
from .engine import IntelliVerse, SearchResponse, build_default_engine
from .models import (
    ContentItem,
    ContentType,
    Interaction,
    InteractionType,
    ScoredContent,
    User,
    UserProfile,
)
from .profile import UserProfileService
from .ranking import RecommendationEngine

__version__ = "0.1.0"

__all__ = [
    "IntelliVerse",
    "SearchResponse",
    "build_default_engine",
    "Settings",
    "RankingWeights",
    "ContentItem",
    "ContentType",
    "Interaction",
    "InteractionType",
    "User",
    "UserProfile",
    "ScoredContent",
    "DiscoveryService",
    "UserProfileService",
    "RecommendationEngine",
    "build_embedding_service",
    "SentenceTransformerEmbeddingService",
    "HashingEmbeddingService",
]
