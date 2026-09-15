-- IntelliVerse database schema
-- PostgreSQL 14+ with the pgvector extension.
-- Implements Tables 3.3-3.6 of Chapter Three.
--
-- Vector dimension is 384, matching all-MiniLM-L6-v2. Changing the embedding
-- model means changing this dimension and reindexing the corpus.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table 3.3 --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        VARCHAR(255) NOT NULL,
    email       VARCHAR(255) NOT NULL UNIQUE,
    joined_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Table 3.5 --------------------------------------------------------------
-- Declared before user_profiles only for readability; there is no FK between
-- them. source_url is UNIQUE, which is what makes ingestion idempotent: the
-- same item rediscovered from the same URL updates rather than duplicates.
CREATE TABLE IF NOT EXISTS content_items (
    content_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title           TEXT         NOT NULL,
    description     TEXT,
    content_type    VARCHAR(16)  NOT NULL
                    CHECK (content_type IN ('video', 'audio', 'text')),
    source_platform VARCHAR(255) NOT NULL,
    source_url      TEXT         NOT NULL UNIQUE,
    topics          JSONB        NOT NULL DEFAULT '[]'::jsonb,
    embedding       VECTOR(384),
    published_at    TIMESTAMPTZ,
    discovered_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Table 3.4 --------------------------------------------------------------
-- NOTE: interaction_count is an addition to Table 3.4 as printed in Chapter
-- Three. It is required so that the ranker can distinguish a cold-start user
-- from an established one without counting rows in `interactions` on every
-- query. Chapter Three should be amended to include it.
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id                  UUID PRIMARY KEY
                             REFERENCES users(user_id) ON DELETE CASCADE,
    interest_embedding       VECTOR(384),
    content_type_preferences JSONB       NOT NULL DEFAULT '{}'::jsonb,
    interaction_count        INTEGER     NOT NULL DEFAULT 0,
    last_updated             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table 3.6 --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id          UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    content_id       UUID NOT NULL REFERENCES content_items(content_id) ON DELETE CASCADE,
    interaction_type VARCHAR(16) NOT NULL
                     CHECK (interaction_type IN ('click', 'view', 'completion',
                                                 'save', 'like', 'dislike', 'skip')),
    strength         REAL        NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes ----------------------------------------------------------------

-- HNSW over cosine distance (Malkov & Yashunin, 2020). vector_cosine_ops
-- matches the <=> operator used by the retrieval query. m and
-- ef_construction trade index build time against recall; these are pgvector's
-- defaults and are an evaluation parameter, not a settled choice.
CREATE INDEX IF NOT EXISTS idx_content_embedding_hnsw
    ON content_items USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Supports the content-type metadata filter applied alongside similarity
-- search, which is the requirement that motivated choosing pgvector over a
-- filtering-less vector library.
CREATE INDEX IF NOT EXISTS idx_content_type        ON content_items (content_type);
CREATE INDEX IF NOT EXISTS idx_content_discovered  ON content_items (discovered_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_topics_gin  ON content_items USING gin (topics);

-- Profile rebuilds and per-user history reads both scan by user and time.
CREATE INDEX IF NOT EXISTS idx_interactions_user
    ON interactions (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_content ON interactions (content_id);
