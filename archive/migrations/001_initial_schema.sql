-- Initial catalogue/session schema. Embedding dimensions follow the configured
-- model. The unconstrained pgvector type avoids guessing a dimension in advance.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS products (
    id text PRIMARY KEY,
    title text,
    description text,
    brand text,
    category text,
    is_secondhand boolean,
    image jsonb,
    raw_source jsonb NOT NULL,
    source_name text NOT NULL DEFAULT 'Shopify/product-catalogue',
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS products_brand_idx ON products (brand);
CREATE INDEX IF NOT EXISTS products_category_idx ON products (category);

CREATE TABLE IF NOT EXISTS product_embeddings (
    product_id text NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    modality text NOT NULL CHECK (modality IN ('text', 'image')),
    model_name text NOT NULL,
    model_version text,
    embedding vector NOT NULL,
    status text NOT NULL DEFAULT 'ready' CHECK (status IN ('ready', 'failed')),
    error text,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (product_id, modality, model_name)
);

CREATE TABLE IF NOT EXISTS recommendation_sessions (
    id uuid PRIMARY KEY,
    preferences jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS product_feedback (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES recommendation_sessions(id) ON DELETE CASCADE,
    product_id text NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    action text NOT NULL CHECK (action IN ('selected', 'liked', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS product_feedback_session_time_idx
    ON product_feedback (session_id, created_at DESC);
