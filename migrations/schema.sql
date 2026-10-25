-- PostgreSQL/pgvector schema for the apparel recommendation application.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE dataset_sources (
    id text PRIMARY KEY CHECK (btrim(id) <> ''),
    name text NOT NULL CHECK (btrim(name) <> ''),
    revision text NOT NULL CHECK (btrim(revision) <> ''),
    source_url text NOT NULL CHECK (btrim(source_url) <> ''),
    rights_status text NOT NULL DEFAULT 'unresolved'
        CHECK (rights_status IN ('unresolved', 'permitted', 'personal_use', 'restricted', 'authored')),
    rights_evidence text,
    curation_config jsonb NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(curation_config) = 'object'),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (name, revision),
    CHECK (rights_status NOT IN ('permitted', 'authored')
        OR (rights_evidence IS NOT NULL AND btrim(rights_evidence) <> '')),
    CONSTRAINT dataset_sources_personal_use_evidence_check CHECK (
        rights_status <> 'personal_use' OR
        (rights_evidence IS NOT NULL AND btrim(rights_evidence) <> ''))
);

CREATE TABLE products (
    id text PRIMARY KEY,
    title text,
    description text,
    brand text,
    category text,
    is_secondhand boolean,
    image jsonb,
    raw_source jsonb NOT NULL,
    source_name text NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    attributes jsonb,
    dataset_source_id text REFERENCES dataset_sources(id),
    source_product_id text,
    source_line bigint CHECK (source_line > 0),
    source_record_sha256 text CHECK (source_record_sha256 ~ '^[0-9a-f]{64}$'),
    features jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(features) = 'array'),
    historical_price_usd numeric(12,2)
        CHECK (historical_price_usd >= 0 AND historical_price_usd <> 'NaN'::numeric),
    image_sha256 text CHECK (image_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT products_attributes_document_check CHECK (
        attributes IS NULL OR
        (jsonb_typeof(attributes) = 'object'
         AND attributes ? 'schema_version'
         AND attributes->'schema_version' IN ('1'::jsonb, '2'::jsonb)
         AND attributes ? 'attributes'
         AND jsonb_typeof(attributes->'attributes') = 'object')),
    CONSTRAINT products_source_identity_check CHECK (
        dataset_source_id IS NULL OR
        (source_product_id IS NOT NULL AND btrim(source_product_id) <> ''
         AND source_line IS NOT NULL AND source_record_sha256 IS NOT NULL)),
    CONSTRAINT products_dataset_identity_unique UNIQUE (dataset_source_id, source_product_id),
    CONSTRAINT products_apparel_taxonomy_check CHECK (
        raw_source->>'ground_truth_category' IS NOT NULL AND
        (raw_source->>'ground_truth_category' = 'Apparel & Accessories'
         OR starts_with(raw_source->>'ground_truth_category', 'Apparel & Accessories > ')))
);

CREATE INDEX products_brand_idx ON products (brand);
CREATE INDEX products_category_idx ON products (category);

CREATE TABLE product_embeddings (
    product_id text NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    modality text NOT NULL CHECK (modality IN ('text', 'image')),
    model_name text NOT NULL,
    model_version text,
    embedding vector NOT NULL,
    status text NOT NULL DEFAULT 'ready' CHECK (status IN ('ready', 'failed')),
    error text,
    updated_at timestamptz NOT NULL DEFAULT now(),
    dimensions integer GENERATED ALWAYS AS (vector_dims(embedding)) STORED,
    preprocessing_sha256 text CHECK (preprocessing_sha256 ~ '^[0-9a-f]{64}$'),
    input_sha256 text CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (product_id, modality, model_name)
);

CREATE TABLE agent_sessions (
    thread_id text PRIMARY KEY,
    state jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Persistence structures retained for database compatibility; unused by requests.
CREATE TABLE recommendation_sessions (
    id uuid PRIMARY KEY,
    preferences jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE product_feedback (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES recommendation_sessions(id) ON DELETE CASCADE,
    product_id text NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    action text NOT NULL CHECK (action IN ('selected', 'liked', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX product_feedback_session_time_idx ON product_feedback (session_id, created_at DESC);

CREATE TABLE catalog_revision (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0)
);
INSERT INTO catalog_revision(singleton, revision) VALUES (true, 0);

CREATE FUNCTION bump_catalog_revision() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE catalog_revision SET revision = revision + 1 WHERE singleton = true;
    RETURN NULL;
END;
$$;

CREATE TRIGGER products_catalog_revision
AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON products
FOR EACH STATEMENT EXECUTE FUNCTION bump_catalog_revision();
