-- Add source-neutral provenance without changing existing runtime interfaces.
CREATE TABLE dataset_sources (
    id text PRIMARY KEY CHECK (btrim(id) <> ''),
    name text NOT NULL CHECK (btrim(name) <> ''),
    revision text NOT NULL CHECK (btrim(revision) <> ''),
    source_url text NOT NULL CHECK (btrim(source_url) <> ''),
    rights_status text NOT NULL DEFAULT 'unresolved'
        CHECK (rights_status IN ('unresolved', 'permitted', 'restricted', 'authored')),
    rights_evidence text,
    curation_config jsonb NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(curation_config) = 'object'),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (name, revision),
    CHECK (rights_status NOT IN ('permitted', 'authored')
        OR (rights_evidence IS NOT NULL AND btrim(rights_evidence) <> ''))
);

ALTER TABLE products
    ADD COLUMN dataset_source_id text REFERENCES dataset_sources(id),
    ADD COLUMN source_product_id text,
    ADD COLUMN source_line bigint CHECK (source_line > 0),
    ADD COLUMN source_record_sha256 text CHECK (source_record_sha256 ~ '^[0-9a-f]{64}$'),
    ADD COLUMN features jsonb NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(features) = 'array'),
    ADD COLUMN historical_price_usd numeric(12,2)
        CHECK (historical_price_usd >= 0 AND historical_price_usd <> 'NaN'::numeric),
    ADD COLUMN image_sha256 text CHECK (image_sha256 ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT products_source_identity_check CHECK (
        dataset_source_id IS NULL OR
        (source_product_id IS NOT NULL AND btrim(source_product_id) <> ''
         AND source_line IS NOT NULL AND source_record_sha256 IS NOT NULL)),
    ADD CONSTRAINT products_dataset_identity_unique UNIQUE (dataset_source_id, source_product_id),
    ADD CONSTRAINT products_apparel_taxonomy_check CHECK (
        raw_source->>'ground_truth_category' IS NOT NULL AND
        (raw_source->>'ground_truth_category' = 'Apparel & Accessories'
         OR starts_with(raw_source->>'ground_truth_category', 'Apparel & Accessories > ')));

ALTER TABLE product_embeddings
    ADD COLUMN dimensions integer GENERATED ALWAYS AS (vector_dims(embedding)) STORED,
    ADD COLUMN preprocessing_sha256 text
        CHECK (preprocessing_sha256 ~ '^[0-9a-f]{64}$'),
    ADD COLUMN input_sha256 text CHECK (input_sha256 ~ '^[0-9a-f]{64}$');
