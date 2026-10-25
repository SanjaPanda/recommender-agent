-- Keep generated visual descriptions separate from original catalogue text.
ALTER TABLE products ADD COLUMN IF NOT EXISTS image_description text;
ALTER TABLE products ADD COLUMN IF NOT EXISTS image_description_meta jsonb;
ALTER TABLE product_embeddings DROP CONSTRAINT IF EXISTS product_embeddings_modality_check;
ALTER TABLE product_embeddings ADD CONSTRAINT product_embeddings_modality_check
    CHECK (modality IN ('text', 'image', 'image_description'));
