-- Remove the retired image-description pipeline after a verified database backup.
-- This deletes caption vectors and visual-description fields, preserving products,
-- text vectors, CLIP image vectors, sessions, and all source product images.
DELETE FROM product_embeddings WHERE modality = 'image_description';

ALTER TABLE product_embeddings DROP CONSTRAINT IF EXISTS product_embeddings_modality_check;

ALTER TABLE product_embeddings ADD CONSTRAINT product_embeddings_modality_check
    CHECK (modality IN ('text', 'image'));

ALTER TABLE products DROP CONSTRAINT IF EXISTS products_image_object_count_check;

ALTER TABLE products
    DROP COLUMN IF EXISTS image_description,
    DROP COLUMN IF EXISTS image_description_meta,
    DROP COLUMN IF EXISTS image_product_type,
    DROP COLUMN IF EXISTS image_colours,
    DROP COLUMN IF EXISTS image_pattern,
    DROP COLUMN IF EXISTS image_components,
    DROP COLUMN IF EXISTS image_object_count,
    DROP COLUMN IF EXISTS image_other_details;
