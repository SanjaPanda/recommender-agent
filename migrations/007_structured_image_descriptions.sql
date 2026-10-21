-- Store validated image observations as queryable nullable apparel fields.
-- The readable image_description and its existing vector remain compatible.
ALTER TABLE products ADD COLUMN image_product_type text;
ALTER TABLE products ADD COLUMN image_colours text[];
ALTER TABLE products ADD COLUMN image_pattern text;
ALTER TABLE products ADD COLUMN image_components text[];
ALTER TABLE products ADD COLUMN image_object_count integer;
ALTER TABLE products ADD COLUMN image_other_details text;
ALTER TABLE products ADD CONSTRAINT products_image_object_count_check
    CHECK (image_object_count IS NULL OR image_object_count > 0);
