-- NULL means enrichment has not run. Original catalogue fields stay intact.
ALTER TABLE products ADD COLUMN IF NOT EXISTS attributes jsonb;
ALTER TABLE products ADD CONSTRAINT products_attributes_document_check
    CHECK (attributes IS NULL OR
        (jsonb_typeof(attributes) = 'object'
         AND attributes ? 'schema_version'
         AND attributes->'schema_version' = '1'::jsonb
         AND attributes ? 'attributes'
         AND jsonb_typeof(attributes->'attributes') = 'object'));
