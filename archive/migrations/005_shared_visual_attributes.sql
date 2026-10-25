-- Keep historical v1 documents readable. v2 uses one compact visual schema.
ALTER TABLE products DROP CONSTRAINT products_attributes_document_check;
ALTER TABLE products ADD CONSTRAINT products_attributes_document_check
    CHECK (attributes IS NULL OR
        (jsonb_typeof(attributes) = 'object'
         AND attributes ? 'schema_version'
         AND attributes->'schema_version' IN ('1'::jsonb, '2'::jsonb)
         AND attributes ? 'attributes'
         AND jsonb_typeof(attributes->'attributes') = 'object'));
