-- Commit-visible revision for process-local derived catalog caches.
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
