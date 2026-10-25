-- Every catalog writer must supply the product's source identity explicitly.
ALTER TABLE products ALTER COLUMN source_name DROP DEFAULT;
