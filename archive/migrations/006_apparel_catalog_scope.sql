-- Keep only the active Apparel & Accessories catalog. Foreign keys cascade
-- removal of product embeddings and any product feedback for deleted products.
DELETE FROM products
WHERE NOT (
    raw_source->>'ground_truth_category' = 'Apparel & Accessories'
    OR starts_with(raw_source->>'ground_truth_category', 'Apparel & Accessories > ')
);
