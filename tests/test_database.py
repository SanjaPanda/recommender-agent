"""Opt-in PostgreSQL contract tests using authored rows, rolled back after each test.

Set RUN_DATABASE_TESTS=1 and run the normal unittest discovery command after
applying migrations. Routine offline discovery does not connect to PostgreSQL.
"""
import os
import unittest
import uuid

import psycopg
from psycopg.types.json import Jsonb

from agent.config import settings


@unittest.skipUnless(os.environ.get("RUN_DATABASE_TESTS") == "1", "Opt-in PostgreSQL integration tests")
class DatabaseContractTests(unittest.TestCase):
    """Validate actual database constraints without leaving catalog fixtures behind."""

    def test_cart_ids_are_excluded_from_browsing_before_limit(self):
        from rag.retrieval import retrieve_products
        baseline = retrieve_products(limit=3)
        if not baseline:
            self.skipTest('Read-only browsing check requires catalog records.')
        excluded = [baseline[0]['id']]
        rows = retrieve_products(limit=3, excluded_product_ids=excluded)
        self.assertNotIn(excluded[0], [row['id'] for row in rows])
        expected = self.connection.execute('SELECT id FROM products WHERE id <> ALL(%s) ORDER BY id LIMIT 3',
                                           (excluded,)).fetchall()
        self.assertEqual([row['id'] for row in rows], [row[0] for row in expected])

    def test_catalog_revision_trigger_includes_truncate_and_rolls_back(self):
        definition = self.connection.execute(
            "SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE tgname='products_catalog_revision'"
        ).fetchone()[0]
        self.assertIn('TRUNCATE', definition)
        self.connection.execute('CREATE TEMP TABLE revision_truncate_fixture(id integer)')
        self.connection.execute('CREATE TRIGGER fixture_revision AFTER TRUNCATE ON revision_truncate_fixture '
                                'FOR EACH STATEMENT EXECUTE FUNCTION bump_catalog_revision()')
        before = self.connection.execute('SELECT revision FROM catalog_revision').fetchone()[0]
        with self.connection.transaction(force_rollback=True):
            self.connection.execute('TRUNCATE revision_truncate_fixture')
            self.assertEqual(self.connection.execute('SELECT revision FROM catalog_revision').fetchone()[0], before + 1)
        self.assertEqual(self.connection.execute('SELECT revision FROM catalog_revision').fetchone()[0], before)

    def test_open_product_phrase_regex_matches_python_and_postgresql(self):
        from rag.phrases import phrase_in_title, phrase_pattern
        for phrase, title, expected in (
            ('dress', 'Blue Dress Shoes', False),
            ('dress', 'Blue Dress, Size 8', True),
            ('ear cuff', 'Silver Ear-Cuffs', True),
            ('sling bag', 'Leather Sling Bags', True),
            ('bow tie', 'Blue Bowtie', True),
            ('bow tie', 'Blue Necktie', False),
            ('mary jane shoes', 'Mary Jane Shoes', True),
            ('bow tie', 'Shirt with Bow Tie', False),
        ):
            with self.subTest(phrase=phrase, title=title):
                actual = self.connection.execute(
                    "SELECT (regexp_split_to_array(%s, '\\s+(?:for|with|fits|similar to)\\s+', 'i'))[1] ~* %s",
                    (title, r'\m' + phrase_pattern(phrase) + r'\M(?=$|\s*[^[:alpha:][:space:]])'),
                ).fetchone()[0]
                self.assertEqual(actual, expected)
                self.assertEqual(phrase_in_title(title, phrase), expected)

    def setUp(self):
        self.connection = psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5)
        self.addCleanup(self.connection.close)
        self.addCleanup(self.connection.rollback)
        self.connection.execute("BEGIN")
        self.product_id = "authored-test-" + uuid.uuid4().hex

    def insert_product(self, category="Apparel & Accessories > Clothing"):
        """Insert one authored product using the existing runtime field layout."""
        self.connection.execute(
            "INSERT INTO products (id,title,raw_source,source_name) VALUES (%s,%s,%s,%s)",
            (self.product_id, "Authored test shirt", Jsonb({"ground_truth_category": category}), "authored-test"),
        )

    def test_taxonomy_rejects_non_apparel_and_missing_roots(self):
        for category in ("Home & Garden", None, "Apparel & Accessories Extra"):
            with self.subTest(category=category), self.assertRaises(psycopg.errors.CheckViolation):
                with self.connection.transaction():
                    self.insert_product(category)

    def test_source_requires_evidence_and_product_provenance(self):
        with self.assertRaises(psycopg.errors.CheckViolation):
            with self.connection.transaction():
                self.connection.execute(
                    "INSERT INTO dataset_sources (id,name,revision,source_url,rights_status) "
                    "VALUES (%s,%s,'v1','authored:fixture','permitted')",
                    (self.product_id, self.product_id),
                )
        self.connection.execute(
            "INSERT INTO dataset_sources (id,name,revision,source_url,rights_status,rights_evidence) "
            "VALUES (%s,%s,'v1','authored:fixture','authored','Authored for database contract testing')",
            (self.product_id, self.product_id),
        )
        self.insert_product()
        with self.assertRaises(psycopg.errors.CheckViolation):
            with self.connection.transaction():
                self.connection.execute("UPDATE products SET dataset_source_id=%s WHERE id=%s",
                                        (self.product_id, self.product_id))
        self.connection.execute(
            "UPDATE products SET dataset_source_id=%s,source_product_id='shirt',"
            "source_line=1,source_record_sha256=%s WHERE id=%s",
            (self.product_id, "a" * 64, self.product_id),
        )

    def test_vector_dimensions_are_derived_and_deletion_cascades(self):
        self.insert_product()
        self.connection.execute(
            "INSERT INTO product_embeddings (product_id,modality,model_name,model_version,embedding) "
            "VALUES (%s,'text','authored-model','test-v1','[1,0,0]')", (self.product_id,),
        )
        self.assertEqual(self.connection.execute(
            "SELECT dimensions FROM product_embeddings WHERE product_id=%s", (self.product_id,)
        ).fetchone()[0], 3)
        self.connection.execute("DELETE FROM products WHERE id=%s", (self.product_id,))
        self.assertEqual(self.connection.execute(
            "SELECT count(*) FROM product_embeddings WHERE product_id=%s", (self.product_id,)
        ).fetchone()[0], 0)

    def test_invalid_hash_price_and_feature_shapes_are_rejected(self):
        self.insert_product()
        for assignment in ("image_sha256='invalid'", "historical_price_usd=-1",
                           "historical_price_usd='NaN'", "features='{}'::jsonb"):
            with self.subTest(assignment=assignment), self.assertRaises(psycopg.errors.CheckViolation):
                with self.connection.transaction():
                    self.connection.execute(f"UPDATE products SET {assignment} WHERE id=%s", (self.product_id,))
