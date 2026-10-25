"""Authored metadata/image fixtures for the bounded curated catalog importer."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import psycopg
from PIL import Image

from agent.config import settings
from rag.catalog import Manifest, historical_price, image_urls, ingest, local_asset, normalize
from rag.embed_text import document_text


def fixture(root):
    """Create one local authored image, rich record and source manifest."""
    Image.new("RGB", (224, 224), "blue").save(root / "shirt.png")
    row = {"parent_asin": "AUTHORED1", "title": "Authored cotton shirt",
           "main_category": "Clothing, Shoes & Jewelry", "categories": ["Women", "Clothing"],
           "description": [" ".join("authored" + str(i) for i in range(55))],
           "features": ["Cotton fabric", "Long sleeves"],
           "details": {"Brand": "Fixture Brand", "Material": "Cotton"}, "price": "29.99",
           "store": "A different seller", "images": [{"variant": "MAIN", "large": "https://example.invalid/shirt.png"}]}
    config = {"source_id": "authored-" + uuid.uuid4().hex, "source_name": "Authored metadata",
              "revision": "authored-v1", "source_url": "authored:fixture", "rights_status": "authored",
              "rights_evidence": "Created locally for tests", "records": "records.jsonl",
              "images": {"AUTHORED1": {"local_path": "shirt.png", "source_url": "https://example.invalid/shirt.png"}}}
    (root / "records.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(config), encoding="utf-8")
    return row, config


class CatalogTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.row, self.config = fixture(self.root)
        self.manifest = Manifest(**self.config)

    def test_source_fields_are_preserved_and_store_is_not_brand(self):
        self.row["details"] = json.dumps(self.row["details"])
        product = normalize(self.row, self.manifest, self.root)
        self.assertEqual(product["brand"], "Fixture Brand")
        self.assertEqual(product["category"], "Apparel & Accessories > Clothing")
        self.assertEqual(product["features"].obj, self.row["features"])
        self.assertEqual(product["raw_source"].obj["details"], self.row["details"])
        self.assertEqual(product["image"].obj["width"], 224)

    def test_real_fashion_domain_requires_explicit_apparel_category_root(self):
        self.row['main_category'] = 'AMAZON FASHION'
        self.row['categories'] = ['Clothing, Shoes & Jewelry', 'Women', 'Clothing']
        self.assertEqual(normalize(self.row, self.manifest, self.root)['category'],
                         'Apparel & Accessories > Clothing')
        self.row['categories'] = ['Home & Kitchen', 'Clothing']
        with self.assertRaisesRegex(ValueError, 'taxonomy'):
            normalize(self.row, self.manifest, self.root)

    def test_features_do_not_fill_a_missing_description(self):
        self.row["description"] = []
        self.row["features"] = ["rich features " * 100]
        with self.assertRaisesRegex(ValueError, "short_description"):
            normalize(self.row, self.manifest, self.root)

    def test_outside_taxonomy_and_unknown_branches_are_rejected(self):
        for categories, main in [(["Clothing"], "Home & Kitchen"), (["Watch Bands"], "Clothing, Shoes & Jewelry")]:
            with self.subTest(categories=categories):
                self.row.update(categories=categories, main_category=main)
                with self.assertRaisesRegex(ValueError, "taxonomy"):
                    normalize(self.row, self.manifest, self.root)

    def test_images_require_source_provenance_and_decodable_dimensions(self):
        self.config["images"]["AUTHORED1"]["source_url"] = "https://example.invalid/unrelated.png"
        with self.assertRaisesRegex(ValueError, "image_provenance"):
            normalize(self.row, Manifest(**self.config), self.root)
        Image.new("RGB", (100, 100)).save(self.root / "shirt.png")
        with self.assertRaisesRegex(ValueError, "image_dimensions"):
            normalize(self.row, self.manifest, self.root)
        (self.root / "shirt.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "image_decode"):
            normalize(self.row, self.manifest, self.root)

    def test_asset_paths_cannot_escape_and_api_serves_catalog_images(self):
        for path in ("../outside.png", str(self.root / "shirt.png")):
            with self.assertRaisesRegex(ValueError, "asset_path"):
                local_asset(self.root, path)
        from agent.api import _find_image_path
        target = self.root / ".data" / "catalog"
        target.mkdir(parents=True)
        (target / "shirt.png").write_bytes((self.root / "shirt.png").read_bytes())
        with patch("agent.api.ROOT", self.root):
            self.assertEqual(_find_image_path({"local_path": "shirt.png"}), target / "shirt.png")
            self.assertIsNone(_find_image_path({"local_path": "../shirt.png"}))

    def test_price_unknowns_and_image_layout_variants(self):
        for value in (None, "Unknown", "NaN", "Infinity", -1, True):
            self.assertIsNone(historical_price(value))
        self.assertEqual(str(historical_price("12.50")), "12.50")
        self.assertEqual(image_urls({"large": ["https://example.invalid/image"]}), {"https://example.invalid/image"})

    def test_manifest_requires_pinned_and_recorded_source_decision(self):
        for override in ({"revision": "main"}, {"rights_status": "unresolved"}, {"rights_evidence": " "}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                Manifest(**{**self.config, **override})

    def test_source_features_enter_embedding_text(self):
        self.assertEqual(document_text("Shirt", "Cotton", ["Long sleeves"]),
                         "search_document: Shirt\nCotton\nLong sleeves")


@unittest.skipUnless(os.environ.get("RUN_DATABASE_TESTS") == "1", "Opt-in PostgreSQL integration tests")
class CatalogDatabaseTests(unittest.TestCase):
    def test_import_holds_maintenance_lock_until_transaction_ends(self):
        from rag import CATALOG_WRITE_LOCK
        ingest(self.connection, self.root / "manifest.json")
        with psycopg.connect(settings.database_url.get_secret_value(), autocommit=True) as other:
            self.assertFalse(other.execute("SELECT pg_try_advisory_lock(%s)", (CATALOG_WRITE_LOCK,)).fetchone()[0])
            self.connection.rollback()
            self.assertTrue(other.execute("SELECT pg_try_advisory_lock(%s)", (CATALOG_WRITE_LOCK,)).fetchone()[0])
            other.execute("SELECT pg_advisory_unlock(%s)", (CATALOG_WRITE_LOCK,))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.row, self.config = fixture(self.root)
        self.connection = psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5)
        self.addCleanup(self.connection.close)
        self.addCleanup(self.connection.rollback)
        self.connection.execute("BEGIN")

    def test_idempotency_provenance_and_changed_records_invalidate_vectors(self):
        path = self.root / "manifest.json"
        self.assertEqual(ingest(self.connection, path)["accepted"], 1)
        product_id = self.config["source_id"] + ":AUTHORED1"
        self.connection.execute("INSERT INTO product_embeddings(product_id,modality,model_name,embedding) "
                                "VALUES (%s,'text','authored-model','[1,0,0]')", (product_id,))
        self.assertEqual(ingest(self.connection, path)["accepted"], 1)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM product_embeddings WHERE product_id=%s", (product_id,)).fetchone()[0], 1)
        self.row["description"].append("A changed source description.")
        (self.root / "records.jsonl").write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        ingest(self.connection, path)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM products WHERE id=%s", (product_id,)).fetchone()[0], 1)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM product_embeddings WHERE product_id=%s", (product_id,)).fetchone()[0], 0)
        self.config["revision"] = "different-revision"
        path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different provenance"):
            ingest(self.connection, path)

    def test_scan_bounds_and_rejection_counts(self):
        path = self.root / "manifest.json"
        self.config["max_lines"] = 3
        path.write_text(json.dumps(self.config), encoding="utf-8")
        (self.root / "records.jsonl").write_text("invalid json\n" + json.dumps(self.row) + "\n" + json.dumps(self.row) + "\n", encoding="utf-8")
        report = ingest(self.connection, path)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["scanned"], 3)
        self.assertEqual(report["rejected"], {"invalid_record": 1, "duplicate_id": 1})
        self.assertEqual(report["stop_reason"], "line_limit")

    def test_expansion_increases_only_operational_limits_and_retains_vectors(self):
        path = self.root / "manifest.json"
        self.config["max_products"] = 1
        path.write_text(json.dumps(self.config), encoding="utf-8")
        ingest(self.connection, path)
        product_id = self.config["source_id"] + ":AUTHORED1"
        self.connection.execute("INSERT INTO product_embeddings(product_id,modality,model_name,embedding) "
                                "VALUES (%s,'text','authored-model','[1,0,0]')", (product_id,))
        self.config["max_products"] = 2000
        path.write_text(json.dumps(self.config), encoding="utf-8")
        self.assertEqual(ingest(self.connection, path)["accepted"], 1)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM product_embeddings WHERE product_id=%s", (product_id,)).fetchone()[0], 1)
        self.config["min_description_words"] = 60
        path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different provenance"):
            ingest(self.connection, path)

    def test_bm25_searches_features_in_filtered_database_and_excludes_seeds(self):
        from rag.retrieval import _search_bm25, filter_clause
        from psycopg.types.json import Jsonb
        self.row["details"]["Brand"] = self.config["source_id"]
        self.row["features"] = ["Waterproof needleword cotton"]
        (self.root / "records.jsonl").write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        ingest(self.connection, self.root / "manifest.json")
        clause, params = filter_clause({"brand": self.config["source_id"]}, [])
        kwargs = dict(query="needleword", category_clause=clause, category_params=params, limit=5)
        rows = _search_bm25(self.connection, excluded_ids=[], **kwargs)
        product_id = self.config["source_id"] + ":AUTHORED1"
        self.assertEqual([r[0] for r in rows], [product_id])
        self.assertGreater(rows[0][6], 0)
        self.assertEqual(_search_bm25(self.connection, excluded_ids=[product_id], **kwargs), [])
        self.connection.execute('UPDATE products SET features=%s WHERE id=%s', (Jsonb(['Freshword linen']), product_id))
        self.assertEqual(_search_bm25(self.connection, excluded_ids=[], **kwargs), [])
        refreshed = _search_bm25(self.connection, excluded_ids=[], **{**kwargs, 'query': 'freshword'})
        self.assertEqual([r[0] for r in refreshed], [product_id])
        self.assertEqual(refreshed[0][7], ['Freshword linen'])
        self.connection.execute('DELETE FROM products WHERE id=%s', (product_id,))
        self.assertEqual(_search_bm25(self.connection, excluded_ids=[], **{**kwargs, 'query': 'freshword'}), [])

    def test_byte_budget_stops_before_processing_an_oversized_record(self):
        self.config["max_bytes"] = 10
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.config), encoding="utf-8")
        report = ingest(self.connection, path)
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["scanned"], 0)
        self.assertEqual(report["stop_reason"], "line_or_byte_limit")
        self.assertEqual(report["deficit"], self.config.get("max_products", 2000))
