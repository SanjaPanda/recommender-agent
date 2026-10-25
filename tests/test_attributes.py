"""Exact evidence, ambiguity and database preservation tests for extraction."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import psycopg

from agent.config import settings
from rag.attributes import enrich, extract
from rag.catalog import ingest
from test_catalog import fixture


class AttributeTests(unittest.TestCase):
    def test_exact_source_offsets_survive_html_and_lists(self):
        row = {"features": ["<li>95% Cotton, 5% Spandex</li>", "Machine washable"],
               "description": ["Relaxed fit with button closure."], "details": {"Material": " Cotton "}}
        report = extract(row)
        self.assertTrue(report["candidates"])
        for candidate in report["candidates"]:
            source = candidate["source"]
            text = row[source["field"]][source.get("index", source.get("key"))]
            self.assertEqual(text[candidate["start"]:candidate["end"]], candidate["matched_text"])
            self.assertEqual(candidate["review_status"], "unreviewed")
        self.assertIn("composition_mention", {c["field"] for c in report["candidates"]})

    def test_components_faux_and_negation_are_not_full_material_facts(self):
        result = extract({"features": ["Faux leather upper with cotton lining. Without wool."]})
        mentions = [c for c in result["candidates"] if c["field"] == "material_mention"]
        self.assertEqual([c["value"] for c in mentions], ["Faux leather", "cotton", "wool"])
        self.assertTrue(mentions[-1]["negation_flag"])
        self.assertNotIn("material", {c["field"] for c in result["candidates"]})

    def test_conflicts_preserve_both_claims_instead_of_picking_one(self):
        result = extract({"details": {"Color": "Blue"}, "features": ["Color: Red; Size: Large"]})
        self.assertEqual(result["conflicting_fields"], ["color"])
        colors = [c for c in result["candidates"] if c["field"] == "color"]
        self.assertEqual({c["value"] for c in colors}, {"Blue", "Red"})
        self.assertTrue(all(c["conflict_flag"] for c in colors))

    def test_unknowns_manufacturer_seller_and_script_text_do_not_fill_fields(self):
        row = {"details": {"Color": "unknown", "Size": "N/A", "Manufacturer": "Factory"},
               "store": "Seller", "features": ["<script>100% cotton</script><style>wool</style>"]}
        self.assertEqual(extract(row)["candidates"], [])

    def test_input_is_preserved_and_string_description_is_supported(self):
        row = {"details": json.dumps({"Size": "Large"}), "description": "Hand wash; slim fit."}
        original = copy.deepcopy(row)
        result = extract(row)
        self.assertEqual(row, original)
        self.assertIn("care_instruction", {c["field"] for c in result["candidates"]})
        self.assertTrue(all(c["source"].get("index") is None for c in result["candidates"]))

    def test_negative_care_directives_keep_the_prohibition(self):
        candidates = extract({"features": ["Do not machine wash; never tumble dry."]})["candidates"]
        care = [c["value"] for c in candidates if c["field"] == "care_instruction"]
        self.assertEqual(care, ["Do not machine wash", "never tumble dry"])

    def test_component_scope_does_not_cross_clauses(self):
        candidates = extract({"features": ["Leather upper, cotton lining. Wool."]})["candidates"]
        materials = [c for c in candidates if c["field"] == "material_mention"]
        self.assertEqual([c["applies_to"] for c in materials], ["upper", "lining", "unknown"])

    def test_variant_conditions_are_separate_and_do_not_leak_across_sentences(self):
        text = "Solid colors: 100% Cotton; Heather Grey: 90% Cotton, 10% Polyester. Wool lining."
        candidates = extract({"features": [text]})["candidates"]
        compositions = [c for c in candidates if c["field"] == "composition_mention"]
        self.assertEqual([c["variant_condition"] for c in compositions], ["Solid colors", "Heather Grey", "Heather Grey"])
        wool = next(c for c in candidates if c["value"] == "Wool")
        self.assertEqual(wool["variant_condition"], "unknown")

    def test_measurement_role_and_explicit_size_are_retained(self):
        text = "Size L: chest 96 cm; Size M: waist 80 cm"
        candidates = [c for c in extract({"description": [text]})["candidates"] if c["field"] == "measurement_mention"]
        self.assertEqual([(c["size_label"], c["measurement_role"]) for c in candidates], [("L", "chest"), ("M", "waist")])

    def test_compound_labels_and_structured_component_keys_are_preserved(self):
        result = extract({"details": {"Upper Material": "Leather", "Lining Material": "Cotton"},
                          "features": ["About Cup Style: With padding bra", "Sleeve Style: Batwing"]})
        self.assertEqual({c["field"] for c in result["candidates"]}, {"material", "cup_style", "sleeve_style"})
        self.assertEqual(result["conflicting_fields"], [])
        self.assertEqual([c["applies_to"] for c in result["candidates"][:2]], ["upper", "lining"])

    def test_dense_size_chart_is_left_ambiguous(self):
        candidates = extract({"description": ["Size table: S---96cm---30cm M---100cm---40cm"]})["candidates"]
        measures = [c for c in candidates if c["field"] == "measurement_mention"]
        self.assertTrue(measures)
        self.assertTrue(all(c["size_label"] == "unknown" and "size_table_requires_review" in c["scope_warnings"] for c in measures))


@unittest.skipUnless(os.environ.get("RUN_DATABASE_TESTS") == "1", "Opt-in PostgreSQL integration tests")
class AttributeDatabaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.row, self.config = fixture(self.root)
        self.connection = psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5)
        self.addCleanup(self.connection.close)
        self.addCleanup(self.connection.rollback)
        self.connection.execute("BEGIN")
        ingest(self.connection, self.root / "manifest.json")
        self.product_id = self.config["source_id"] + ":AUTHORED1"

    def test_enrichment_is_idempotent_preserves_source_details_and_vectors(self):
        self.connection.execute("INSERT INTO product_embeddings(product_id,modality,model_name,embedding) "
                                "VALUES (%s,'text','authored-test','[1,0,0]')", (self.product_id,))
        before = self.connection.execute("SELECT raw_source,attributes FROM products WHERE id=%s", (self.product_id,)).fetchone()
        first = enrich(self.connection, self.root / "manifest.json")
        second = enrich(self.connection, self.root / "manifest.json")
        after = self.connection.execute("SELECT raw_source,attributes FROM products WHERE id=%s", (self.product_id,)).fetchone()
        self.assertEqual(first, second)
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1]["attributes"], after[1]["attributes"])
        self.assertEqual(after[1]["extraction"]["extractor_version"], "rules-v2")
        self.assertEqual(self.connection.execute("SELECT count(*) FROM product_embeddings WHERE product_id=%s", (self.product_id,)).fetchone()[0], 1)

    def test_changed_source_is_not_enriched_until_reimported(self):
        self.row["features"].append("Hand wash")
        (self.root / "records.jsonl").write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            enrich(self.connection, self.root / "manifest.json")
