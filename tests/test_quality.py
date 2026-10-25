"""Authored tests for evidence-based curation without model inference."""
import unittest

from rag.catalog import normalize_metadata
from rag.quality import assess, clean_text


def product():
    """A source-authored product description with meaningful specification text."""
    return {"parent_asin": "AUTHOREDQ", "title": "Cotton shirt", "main_category": "AMAZON FASHION",
            "categories": ["Clothing, Shoes & Jewelry", "Clothing"],
            "description": [
                "This cotton shirt has long sleeves and a button closure with two front pockets. "
                "The relaxed fit provides room around the chest and waist for comfortable movement. "
                "Machine wash with similar colors and line dry after cleaning. "
                "The stitched hem and reinforced seams provide durable construction for everyday use. "
                "Adjustable cuffs let the wearer change the sleeve opening."
            ], "features": ["<li>Cotton fabric</li>", "Machine wash"],
            "details": {"Material": "Cotton", "Department": "Adult", "Date First Available": "2020"}}


class QualityTests(unittest.TestCase):
    def test_admin_metadata_does_not_qualify_as_useful_attributes(self):
        row = product()
        row["details"] = {"Department": "Adult", "Manufacturer": "Factory", "Package Dimensions": "1 x 2"}
        result = assess(row)
        self.assertIn("no_useful_structured_attributes", result["failures"])
        self.assertTrue(result["feature_evidence"])
        self.assertEqual(result["useful_detail_keys"], [])

    def test_features_never_substitute_for_product_description(self):
        row = product()
        row["features"] = row["description"]
        row["description"] = []
        self.assertIn("insufficient_product_description", assess(row)["failures"])

    def test_brand_history_does_not_fill_description_word_gate(self):
        row = product()
        row["description"] = ["Cotton shirt with button closure. About Us: " + "Our company creates wonderful collections. " * 30]
        result = assess(row)
        self.assertTrue(result["boilerplate_flagged"])
        self.assertLess(result["nonflagged_description_words"], 50)
        self.assertIn("insufficient_product_description", result["failures"])

    def test_evidence_v2_accepts_product_text_with_structured_specs(self):
        row = product()
        result = assess(row)
        self.assertTrue(result["review_ready"])
        self.assertEqual(result["useful_detail_keys"], ["Material"])
        normalized = normalize_metadata(row, quality_policy="evidence-v2")
        self.assertEqual(normalized["description"], row["description"][0])

    def test_evidence_v2_rejects_generic_word_count_while_legacy_is_preserved(self):
        row = product()
        row["description"] = ["A wonderful collection for every day of the year. " * 10]
        normalize_metadata(row)
        with self.assertRaisesRegex(ValueError, "weak_description_signals"):
            normalize_metadata(row, quality_policy="evidence-v2")

    def test_feature_evidence_is_not_promoted_to_material_composition(self):
        row = product()
        row["features"] = ["Leather lining with cotton outer fabric"]
        result = assess(row)
        match = result["feature_evidence"][0]
        self.assertEqual(match["signal"], "material_mention")
        self.assertEqual(match["source"], "features[0]")
        self.assertIn(match["matched_text"], row["features"][0])
        self.assertNotIn("material", result)

    def test_unknown_details_and_markup_do_not_create_specifications(self):
        row = product()
        row["details"] = {"Material": "Unknown", "Color": "N/A", "Size": " "}
        self.assertEqual(assess(row)["useful_detail_keys"], [])
        self.assertEqual(clean_text("<p>Cotton &amp; wool</p><script>leather</script>"), "Cotton & wool")
