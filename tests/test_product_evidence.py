"""Card evidence must distinguish mismatch, missing data and similarity."""
import unittest
from unittest.mock import MagicMock, patch

from agent.product_evidence import listing_checks, source_listing


class ProductEvidenceTests(unittest.TestCase):
    def test_bow_tie_check_rejects_regular_tie(self):
        for title, expected in (("Blue Tie", "mismatch"), ("Blue Bow Tie", "match")):
            checks = listing_checks(title, {}, "blue bow tie", {"title_product": "tie"})
            self.assertEqual({c["label"]: c["status"] for c in checks},
                             {"Blue": "match", "Bow tie": expected})

    def test_blue_request_black_dress_displays_color_cross_type_tick(self):
        checks = listing_checks("Black Dress", {}, "blue dress", {})
        self.assertEqual({c["label"]: c["status"] for c in checks}, {"Blue": "mismatch", "Dress": "match"})

    def test_missing_conflicting_and_multivariant_color_are_unknown(self):
        for title, details in (("Dress", {}), ("Black Dress", {"Color": "Blue"}), ("Blue and Black Dress", {})):
            with self.subTest(title=title):
                checks = listing_checks(title, details, "blue dress", {})
                self.assertEqual(checks[0]["status"], "unknown")

    def test_positive_negative_and_spelling_alias_colors(self):
        self.assertEqual(listing_checks("Grey Dress", {}, "gray dress", {})[0]["status"], "match")
        self.assertEqual(listing_checks("Blue Dress", {}, "not blue", {})[0]["status"], "mismatch")
        self.assertEqual(listing_checks("Black Dress", {}, "not blue", {})[0]["status"], "match")

    def test_missing_ratings_are_not_fabricated_and_zero_is_preserved(self):
        self.assertIsNone(source_listing({}, None, [], {})["average_rating"])
        self.assertIsNone(source_listing({"average_rating": float("nan")}, None, [], {})["average_rating"])
        listing = source_listing({"average_rating": 0, "rating_number": 0}, "Description", ["Cotton"], {"Color": "Blue"})
        self.assertEqual(listing["average_rating"], 0)
        self.assertEqual(listing["rating_number"], 0)
        self.assertEqual(listing["features"], ["Cotton"])

    def test_api_keeps_scores_and_exposes_details_in_one_batch(self):
        from agent.api import _serialize_products
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [
            ("dress", "Description", ["Feature"], {"average_rating": 4.5, "rating_number": 12}, {"Color": "Black"})]
        with patch("agent.api.psycopg.connect") as connect, patch("agent.api._find_image_path", return_value=None):
            connect.return_value.__enter__.return_value = connection
            result = _serialize_products([{"id": "dress", "title": "Black Dress", "text_similarity": .7}], "blue dress")
        self.assertEqual(connection.execute.call_count, 1)
        self.assertEqual(result[0]["text_similarity"], .7)
        self.assertEqual(result[0]["listing"]["average_rating"], 4.5)
        self.assertEqual(result[0]["match_checks"][0]["status"], "mismatch")
