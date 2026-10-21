"""Evidence checks for relevance gates and fallback alternative ranking."""
import unittest
from unittest.mock import patch

from agent.models import Filters, Plan
from agent.relevance import is_relevant_product
from agent.workflow import _retrieve


class RelevanceTests(unittest.TestCase):
    """Keep explicit attributes and product terms from being lost in vectors."""

    def test_color_and_product_terms_reject_wrong_color_or_wrong_type(self):
        yellow_tie = {
            "title": "Yellow Bow Tie Shining Silk",
            "description": "Men's pre-tied bowtie handkerchief set",
            "text_similarity": 0.67,
            "image_similarity": 0.262,
        }
        pink_headpiece = {
            "title": "Pink Bridal Flower Headpiece",
            "description": "A pink wedding hair comb",
            "text_similarity": 0.63,
            "image_similarity": 0.20,
        }
        self.assertFalse(is_relevant_product(yellow_tie, "pink bow tie"))
        self.assertFalse(is_relevant_product(pink_headpiece, "pink bow tie"))

    def test_exact_catalog_text_can_pass_even_when_similarity_is_below_cutoff(self):
        pink_tie = {
            "title": "Pink Bow Tie",
            "description": "Pre-tied accessory",
            "text_similarity": 0.55,
            "image_similarity": 0.20,
        }
        blue_clothes = {
            "title": "Cupro Lounge Shorts - midnight blue",
            "description": "Lightweight lounge wear",
            "text_similarity": 0.55,
            "image_similarity": 0.24,
        }
        self.assertTrue(is_relevant_product(pink_tie, "pink bow tie"))
        self.assertTrue(is_relevant_product(blue_clothes, "blue clothes"))

    def test_semantic_fallback_can_keep_a_close_wrong_color_alternative(self):
        yellow_tie = {
            "title": "Yellow Bow Tie Shining Silk",
            "description": "Men's pre-tied bowtie handkerchief set",
            "text_similarity": 0.67,
            "image_similarity": 0.262,
        }
        self.assertTrue(is_relevant_product(
            yellow_tie, "pink bow tie", ignore_query_attributes=True,
        ))

    def test_excluded_colors_stay_excluded_and_examples_use_separate_score(self):
        blue_shirt = {
            "title": "Blue Cotton Shirt", "description": "",
            "text_similarity": 0.8, "image_similarity": 0.8,
            "example_similarity": 0.8,
        }
        example_match = {
            "title": "Garment", "description": "",
            "text_similarity": None, "image_similarity": 0.1,
            "example_similarity": 0.31,
        }
        weak_example = {**example_match, "example_similarity": 0.1}
        self.assertFalse(is_relevant_product(blue_shirt, "not blue clothes"))
        self.assertTrue(is_relevant_product(example_match, "", has_visual_examples=True))
        self.assertFalse(is_relevant_product(weak_example, "", has_visual_examples=True))

    def test_retrieval_falls_back_only_inside_selected_subcategory(self):
        subcategory = "Apparel & Accessories > Clothing Accessories"
        candidates = [
            {
                "id": "yellow-tie", "title": "Yellow Bow Tie", "description": "Silk bow tie",
                "text_similarity": 0.67, "image_similarity": 0.262,
            },
            {
                "id": "pink-headpiece", "title": "Pink Bridal Headpiece", "description": "Hair comb",
                "text_similarity": 0.63, "image_similarity": 0.20,
            },
        ]
        plan = Plan(
            intent="recommend", query="pink bow tie",
            filters=Filters(category_prefixes=[subcategory]),
        )
        with patch("agent.workflow.retrieve_products", return_value=candidates) as retrieve:
            result = _retrieve({"plan": plan.model_dump(), "selected_product_ids": []})

        self.assertEqual(result["result_kind"], "fallback")
        self.assertEqual({product["id"] for product in result["candidates"]},
                         {"yellow-tie", "pink-headpiece"})
        self.assertEqual(retrieve.call_args.kwargs["filters"]["category_prefixes"], [subcategory])

    def test_strong_exact_match_excludes_weaker_candidates(self):
        plan = Plan(intent="recommend", query="pink bow tie")
        candidates = [
            {
                "id": "pink-tie", "title": "Pink Bow Tie", "description": "Pre-tied silk",
                "text_similarity": 0.55, "image_similarity": 0.20,
            },
            {
                "id": "yellow-tie", "title": "Yellow Bow Tie", "description": "Silk accessory",
                "text_similarity": 0.67, "image_similarity": 0.262,
            },
        ]
        with patch("agent.workflow.retrieve_products", return_value=candidates):
            result = _retrieve({"plan": plan.model_dump(), "selected_product_ids": []})

        self.assertEqual(result["result_kind"], "matches")
        self.assertEqual([product["id"] for product in result["candidates"]], ["pink-tie"])


if __name__ == "__main__":
    unittest.main()
