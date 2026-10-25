"""Product intent eligibility rules and wrong-type retrieval regressions."""
import unittest
from unittest.mock import MagicMock, patch

from agent.relevance import is_relevant_product
from rag.product_types import product_type, requested_product_type
from rag.retrieval import retrieve_products


class ProductTypeTests(unittest.TestCase):
    def setUp(self):
        for name in ('text_model_version', 'clip_model_version'):
            version = patch('rag.retrieval.' + name, return_value='test-revision')
            version.start()
            self.addCleanup(version.stop)

    def test_bow_tie_is_distinct_from_necktie(self):
        for title in ("blue bow tie", "blue bow-tie", "blue bowties",
                      "Boys Wedding Bow Tie 12 Pack Children Bowties Kids Tuxedo Solid Ties",
                      "LED Light Up Flashing Sequin Bow Ties Tie"):
            self.assertEqual(product_type(title), "bowtie", title)
        for title in ("blue tie", "blue necktie", "silk neckties"):
            self.assertEqual(product_type(title), "tie", title)
        self.assertEqual(product_type("Tuxedo Shirt with Bow Tie"), "shirt")
        self.assertIsNone(product_type("Animal Bowtie Tail Costume Kits"))
        self.assertIsNone(product_type("bow tie and necktie"))
        self.assertIsNone(product_type("not a bow tie"))
        self.assertEqual(requested_product_type("blue bow tie", "tie"), "bowtie")
        self.assertEqual(requested_product_type("blue bow tie", "shirt"), "shirt")

    def test_necktie_cannot_satisfy_bow_tie_request(self):
        product = {"title": "Blue Necktie", "description": "Also available: bow ties",
                   "text_similarity": .99, "image_similarity": .99}
        self.assertFalse(is_relevant_product(product, "blue bow tie", required_product_type="tie"))
        self.assertTrue(is_relevant_product({**product, "title": "Blue Bow Tie"}, "blue bow tie"))

    def test_bow_tie_filters_before_limit_even_with_parent_filter(self):
        rows = [("tie", "Blue Tie", "bow tie", None, "Apparel & Accessories", None, None, .99),
                ("bow", "Blue Bow Tie", "", None, "Apparel & Accessories", None, None, .6)]
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = rows
        with patch("rag.retrieval.psycopg.connect") as connect, patch("rag.retrieval.register_vector"), \
             patch("rag.retrieval._search_bm25", return_value=[]), \
             patch("rag.retrieval.embed_texts", return_value=[[1., 0.]]), \
             patch("rag.retrieval.clip_text_embedding", return_value=[1., 0.]):
            connect.return_value.__enter__.return_value = connection
            result = retrieve_products(query_context="blue bow tie", filters={"title_product": "tie"}, limit=1)
        self.assertEqual([r["id"] for r in result], ["bow"])
        for call in connection.execute.call_args_list:
            self.assertIn(r"\m(?:bow[- ]?ties?)\M", call.args[1])

    def test_compounds_plurals_and_aliases(self):
        for text, expected in (("cotton shirts", "shirt"), ("button down blouse", "shirt"),
                               ("Dress shoes", "shoe"), ("dress shirt", "shirt"),
                               ("T-Shirt Dress", "dress"), ("Faux Leather Jacket Coat", "jacket"),
                               ("warm wool hat", "hat")):
            with self.subTest(text=text):
                self.assertEqual(product_type(text), expected)

    def test_ambiguous_unknown_negated_and_compatibility_clauses(self):
        for text in ("shirts and jeans", "something lightweight", "not a shirt", "no sweaters",
                     "Faux Fur Collar Coat Accessory"):
            self.assertIsNone(product_type(text))
        self.assertEqual(product_type("gloves for jacket wearers"), "glove")
        self.assertEqual(product_type("shirt with a sweater print"), "shirt")

    def test_description_brand_story_cannot_override_explicit_query_type(self):
        sweater = {"title": "Fisherman Sweater", "description": "Our line includes cotton shirts.",
                   "text_similarity": .99, "image_similarity": .99}
        self.assertFalse(is_relevant_product(sweater, "cotton shirt"))
        shirt = {**sweater, "title": "Cotton Shirt"}
        self.assertTrue(is_relevant_product(shirt, "cotton shirt"))

    def test_plain_query_filters_wrong_types_before_return_limit(self):
        rows = [("sweater", "Cotton Sweater", "Cotton shirt brand story", None,
                 "Apparel & Accessories > Clothing", None, None, .99),
                ("shirt", "Cotton Shirt", "Cotton", None,
                 "Apparel & Accessories > Clothing", None, None, .6)]
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = rows
        with patch("rag.retrieval.psycopg.connect") as connect, patch("rag.retrieval.register_vector"), \
             patch("rag.retrieval._search_bm25", return_value=[]), \
             patch("rag.retrieval.embed_texts", return_value=[[1., 0.]]), \
             patch("rag.retrieval.clip_text_embedding", return_value=[1., 0.]):
            connect.return_value.__enter__.return_value = connection
            result = retrieve_products(query_context="cotton shirt", limit=1)
        self.assertEqual([r["id"] for r in result], ["shirt"])
        self.assertTrue(all("COALESCE(p.title, '') ~* %s" in c.args[0] for c in connection.execute.call_args_list))

    def test_image_only_jacket_search_excludes_higher_scoring_jeans(self):
        connection = MagicMock()
        def execute(sql, params):
            if "SELECT e.product_id,e.embedding::real[]" in sql:
                return MagicMock(fetchall=lambda: [("seed", [1., 0.], "Leather Jacket", "test-revision")])
            self.assertIn("COALESCE(p.title, '') ~* %s", sql)
            return MagicMock(fetchall=lambda: [
                ("jeans", "Black Jeans", "", None, "Apparel & Accessories > Clothing", None, None, .95),
                ("jacket", "Winter Jacket", "", None, "Apparel & Accessories > Clothing", None, None, .7)])
        connection.execute.side_effect = execute
        with patch("rag.retrieval.psycopg.connect") as connect, patch("rag.retrieval.register_vector"), \
             patch("rag.retrieval.embed_texts") as embed:
            connect.return_value.__enter__.return_value = connection
            result = retrieve_products(selected_product_ids=["seed"])
            embed.assert_not_called()
        self.assertEqual([r["id"] for r in result], ["jacket"])

    def test_explicit_type_filter_overrides_image_seed_type(self):
        connection = MagicMock()
        def execute(sql, params):
            if "SELECT e.product_id,e.embedding::real[]" in sql:
                return MagicMock(fetchall=lambda: [("seed", [1., 0.], "Leather Jacket", "test-revision")])
            return MagicMock(fetchall=lambda: [
                ("shirt", "Cotton Shirt", "", None, "Apparel & Accessories > Clothing", None, None, .7)])
        connection.execute.side_effect = execute
        with patch("rag.retrieval.psycopg.connect") as connect, patch("rag.retrieval.register_vector"):
            connect.return_value.__enter__.return_value = connection
            result = retrieve_products(selected_product_ids=["seed"], filters={"title_product": "shirt"})
        self.assertEqual([r["id"] for r in result], ["shirt"])
