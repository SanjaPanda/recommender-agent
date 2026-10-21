"""Regression tests for the active apparel text-and-CLIP retrieval ranker."""
import unittest
from unittest.mock import MagicMock, patch

from rag.retrieval import retrieve_products


class RetrievalTests(unittest.TestCase):
    """Ensure query retrieval uses only product-text and CLIP image vectors."""

    def test_query_ranker_searches_text_and_clip_without_caption_modality(self):
        connection = MagicMock()

        def execute(sql, params):
            """Return one stable apparel product for either vector search."""
            modality = params[2]
            self.assertIn(modality, {"text", "image"})
            return MagicMock(fetchall=lambda: [(
                "shirt-1", "Blue Cotton Shirt", "Cotton shirt", "Example",
                "Apparel & Accessories > Clothing", {"local_path": "shirt.jpg"},
                False, 0.9,
            )])

        connection.execute.side_effect = execute
        with patch("rag.retrieval.psycopg.connect") as connect, \
             patch("rag.retrieval.register_vector"), \
             patch("rag.retrieval.embed_texts", return_value=[[1.0, 0.0]]), \
             patch("rag.retrieval.clip_text_embedding", return_value=[0.0, 1.0]):
            connect.return_value.__enter__.return_value = connection
            results = retrieve_products(query_context="blue cotton shirt")

        searched_modalities = [
            call.args[1][2]
            for call in connection.execute.call_args_list
            if "e.modality=%s" in call.args[0]
        ]
        self.assertCountEqual(searched_modalities, ["text", "image"])
        self.assertEqual([product["id"] for product in results], ["shirt-1"])
        self.assertEqual(results[0]["text_similarity"], 0.9)
        self.assertEqual(results[0]["image_similarity"], 0.9)
        self.assertNotIn("image_description_similarity", results[0])

    def test_selected_example_score_does_not_replace_query_image_score(self):
        connection = MagicMock()
        image_searches = 0

        def execute(sql, params):
            """Return distinct scores for the example and query-image rankings."""
            nonlocal image_searches
            if "SELECT e.product_id,e.embedding::real[]" in sql:
                return MagicMock(fetchall=lambda: [("example-1", [1.0, 0.0])])
            modality = params[2]
            similarity = 0.7
            if modality == "image":
                image_searches += 1
                similarity = 0.93 if image_searches == 1 else 0.25
            return MagicMock(fetchall=lambda: [(
                "shirt-1", "Blue Cotton Shirt", "Cotton shirt", "Example",
                "Apparel & Accessories > Clothing", {"local_path": "shirt.jpg"},
                False, similarity,
            )])

        connection.execute.side_effect = execute
        with patch("rag.retrieval.psycopg.connect") as connect, \
             patch("rag.retrieval.register_vector"), \
             patch("rag.retrieval.embed_texts", return_value=[[0.0, 1.0]]), \
             patch("rag.retrieval.clip_text_embedding", return_value=[0.0, 1.0]):
            connect.return_value.__enter__.return_value = connection
            results = retrieve_products(
                query_context="blue cotton shirt", selected_product_ids=["example-1"])

        self.assertEqual(image_searches, 2)
        self.assertEqual(results[0]["image_similarity"], 0.25)
        self.assertEqual(results[0]["example_similarity"], 0.93)


if __name__ == "__main__":
    unittest.main()
