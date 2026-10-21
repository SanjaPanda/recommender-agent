"""Caption safety and retrieval contracts without model or database services."""
import json
import unittest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from rag.describe_images import (describe, installed_digest, validate_vector, image_path,
                                 summarize_run_status, caption_matches_source, PROMPT_VERSION)
from rag.retrieval import retrieve_products
from agent.config import settings


class CaptionTests(unittest.TestCase):
    def test_caption_request_uses_image_and_rejects_truncation(self):
        response = Mock()
        content = json.dumps({"product_type": "shirt", "colours": ["red"], "pattern": None,
                              "visible_components": ["shirt"], "visible_object_count": 1,
                              "other_visual_details": None})
        response.read.return_value = json.dumps({"message": {"content": content}, "done_reason": "stop",
                                                 "eval_count": 7, "eval_duration": 1000}).encode()
        with patch("rag.describe_images.urllib.request.urlopen") as request:
            request.return_value.__enter__.return_value = response
            diagnostics = {}
            self.assertEqual(describe(b"image bytes", "gemma3:4b", 9,
                                      diagnostics=diagnostics), "Product type: shirt. Colours: red. Visible components: shirt. Visible product objects: 1.")
            self.assertEqual(diagnostics["done_reason"], "stop")
            self.assertEqual(diagnostics["response_chars"], len(content))
            self.assertEqual(diagnostics["eval_count"], 7)
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload["messages"][0]["images"], ["aW1hZ2UgYnl0ZXM="])
            self.assertEqual(payload["format"]["type"], "object")
            self.assertEqual(request.call_args.kwargs["timeout"], 9)
            response.read.return_value = json.dumps({"message": {"content": "A red"}, "done_reason": "length"}).encode()
            with self.assertRaisesRegex(ValueError, "truncated by the model"):
                describe(b"image bytes", "gemma3:4b", 9)

    def test_empty_caption_is_returned_and_oversized_output_is_rejected(self):
        response = Mock()
        with patch("rag.describe_images.urllib.request.urlopen") as request:
            request.return_value.__enter__.return_value = response
            response.read.return_value = json.dumps({"message": {"content": " "},
                                                     "done_reason": "stop"}).encode()
            diagnostics = {}
            self.assertEqual(describe(b"image bytes", "gemma3:4b", 9,
                                      diagnostics=diagnostics), "")
            self.assertEqual(diagnostics["done_reason"], "stop")
            self.assertEqual(diagnostics["response_chars"], 1)

            oversize = json.dumps({"product_type": "x" * 161, "colours": ["red"], "pattern": None,
                                   "visible_components": ["shirt"], "visible_object_count": None,
                                   "other_visual_details": None})
            response.read.return_value = json.dumps({"message": {"content": oversize},
                                                     "done_reason": "stop"}).encode()
            with self.assertRaises(ValidationError):
                describe(b"image bytes", "gemma3:4b", 9)

    def test_image_path_rejects_traversal(self):
        for value in ("../outside.jpg", "C:/outside.jpg", None):
            self.assertIsNone(image_path({"local_path": value}))

    def test_model_tag_and_vector_validation(self):
        self.assertEqual(installed_digest("nomic-embed-text", [{"name": "nomic-embed-text:latest", "digest": "abc"}]), "abc")
        with self.assertRaises(ValueError):
            installed_digest("gemma3:4b", [{"name": "gemma3:16k", "digest": "wrong"}])
        for vector in ([], [0, 0], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                validate_vector(vector)

    def test_run_status_reports_success_and_continued_item_errors(self):
        self.assertEqual(summarize_run_status([{"status": "ready"}]), "success")
        self.assertEqual(summarize_run_status([{"status": "empty_caption"}]), "completed_with_issues")
        self.assertEqual(summarize_run_status([{"status": "failed"}, {"status": "ready"}]),
                         "completed_with_errors")

    def test_saved_empty_caption_matches_source_and_can_be_skipped(self):
        provenance = {"model_digest": "model", "image_sha256": "image",
                      "prompt_version": PROMPT_VERSION}
        self.assertTrue(caption_matches_source("", provenance, "model", "image"))
        self.assertFalse(caption_matches_source(None, provenance, "model", "image"))
        self.assertFalse(caption_matches_source("", provenance, "new-model", "image"))


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.row = ("result", "Red shirt", "Catalogue description", "Acme",
                    "Apparel & Accessories > Clothing", {}, False, .9)
        self.conn = Mock()
        self.connect = patch("rag.retrieval.psycopg.connect")
        self.connect.start().return_value.__enter__.return_value = self.conn
        self.addCleanup(self.connect.stop)
        self.register = patch("rag.retrieval.register_vector")
        self.register.start()
        self.addCleanup(self.register.stop)

    def test_selected_clip_example_uses_apparel_vectors_and_filters(self):
        self.conn.execute.return_value.fetchall.side_effect = [[("seed", [1., 0.])], [self.row]]
        products = retrieve_products(selected_product_ids=["seed"],
            filters={"brand": "Acme", "is_secondhand": False})
        self.assertEqual(products[0]["description"], "Catalogue description")
        self.assertEqual(products[0]["image_similarity"], .9)
        self.assertIn("image", self.conn.execute.call_args_list[1].args[1])
        self.assertIn("lower(p.brand)", self.conn.execute.call_args_list[1].args[0])
        self.assertIn(False, self.conn.execute.call_args_list[1].args[1])
        self.assertIn(["seed"], self.conn.execute.call_args_list[0].args[1])

    def test_query_fuses_text_caption_and_clip_rankings(self):
        self.conn.execute.return_value.fetchall.side_effect = [[self.row], [self.row], [self.row]]
        with patch("rag.retrieval.embed_texts", return_value=[[1., 0.]]) as model, \
             patch("rag.retrieval.clip_text_embedding", return_value=[1., 0.]) as clip:
            products = retrieve_products(query_context="red shirt")
        model.assert_called_once()
        clip.assert_called_once_with("red shirt")
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0]["text_similarity"], .9)
        self.assertEqual(products[0]["image_description_similarity"], .9)
        self.assertEqual(products[0]["image_similarity"], .9)
        self.assertEqual(self.conn.execute.call_count, 3)
        for call in self.conn.execute.call_args_list:
            self.assertIn("ground_truth_category", call.args[0])

    def test_no_query_or_example_lists_apparel_without_model_calls(self):
        self.conn.execute.return_value.fetchall.return_value = [(*self.row[:6], False)]
        with patch("rag.retrieval.embed_texts") as model, patch("rag.retrieval.clip_text_embedding") as clip:
            products = retrieve_products(filters={"is_secondhand": False})
        model.assert_not_called()
        clip.assert_not_called()
        self.assertFalse(products[0]["is_secondhand"])

    def test_missing_seed_fails_explicitly(self):
        self.conn.execute.return_value.fetchall.return_value = []
        with self.assertRaisesRegex(ValueError, "missing"):
            retrieve_products(selected_product_ids=["missing"])


if __name__ == "__main__":
    unittest.main()
