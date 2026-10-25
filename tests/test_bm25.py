"""Lexical ranking and cross-signal fusion behavior on authored products."""
import math
import unittest
from unittest.mock import MagicMock, patch

from rag.bm25 import rank, tokens
import rag.bm25 as bm25
from rag.retrieval import _search_bm25, retrieve_products


def product(identifier, score):
    return (identifier, "Cotton shirt", "shirt", None,
            "Apparel & Accessories > Clothing", None, False, score)


class BM25Tests(unittest.TestCase):
    def setUp(self):
        for name in ('text_model_version', 'clip_model_version'):
            version = patch('rag.retrieval.' + name, return_value='test-revision')
            version.start()
            self.addCleanup(version.stop)

    def test_cached_stats_are_scoped_to_filtered_documents_and_content(self):
        bm25._prepare_corpus.cache_clear()
        bm25._document_counts.cache_clear()
        full = ['cotton shirt', 'cotton shirt', 'linen dress']
        original = rank('cotton', full)
        with patch('rag.bm25.tokens', wraps=tokens) as tokenize:
            self.assertEqual(rank('cotton', full), original)
            self.assertEqual(tokenize.call_count, 1)  # query only
        self.assertAlmostEqual(rank('cotton', [full[0]])[0][1], math.log(1 + .5 / 1.5))
        self.assertNotEqual(rank('cotton', [full[0]])[0][1], original[0][1])
        self.assertEqual(rank('cotton', ['linen shirt', 'linen dress']), [])
        self.assertEqual([index for index, score in original], [0, 1])
        self.assertEqual(rank('cotton', list(reversed(full)))[0][0], 1)

    def test_cached_documents_and_statistics_are_immutable_and_bounded(self):
        corpus = bm25._prepare_corpus(('cotton shirt',))
        with self.assertRaises(TypeError):
            corpus.counts[0]['cotton'] = 100
        with self.assertRaises(TypeError):
            corpus.frequencies['cotton'] = 100
        for number in range(12):
            rank('cotton', ['cotton ' + str(number)])
        self.assertLessEqual(bm25._prepare_corpus.cache_info().currsize, 8)
        self.assertEqual(bm25._document_counts.cache_info().maxsize, 4096)

    def test_formula_repeated_query_and_zero_matches(self):
        result = rank("cotton cotton", ["cotton", "linen"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 0)
        self.assertAlmostEqual(result[0][1], math.log(2))
        self.assertEqual(rank("cotton", ["linen", ""]), [])
        self.assertEqual(rank("", ["cotton"]), [])
        self.assertEqual(rank("shirt", []), [])

    def test_length_normalization_and_html_unicode(self):
        self.assertEqual(tokens("<b>BLUE</b> café"), ["blue", "café"])
        self.assertEqual(rank("cotton", ["cotton " + "other " * 100, "cotton"])[0][0], 1)

    def test_features_are_searchable_and_sql_preserves_filters(self):
        connection = MagicMock()
        row = (*product("a", 0)[:7], ["Waterproof cotton fabric"])
        connection.execute.return_value.fetchall.return_value = [row]
        result = _search_bm25(connection, query="waterproof", category_clause=" AND p.category=%s",
                              category_params=["Apparel & Accessories"], excluded_ids=["seed"], limit=5)
        self.assertEqual(result[0][0], "a")
        self.assertGreater(result[0][7], 0)
        sql, params = connection.execute.call_args.args
        self.assertIn("p.id <> ALL(%s)", sql)
        self.assertEqual(params, ["Apparel & Accessories", ["seed"]])

    def test_feature_only_keyword_can_pass_gate_without_verifying_a_color(self):
        from agent.relevance import is_relevant_product
        item = {"title": "Shirt", "description": "A garment", "features": ["Waterproof fabric; blue variant"]}
        self.assertTrue(is_relevant_product(item, "waterproof"))
        self.assertFalse(is_relevant_product(item, "blue waterproof"))

    def test_consensus_wins_rrf_and_contributions_sum(self):
        connection = MagicMock()
        with patch("rag.retrieval.psycopg.connect") as connect, \
             patch("rag.retrieval.register_vector"), \
             patch("rag.retrieval.embed_texts", return_value=[[1., 0.]]), \
             patch("rag.retrieval.clip_text_embedding", return_value=[1., 0.]), \
             patch("rag.retrieval._search_bm25", return_value=[product("keyword-only", 20), product("consensus", 3)]), \
             patch("rag.retrieval._search_embedding", side_effect=[
                 [product("consensus", .8)], [product("consensus", .3)]
             ]):
            connect.return_value.__enter__.return_value = connection
            result = retrieve_products(query_context="cotton shirt")
        self.assertEqual(result[0]["id"], "consensus")
        self.assertEqual(result[0]["bm25_score"], 3)
        self.assertAlmostEqual(result[0]["rrf_score"], 1/3/62 + 2/3/61)
        self.assertAlmostEqual(result[0]["rrf_score"], sum(p["contribution"] for p in result[0]["rrf_contributions"]))
        self.assertAlmostEqual(sum(p["weight"] for p in result[0]["rrf_contributions"]), 1)
        self.assertEqual(result[0]["score"], result[0]["rrf_score"])
