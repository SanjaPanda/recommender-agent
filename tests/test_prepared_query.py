"""Protect ranking and filter semantics when primary/fallback reuse encodings."""
import unittest
from unittest.mock import MagicMock, patch

from agent.models import Plan, SearchIntent
from agent.workflow import _retrieve
from agent.tools import retrieval


class PreparedQueryTests(unittest.TestCase):
    def setUp(self):
        self.text = self.enterContext(patch.object(retrieval, 'embed_texts', return_value=[[1., 0.]]))
        self.clip = self.enterContext(patch.object(retrieval, 'clip_text_embedding', return_value=[0., 1.]))
        self.versions = self.enterContext(patch.object(retrieval, 'text_model_version', return_value='text-revision'))
        self.enterContext(patch.object(retrieval, 'clip_model_version', return_value='image-revision'))
        connection = self.enterContext(patch.object(retrieval.psycopg, 'connect'))
        connection.return_value.__enter__.return_value = MagicMock()
        self.enterContext(patch.object(retrieval, 'register_vector'))

    def test_fallback_encodes_once_and_retains_hard_scope_and_cart_exclusion(self):
        lexical = self.enterContext(patch.object(retrieval, '_search_bm25', return_value=[]))
        vectors = self.enterContext(patch.object(retrieval, '_search_embedding', return_value=[]))
        plan = Plan(intent='recommend', query='blue dress', filters={
            'title_color': 'blue', 'brand': 'Brand',
            'category_prefixes': ['Apparel & Accessories > Clothing']})
        result = _retrieve({'plan': plan.model_dump(), 'cart': [{'id': 'cart'}],
            'intent_query': 'blue dress', 'search_intent': SearchIntent(product_phrase='dress').model_dump()})
        self.assertEqual(result['result_kind'], 'none')
        self.text.assert_called_once_with('search_query: blue dress')
        self.clip.assert_called_once_with('blue dress')
        self.assertEqual(self.versions.call_count, 2)  # Check before and after encoding.
        self.assertEqual(lexical.call_count, 2)
        self.assertEqual(vectors.call_count, 4)
        for call in lexical.call_args_list:
            self.assertIn('Apparel & Accessories', call.kwargs['category_params'])
            self.assertIn('Apparel & Accessories > Clothing', call.kwargs['category_params'])
            self.assertIn('Brand', call.kwargs['category_params'])
            self.assertIn(['cart'], call.kwargs['category_params'])
            self.assertIn('regexp_split_to_array', call.kwargs['category_clause'])
        self.assertIn(r'\mblue\M', lexical.call_args_list[0].kwargs['category_params'])
        self.assertNotIn(r'\mblue\M', lexical.call_args_list[1].kwargs['category_params'])

    def test_prepared_and_standalone_queries_produce_identical_ranking_details(self):
        row = ('dress', 'Blue Dress', 'Cotton dress', 'Brand',
               'Apparel & Accessories > Clothing', {}, .8)
        self.enterContext(patch.object(retrieval, '_search_bm25', return_value=[row]))
        self.enterContext(patch.object(retrieval, '_search_embedding', return_value=[row]))
        direct = retrieval.retrieve_products(query_context='dress')
        prepared = retrieval.prepare_query('dress')
        self.text.reset_mock()
        self.clip.reset_mock()
        reused = retrieval.retrieve_products(query_context='dress', prepared_query=prepared)
        self.assertEqual(reused, direct)
        self.assertAlmostEqual(reused[0]['rrf_score'], sum(
            contribution['contribution'] for contribution in reused[0]['rrf_contributions']))
        self.text.assert_not_called()
        self.clip.assert_not_called()

    def test_wrong_query_cannot_reuse_vectors(self):
        prepared = retrieval.prepare_query('dress')
        with self.assertRaisesRegex(ValueError, 'different query'):
            retrieval.retrieve_products(query_context='shirt', prepared_query=prepared)

    def test_missing_clip_stays_missing_for_the_whole_turn(self):
        self.clip.side_effect = RuntimeError('No cached CLIP model')
        prepared = retrieval.prepare_query('dress')
        self.assertIsNone(prepared.image)
        self.enterContext(patch.object(retrieval, '_search_bm25', return_value=[]))
        vectors = self.enterContext(patch.object(retrieval, '_search_embedding', return_value=[]))
        retrieval.retrieve_products(query_context='dress', prepared_query=prepared)
        self.assertEqual([call.kwargs['modality'] for call in vectors.call_args_list], ['text'])
        self.clip.assert_called_once()
