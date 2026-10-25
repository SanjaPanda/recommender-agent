"""Exercise API-to-persisted-state paths with deterministic model/database doubles."""
from copy import deepcopy
import unittest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from agent import api
from agent.models import RouterDecision, SearchAttribute, SearchIntent


class TurnPathTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = {}
        self.connection = MagicMock()

        def execute(sql, params=None):
            if 'pg_try_advisory_lock' in sql:
                return MagicMock(fetchone=lambda: (True,))
            if 'SELECT revision' in sql:
                return MagicMock(fetchone=lambda: (1,))
            if 'SELECT state' in sql:
                value = self.snapshots.get(params[0])
                return MagicMock(fetchone=lambda: (deepcopy(value),) if value is not None else None)
            if 'INSERT INTO agent_sessions' in sql:
                self.snapshots.setdefault(params[0], {})
            if 'UPDATE agent_sessions' in sql:
                self.assertEqual(self.snapshots[params[1]], params[2].obj)
                self.snapshots[params[1]] = deepcopy(params[0].obj)
            return MagicMock(rowcount=1)

        self.connection.execute.side_effect = execute
        for target in ('agent.sessions.psycopg.connect', 'agent.api.psycopg.connect'):
            mock = self.enterContext(patch(target))
            mock.return_value.__enter__.return_value = self.connection
        self.enterContext(patch('agent.sessions.sanitize_apparel_products',
            side_effect=lambda connection, state, **kwargs: deepcopy(state)))
        self.enterContext(patch('agent.api.sanitize_apparel_products',
            side_effect=lambda connection, state: deepcopy(state)))
        self.enterContext(patch('agent.api._serialize_products',
            side_effect=lambda products, *args: deepcopy(products)))
        self.enterContext(patch('agent.workflow.list_categories', return_value=[]))
        self.enterContext(patch('agent.tools.retrieval.list_categories', return_value=[]))
        self.model = self.enterContext(patch('agent.workflow.complete'))
        self.enterContext(patch('agent.workflow.prepare_query', return_value=None))
        self.retrieval = self.enterContext(patch('agent.workflow.retrieve_products'))

    @staticmethod
    def decision(color):
        return RouterDecision(intent='recommend', query=f'{color} dress',
            search_intent=SearchIntent(product_phrase='dress', attributes=[
                SearchAttribute(name='color', value=color, evidence=color)]))

    def test_search_refine_cart_and_resume_preserve_the_public_response(self):
        self.model.side_effect = [self.decision('blue'), self.decision('black')]
        self.retrieval.side_effect = [
            [{'id': 'blue', 'title': 'Blue Dress'}],
            [{'id': 'black', 'title': 'Black Dress'}],
        ]
        first = api.discover(api.DiscoverRequest(thread_id='flow', query='blue dress'))
        self.assertEqual(first['products'][0]['id'], 'blue')
        self.assertEqual([step['node'] for step in first['trace']], ['router', 'retrieve', 'respond'])
        refined = api.recommend(api.RecommendRequest(thread_id='flow', refinement='make it black'))
        self.assertEqual(refined['query_context'], 'black dress')
        self.assertEqual(refined['products'][0]['id'], 'black')
        self.assertEqual(self.model.call_count, 2)
        added = api.cart(api.CartRequest(thread_id='flow', action='cart_add', product_ids=['black']))
        self.assertEqual(added['products'], [])
        self.assertEqual([p['id'] for p in added['cart']], ['black'])
        self.assertEqual(added['filters'], refined['filters'])
        resumed = api.session(api.SessionRequest(thread_id='flow'))
        self.assertEqual(resumed['query_context'], 'black dress')
        self.assertEqual(resumed['cart'], added['cart'])
        self.assertEqual(resumed['products'], [])
        self.assertEqual(resumed['trace'], [])
        self.assertEqual(self.model.call_count, 2)
        self.assertEqual(self.retrieval.call_count, 2)
        self.assertNotIn('trace', self.snapshots['flow'])
        self.assertNotIn('turn', self.snapshots['flow'])
        self.assertNotIn('candidates', self.snapshots['flow'])
        self.assertNotIn('route', self.snapshots['flow'])
        self.assertNotIn('query_context', self.snapshots['flow'])
        self.assertEqual(first['a2ui_surface_id'], refined['a2ui_surface_id'])
        self.assertIn('createSurface', first['a2ui_operations'][1])
        self.assertFalse(any('createSurface' in operation for operation in refined['a2ui_operations']))

    def test_reset_and_condition_request_keep_cart_without_model_or_search(self):
        self.snapshots['flow'] = {'plan': {'intent': 'recommend', 'query': 'dress'},
            'recommendations': [{'id': 'dress', 'title': 'Dress'}],
            'cart': [{'id': 'cart', 'title': 'Shirt'}], 'selected_product_ids': ['dress']}
        unsupported = api.discover(api.DiscoverRequest(thread_id='flow', query='only used dresses'))
        self.assertEqual(unsupported['status'], 'clarify')
        self.assertEqual(unsupported['products'][0]['id'], 'dress')
        reset = api.discover(api.DiscoverRequest(thread_id='flow', query='#buynew'))
        self.assertEqual(reset['status'], 'categories')
        self.assertEqual(reset['products'], [])
        self.assertEqual(reset['selected_product_ids'], [])
        self.assertEqual(reset['cart'][0]['id'], 'cart')
        self.model.assert_not_called()
        self.retrieval.assert_not_called()

    def test_search_and_cart_keep_their_distinct_invalid_action_statuses(self):
        calls = [
            (api.discover, api.DiscoverRequest(query='dress'), 503),
            (api.recommend, api.RecommendRequest(thread_id='flow', refinement='dress'), 503),
            (api.cart, api.CartRequest(thread_id='flow', action='cart_add', product_ids=['missing']), 400),
        ]
        with patch('agent.api.run_recommendation_turn', side_effect=ValueError('Invalid action')):
            for endpoint, request, status in calls:
                with self.subTest(endpoint=endpoint.__name__), self.assertRaises(HTTPException) as error:
                    endpoint(request)
                self.assertEqual(error.exception.status_code, status)
                self.assertEqual(error.exception.detail, 'Invalid action')
