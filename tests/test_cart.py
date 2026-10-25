"""Cart actions are independent of inference and mutually exclusive with recommendations."""
from copy import deepcopy
import unittest
from unittest.mock import MagicMock, patch

from agent.models import Filters, Plan, RouterDecision, RouterRequest, SearchIntent
from agent.workflow import build_recommendation_agent, sanitize_apparel_products, _retrieve
from rag.retrieval import retrieve_products


class CartTests(unittest.TestCase):
    def test_add_then_remove_preserves_search_without_inference(self):
        state = {'plan': Plan(intent='recommend', query='dress', filters=Filters(brand='Brand')).model_dump(),
                 'recommendations': [{'id': 'a', 'title': 'Dress'}, {'id': 'b', 'title': 'Other Dress'}],
                 'selected_product_ids': ['a'], 'messages': [{'role': 'user', 'content': 'dress'}]}
        before = deepcopy(state)
        graph = build_recommendation_agent()
        with patch('agent.workflow.complete') as model, patch('agent.workflow.retrieve_products') as retrieve, \
             patch('agent.workflow.list_categories') as categories:
            added = graph.invoke({**state, 'turn': {'action': 'cart_add', 'product_ids': ['a']}})
            self.assertEqual([p['id'] for p in added['cart']], ['a'])
            self.assertEqual([p['id'] for p in added['recommendations']], ['b'])
            self.assertEqual(added['selected_product_ids'], [])
            removed = graph.invoke({**added, 'turn': {'action': 'cart_remove', 'product_ids': ['a']}})
            self.assertEqual(removed['cart'], [])
            self.assertEqual(removed['recommendations'], added['recommendations'])
            self.assertEqual(removed['plan'], before['plan'])
            self.assertEqual(removed['messages'], before['messages'])
            model.assert_not_called()
            retrieve.assert_not_called()
            categories.assert_not_called()
        self.assertEqual(state, before)

    def test_invalid_batch_does_not_change_input_state(self):
        state = {'cart': [], 'recommendations': [{'id': 'a'}]}
        before = deepcopy(state)
        with self.assertRaises(ValueError):
            build_recommendation_agent().invoke({**state, 'turn': {
                'action': 'cart_add', 'product_ids': ['a', 'invented']}})
        self.assertEqual(state, before)

    def test_router_contracts_have_no_cart_fields_or_actions(self):
        from pydantic import ValidationError
        self.assertNotIn('cart', RouterRequest.model_json_schema()['properties'])
        self.assertNotIn('product_ids', RouterDecision.model_json_schema()['properties'])
        with self.assertRaises(ValidationError):
            RouterDecision(intent='cart_add')

    def test_old_cart_plan_and_overlapping_cards_are_normalized_without_mutating_snapshot(self):
        snapshot = {'plan': {'intent': 'cart_add', 'query': 'dress', 'product_ids': ['a'],
                             'filters': {'brand': 'Brand'}},
                    'recommendations': [{'id': 'a'}, {'id': 'b'}], 'cart': [{'id': 'a'}],
                    'selected_product_ids': ['a', 'b'], 'messages': [{'role': 'user', 'content': 'dress'}]}
        before = deepcopy(snapshot)
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [('a',), ('b',)]
        result = sanitize_apparel_products(connection, snapshot)
        plan = Plan.model_validate(result['plan'])
        self.assertEqual(plan.intent, 'recommend')
        self.assertEqual(plan.filters.brand, 'Brand')
        self.assertEqual([p['id'] for p in result['recommendations']], ['b'])
        self.assertEqual(result['selected_product_ids'], ['b'])
        self.assertEqual(result['cart'], [{'id': 'a'}])
        self.assertEqual(snapshot, before)

    def test_cart_exclusions_apply_to_primary_fallback_and_next_search_after_removal(self):
        plan = Plan(intent='recommend', query='blue dress', filters=Filters(title_color='blue'))
        state = {'plan': plan.model_dump(), 'cart': [{'id': 'a'}], 'intent_query': 'blue dress',
                 'search_intent': SearchIntent(product_phrase='dress').model_dump()}
        with patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            _retrieve(state)
            self.assertEqual(retrieve.call_count, 2)
            for call in retrieve.call_args_list:
                self.assertEqual(call.kwargs['excluded_product_ids'], ['a'])
            _retrieve({**state, 'cart': []})
            self.assertEqual(retrieve.call_args.kwargs['excluded_product_ids'], [])

    def test_browse_and_every_ranking_mode_exclude_cart_before_limits(self):
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = []
        with patch('rag.retrieval.psycopg.connect') as connect, patch('rag.retrieval.register_vector'), \
             patch('rag.retrieval.text_model_version', return_value='revision'), \
             patch('rag.retrieval.clip_model_version', return_value='revision'), \
             patch('rag.retrieval.embed_texts', return_value=[[1., 0.]]), \
             patch('rag.retrieval.clip_text_embedding', return_value=[1., 0.]), \
             patch('rag.retrieval._selected_example_vector', return_value=([1., 0.], 'revision', ['Dress'])), \
             patch('rag.retrieval._search_bm25', return_value=[]) as bm25, \
             patch('rag.retrieval._search_embedding', return_value=[]) as vectors:
            connect.return_value.__enter__.return_value = connection
            retrieve_products(excluded_product_ids=['cart'], limit=1)
            sql, params = connection.execute.call_args.args
            self.assertIn('p.id <> ALL(%s)', sql)
            self.assertIn(['cart'], params)
            self.assertEqual(params[-1], 1)
            # Title-color searches can retain selected seeds, but must always omit cart items.
            retrieve_products(query_context='blue dress', selected_product_ids=['seed'],
                              excluded_product_ids=['cart'], filters={'title_color': 'blue'})
            self.assertEqual(vectors.call_count, 3)
            for call in [bm25.call_args, *vectors.call_args_list]:
                self.assertIn('p.id <> ALL(%s)', call.kwargs['category_clause'])
                self.assertIn(['cart'], call.kwargs['category_params'])
