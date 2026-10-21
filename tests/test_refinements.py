"""Explicit follow-ups keep context and constrain retrieval before ranking."""
import unittest
from unittest.mock import patch
from agent.models import Plan, Filters
from agent.refinements import color_refinement
from agent.workflow import build_recommendation_agent
from rag.retrieval import filter_clause


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.old = Plan(intent='recommend', query='leash',
            filters=Filters(category_prefixes=['Apparel & Accessories > Clothing']))

    def test_color_and_count_preserve_product_context(self):
        result = color_refinement(self.old, 'find 1 in black')
        self.assertEqual(result.query, 'leash')
        self.assertEqual(result.result_limit, 1)
        self.assertEqual(result.filters.title_color, 'black')
        self.assertEqual(result.filters.title_product, 'leash')
        self.assertEqual(result.filters.category_prefixes, self.old.filters.category_prefixes)
        self.assertIsNone(self.old.filters.title_color)

    def test_color_refinement_after_category_selection_uses_known_scope(self):
        category_plan = Plan(intent='recommend', filters=Filters(
            category_prefixes=['Apparel & Accessories > Clothing']))
        result = color_refinement(category_plan, 'find 1 in black')
        self.assertIsNotNone(result)
        self.assertEqual(result.result_limit, 1)
        self.assertEqual(result.filters.title_color, 'black')
        self.assertEqual(result.filters.category_prefixes,
                         ['Apparel & Accessories > Clothing'])

    def test_color_change_clear_and_unrecognized_phrases(self):
        black = color_refinement(self.old, 'find one in black')
        blue = color_refinement(black, 'show me two in blue')
        self.assertEqual((blue.result_limit, blue.filters.title_color), (2, 'blue'))
        self.assertIsNone(color_refinement(blue, 'any color is fine').filters.title_color)
        for text in ('not in black', 'prefer black', 'in black or blue', 'find 1 in black and remove the brand'):
            self.assertIsNone(color_refinement(self.old, text))
        self.assertIsNone(color_refinement(Plan(intent='categories'), 'find 1 in black'))

    def test_bare_color_followup_keeps_multiword_product_query(self):
        kids_clothes = Plan(intent='recommend', query='kids clothes', filters=Filters(
            category_prefixes=['Apparel & Accessories > Clothing']))
        blue = color_refinement(kids_clothes, 'blue')
        self.assertIsNotNone(blue)
        self.assertEqual(blue.query, 'kids clothes')
        self.assertEqual(blue.filters.title_color, 'blue')
        self.assertEqual(blue.filters.category_prefixes,
                         ['Apparel & Accessories > Clothing'])

    def test_bare_color_graph_turn_retrieves_with_preserved_product_context(self):
        kids_clothes = Plan(intent='recommend', query='kids clothes', filters=Filters(
            category_prefixes=['Apparel & Accessories > Clothing']))
        product = {'id': 'blue-kids-shirt', 'title': 'Blue Kids Shirt'}
        with patch('agent.workflow.list_categories', return_value=[{'name': 'Apparel & Accessories > Clothing'}]), \
             patch('agent.workflow.complete') as model, \
             patch('agent.workflow.retrieve_products', return_value=[product]) as retrieve:
            state = build_recommendation_agent().invoke({
                'plan': kids_clothes.model_dump(),
                'turn': {'refinement': 'blue'},
            })
        model.assert_not_called()
        self.assertEqual(retrieve.call_args.kwargs['query_context'], 'kids clothes')
        self.assertEqual(retrieve.call_args.kwargs['filters']['title_color'], 'blue')
        self.assertEqual(retrieve.call_args.kwargs['filters']['category_prefixes'],
                         ['Apparel & Accessories > Clothing'])

    def test_graph_keeps_seed_and_cart_and_uses_one_filtered_result(self):
        product = {'id': 'black-leash', 'title': 'Black leash'}
        with patch('agent.workflow.list_categories', return_value=[{'name': 'Apparel & Accessories > Clothing'}]), \
             patch('agent.workflow.complete') as model, \
             patch('agent.workflow.retrieve_products', return_value=[product]) as retrieve:
            state = build_recommendation_agent().invoke({'plan': self.old.model_dump(),
                'selected_product_ids': ['seed'], 'cart': [{'id': 'cart'}],
                'turn': {'refinement': 'find 1 in black'}})
        model.assert_not_called()
        self.assertEqual(retrieve.call_args.kwargs['query_context'], 'leash')
        self.assertEqual(retrieve.call_args.kwargs['selected_product_ids'], ['seed'])
        self.assertEqual(retrieve.call_args.kwargs['limit'], 20)
        self.assertEqual(retrieve.call_args.kwargs['filters']['title_color'], 'black')
        self.assertEqual(state['recommendations'], [product])
        self.assertEqual(state['cart'], [{'id': 'cart'}])
        self.assertNotIn('Fewer than', state['message'])
        self.assertEqual(Plan.model_validate(state['plan']).result_limit, 1)

    def test_color_filter_is_parameterized_and_title_only(self):
        sql, params = filter_clause({'title_color': 'black'}, [])
        self.assertIn('p.title', sql)
        self.assertNotIn('description', sql)
        self.assertNotIn('black', sql)
        self.assertEqual(params, ['Apparel & Accessories', 'Apparel & Accessories > ', r'\mblack\M'])
        with self.assertRaises(ValueError):
            filter_clause({'title_color': "black' OR true"}, [])
