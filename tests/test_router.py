"""One typed router decision updates state and declares filter removals."""
from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
from agent.models import Filters, Plan, RouterDecision, SearchIntent, SessionTurn
from agent.workflow import build_recommendation_agent, sanitize_apparel_products


class RouterTests(unittest.TestCase):
    def setUp(self):
        category = patch('agent.workflow.list_categories', return_value=[
            {'name': 'Apparel & Accessories > Clothing', 'count': 10}])
        category.start()
        self.addCleanup(category.stop)

    def test_new_search_routes_and_interprets_in_one_call(self):
        decision = RouterDecision(intent='recommend', query='blue dress', search_intent=SearchIntent(
            product_phrase='dress', attributes=[dict(name='color', value='blue', evidence='blue')]))
        with patch('agent.workflow.complete', return_value=decision) as model, \
             patch('agent.workflow.extract_search_intent') as extract, \
             patch('agent.workflow.retrieve_products', return_value=[{'id': 'dress', 'title': 'Blue Dress'}]) as retrieve:
            result = build_recommendation_agent().invoke({'turn': {'query': 'blue dress'}})
        model.assert_called_once()
        extract.assert_not_called()
        self.assertTrue(model.call_args.args[2].is_new_session)
        self.assertEqual(retrieve.call_args.kwargs['product_phrase'], 'dress')
        self.assertEqual(result['plan']['query'], 'blue dress')

    def test_existing_filters_and_preferences_are_supplied_to_router(self):
        previous = Plan(intent='recommend', query='dress', filters=Filters(brand='Nike'),
                        preferences=[{'requirement': 'cotton'}])
        decision = RouterDecision(**previous.model_dump(), search_intent=SearchIntent(product_phrase='dress',
            attributes=[dict(name='material', value='cotton', evidence='cotton')]))
        with patch('agent.workflow.complete', return_value=decision) as model, \
             patch('agent.workflow.retrieve_products', return_value=[]):
            result = build_recommendation_agent().invoke({'plan': previous.model_dump(),
                'messages': [{'role': 'user', 'content': 'previous search'}], 'turn': {'refinement': 'keep cotton'}})
        request = model.call_args.args[2]
        self.assertFalse(request.is_new_session)
        self.assertEqual(request.previous_plan, previous)
        self.assertEqual(result['plan']['filters'], previous.filters.model_dump())
        self.assertEqual(result['plan']['preferences'], previous.model_dump()['preferences'])

    def test_router_clears_brand_and_exposes_removal_flag(self):
        previous = Plan(intent='recommend', filters=Filters(brand='Nike'))
        decision = RouterDecision(intent='recommend', removed_filters=['brand'])
        with patch('agent.workflow.complete', return_value=decision), \
             patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            result = build_recommendation_agent().invoke({'plan': previous.model_dump(),
                'turn': {'refinement': 'Any brand is fine'}})
        self.assertIsNone(result['plan']['filters']['brand'])
        self.assertEqual(result['removed_filters'], ['brand'])
        self.assertIsNone(retrieve.call_args.kwargs['filters']['brand'])
        from agent.api import _response
        self.assertEqual(_response(result, 'test', 'Results', create_surface=False)['removed_filters'], ['brand'])

    def test_removal_flags_must_match_actual_filter_change(self):
        previous = Plan(intent='recommend', filters=Filters(brand='Nike'))
        for payload in ({'intent': 'recommend'},
                        {'intent': 'recommend', 'filters': {'brand': 'Nike'}, 'removed_filters': ['brand']},
                        {'intent': 'recommend', 'removed_filters': ['brand', 'brand']}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                RouterDecision.model_validate(payload, context={'previous_plan': previous})
        decision = RouterDecision.model_validate({'intent': 'recommend', 'removed_filters': ['brand']},
                                                context={'previous_plan': previous})
        self.assertIsNone(decision.filters.brand)

    def test_clarification_cannot_change_constraints_or_retrieve(self):
        previous = Plan(intent='recommend', query='dress', filters=Filters(brand='Nike'))
        with self.assertRaises(ValidationError):
            RouterDecision.model_validate({'intent': 'clarify', 'query': 'shoes', 'filters': {'brand': 'Nike'}},
                                          context={'previous_plan': previous})
        decision = RouterDecision(**{**previous.model_dump(), 'intent': 'clarify', 'message': 'Which preference?'})
        with patch('agent.workflow.complete', return_value=decision), patch('agent.workflow.retrieve_products') as retrieve:
            result = build_recommendation_agent().invoke({'plan': previous.model_dump(),
                'recommendations': [{'id': 'dress'}], 'cart': [{'id': 'cart'}],
                'turn': {'query': 'Change that'}})
        self.assertEqual(result['plan'], previous.model_dump())
        self.assertEqual(result['cart'], [{'id': 'cart'}])
        self.assertEqual(result['recommendations'], [{'id': 'dress'}])
        retrieve.assert_not_called()

    def test_color_decision_preserves_product_and_updates_effective_query(self):
        previous = Plan(intent='recommend', query='blue dress', preferences=[{'requirement': 'not black'}])
        decision = RouterDecision(intent='recommend', query='black dress', result_limit=1,
            filters=Filters(title_color='black'), search_intent=SearchIntent(product_phrase='dress',
                attributes=[dict(name='color', value='black', evidence='black')]))
        with patch('agent.workflow.complete', return_value=decision) as model, \
             patch('agent.workflow.extract_search_intent') as extract, \
             patch('agent.workflow.retrieve_products', return_value=[{'id': 'dress', 'title': 'Black Dress'}]) as retrieve:
            result = build_recommendation_agent().invoke({'plan': previous.model_dump(),
                'turn': {'refinement': 'find one in black'}})
        model.assert_called_once()
        extract.assert_not_called()
        self.assertEqual(retrieve.call_args.kwargs['query_context'], 'black dress')
        self.assertEqual(retrieve.call_args.kwargs['filters']['title_color'], 'black')
        self.assertEqual(result['plan']['preferences'], [])
        self.assertEqual(len(result['recommendations']), 1)
        self.assertEqual(previous.query, 'blue dress')

    def test_explicit_category_action_and_examples_use_existing_state(self):
        with patch('agent.workflow.complete') as model, \
             patch('agent.workflow.extract_search_intent') as extract, \
             patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            result = build_recommendation_agent().invoke({'turn': {
                'category_prefixes': ['Apparel & Accessories > Clothing']}})
            self.assertEqual(result['plan']['filters']['category_prefixes'], ['Apparel & Accessories > Clothing'])
            build_recommendation_agent().invoke({'plan': Plan(intent='recommend').model_dump(),
                'recommendations': [{'id': 'seed'}], 'turn': {'selected_product_ids': ['seed']}})
        self.assertEqual(retrieve.call_args.kwargs['selected_product_ids'], ['seed'])
        model.assert_not_called()
        extract.assert_not_called()

    def test_invalid_categories_and_duplicate_examples_fail_pydantic_validation(self):
        for value in ('Home & Garden', 'Apparel & Accessories Extra'):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Filters(category_prefixes=[value])
        with self.assertRaises(ValidationError):
            SessionTurn(selected_product_ids=['a', 'a'])
        with self.assertRaises(ValidationError):
            RouterDecision(intent='recommend', query='dress')

    def test_saved_unavailable_categories_are_removed_without_losing_cart_or_history(self):
        snapshot = {'plan': {'intent': 'recommend', 'filters': {'brand': 'Brand', 'category_prefixes': [
            'Home & Garden', 'Apparel & Accessories > Clothing']}}, 'recommendations': [{'id': 'dress'}],
            'selected_product_ids': ['dress'], 'cart': [{'id': 'dress'}],
            'messages': [{'role': 'user', 'content': 'saved request'}], 'uncertain': [], 'excluded_count': 0}
        before = deepcopy(snapshot)
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [('dress', 'Dress', '', 'Brand',
            'Apparel & Accessories > Clothing', None, [], None, None)]
        result = sanitize_apparel_products(connection, snapshot)
        Plan.model_validate(result['plan'])
        self.assertEqual(result['plan']['filters']['category_prefixes'], ['Apparel & Accessories > Clothing'])
        self.assertEqual(result['plan']['filters']['brand'], 'Brand')
        self.assertEqual(result['recommendations'], [])
        self.assertEqual(result['selected_product_ids'], [])
        self.assertEqual(result['cart'][0]['id'], 'dress')
        self.assertEqual(result['messages'], before['messages'])
        self.assertNotIn('uncertain', result)
        self.assertNotIn('excluded_count', result)
        self.assertEqual(snapshot, before)

    def test_router_json_is_validated_with_previous_state_before_returning(self):
        from agent.llm import complete, ModelError
        from agent.models import RouterRequest
        previous = Plan(intent='recommend', filters=Filters(brand='Nike'))
        response = MagicMock()
        response.read.return_value = json.dumps({'message': {'content': json.dumps({'intent': 'recommend'})}}).encode()
        with patch('agent.llm.urllib.request.urlopen') as request:
            request.return_value.__enter__.return_value = response
            with self.assertRaises(ModelError):
                complete(RouterDecision, 'Route', RouterRequest(previous_plan=previous, is_new_session=False,
                    user_message='Any brand', categories=[], displayed_products=[]),
                    validation_context={'previous_plan': previous})

    def test_withdrawal_and_brand_replacement_are_used_without_reinterpreting_edits(self):
        previous = Plan(intent='recommend', query='dress', filters=Filters(brand='Nike'),
                        preferences=[{'requirement': 'not blue'}])
        decision = RouterDecision(intent='recommend', query='dress', filters=Filters(brand='Adidas'),
                                  search_intent=SearchIntent(product_phrase='dress'))
        with patch('agent.workflow.complete', return_value=decision), \
             patch('agent.workflow.extract_search_intent') as extract, \
             patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            result = build_recommendation_agent().invoke({'plan': previous.model_dump(),
                'turn': {'refinement': 'Adidas instead. Blue is fine; remove the exclusion.'}})
        self.assertEqual(result['plan']['preferences'], [])
        self.assertEqual(result['removed_filters'], [])  # Replacement is not removal.
        self.assertEqual(retrieve.call_args.kwargs['query_context'], 'dress')
        self.assertEqual(retrieve.call_args.kwargs['filters']['brand'], 'Adidas')
        extract.assert_not_called()

    def test_explicit_category_and_text_are_both_applied(self):
        category = 'Apparel & Accessories > Clothing'
        decision = RouterDecision(intent='recommend', query='dress', filters=Filters(category_prefixes=[category]),
                                  search_intent=SearchIntent(product_phrase='dress'))
        with patch('agent.workflow.complete', return_value=decision) as model, \
             patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            build_recommendation_agent().invoke({'turn': {'query': 'dress', 'category_prefixes': [category]}})
        self.assertEqual(model.call_args.args[2].previous_plan.filters.category_prefixes, [category])
        self.assertEqual(retrieve.call_args.kwargs['query_context'], 'dress')
        self.assertEqual(retrieve.call_args.kwargs['filters']['category_prefixes'], [category])

    def test_invalid_interpretation_and_conflicting_preferences_fail_before_state_update(self):
        with self.assertRaises(ValidationError):
            RouterDecision(intent='recommend', query='bow tie', search_intent=SearchIntent(product_phrase='tie'))
        for requirements in (['cotton', 'Cotton'], ['blue', 'not blue']):
            with self.subTest(requirements=requirements), self.assertRaises(ValidationError):
                Plan(intent='recommend', preferences=[{'requirement': value} for value in requirements])

    def test_router_uses_configured_timeout_and_releases_model(self):
        from agent.config import settings
        from agent.llm import complete, ModelError
        from agent.models import RouterRequest
        with patch.object(settings, 'agent_router_timeout_seconds', 19), \
             patch('agent.llm.urllib.request.urlopen', side_effect=TimeoutError) as request:
            with self.assertRaises(ModelError):
                complete(RouterDecision, 'Route', RouterRequest(previous_plan=Plan(intent='categories'),
                    is_new_session=True, user_message='dress', categories=[], displayed_products=[]))
        self.assertEqual(request.call_args.kwargs['timeout'], 19)
        self.assertEqual(json.loads(request.call_args.args[0].data)['keep_alive'], 0)
