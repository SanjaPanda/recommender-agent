"""Behavioral checks for constraints, routing, verification and custom-model failures."""
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from agent.models import Decision, Filters, Plan
from agent.llm import ModelError
from agent.verification import verify_candidates
from agent.workflow import build_recommendation_agent
from rag.retrieval import filter_clause

CATEGORIES = [{'name': 'Apparel & Accessories', 'count': 10}]
PRODUCT = {'id': 'one', 'title': 'Red shirt', 'description': 'Cotton shirt', 'score': 1.0}


class AgentTests(unittest.TestCase):
    def test_verification_passes_remaining_budget_to_vision(self):
        call = Mock(side_effect=[Decision(verdict='uncertain', evidence='No color text'),
                                 Decision(verdict='matches', evidence='Red garment')])
        with patch('agent.verification.time.monotonic', side_effect=[0, 10, 30, 35, 40]), \
             patch('agent.verification.local_image', return_value=Path('example.jpg')):
            eligible, uncertain, _ = verify_candidates([PRODUCT],
                [{'requirement': 'red', 'visual': True}], call=call)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(uncertain, [])
        self.assertEqual(call.call_args_list[0].kwargs['timeout'], 110)
        self.assertEqual(call.call_args_list[1].kwargs['timeout'], 85)

    def test_late_verdict_is_uncertain_and_stops_further_calls(self):
        call = Mock(return_value=Decision(verdict='matches', evidence='Red garment'))
        with patch('agent.verification.time.monotonic', side_effect=[0, 1, 121, 122]):
            eligible, uncertain, excluded = verify_candidates(
                [PRODUCT, {**PRODUCT, 'id': 'two'}], [{'requirement': 'red'}], call=call)
        self.assertEqual(eligible + excluded, [])
        self.assertEqual(len(uncertain), 2)
        call.assert_called_once()

    def test_model_uses_supplied_timeout(self):
        from agent.llm import complete
        response = Mock()
        response.read.return_value = b'{"message":{"content":"{\\"verdict\\":\\"matches\\",\\"evidence\\":\\"Red garment\\"}"}}'
        with patch.dict('os.environ', {'AGENT_MODEL': 'test', 'OLLAMA_URL': 'http://127.0.0.1:11434'}), \
             patch('agent.llm.urllib.request.urlopen') as request:
            request.return_value.__enter__.return_value = response
            self.assertEqual(complete(Decision, 'Check', {}, timeout=7).verdict, 'matches')
            self.assertEqual(request.call_args.kwargs['timeout'], 7)
            request.reset_mock()
            with self.assertRaises(ModelError):
                complete(Decision, 'Check', {}, timeout=0)
            request.assert_not_called()

    def test_vision_only_for_unresolved_visual_requirement(self):
        call = Mock(side_effect=[Decision(verdict='uncertain', evidence='No color information'),
                                 Decision(verdict='violates', evidence='Garment is blue')])
        with patch('agent.verification.local_image', return_value=Path('example.jpg')):
            eligible, _, excluded = verify_candidates([PRODUCT],
                [{'requirement': 'not blue', 'visual': True}], call=call)
        self.assertEqual(eligible, [])
        self.assertEqual(len(excluded), 1)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args.kwargs['image'], Path('example.jpg'))
        known = Mock(return_value=Decision(verdict='matches', evidence='Explicitly red, no blue'))
        with patch('agent.verification.local_image') as resolve:
            verify_candidates([PRODUCT], [{'requirement': 'not blue', 'visual': True}], call=known)
            resolve.assert_not_called()

    @patch('agent.workflow.list_categories', return_value=CATEGORIES)
    def test_replacement_plan_overwrites_previous_constraints(self, _):
        old = Plan(intent='recommend', filters=Filters(brand='Nike'), preferences=[{'requirement': 'not blue'}])
        new = Plan(intent='recommend', filters=Filters(brand='Adidas'), preferences=[])
        with patch('agent.workflow.complete', return_value=new) as router, \
             patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            state = build_recommendation_agent().invoke({'plan': old.model_dump(),
                'turn': {'query': 'Actually Adidas, and blue is fine'}})
            self.assertEqual(router.call_args.args[2]['previous_plan']['filters']['brand'], 'Nike')
            self.assertEqual(state['plan']['preferences'], [])
            self.assertEqual(retrieve.call_args.kwargs['filters']['brand'], 'Adidas')

    def test_filters_are_parameterized_and_false_is_preserved(self):
        sql, params = filter_clause({'brand': "Nike' OR true --", 'is_secondhand': False}, ['Apparel & Accessories'])
        self.assertNotIn('Nike', sql)
        self.assertIn(False, params)
        self.assertIn("Nike' OR true --", params)
        self.assertNotIn('ILIKE', sql)
        with self.assertRaises(ValueError):
            filter_clause({'price': 100}, [])

    def test_hard_exclusion_uncertainty_and_soft_ranking(self):
        call = Mock(side_effect=[Decision(verdict='matches', evidence='Red garment'),
                                 Decision(verdict='violates', evidence='Blue garment'),
                                 Decision(verdict='uncertain', evidence='Image unavailable')])
        products = [{**PRODUCT, 'id': str(i)} for i in range(3)]
        eligible, uncertain, excluded = verify_candidates(products, [{'requirement': 'not blue', 'strength': 'hard'}], call=call)
        self.assertEqual([p['id'] for p in eligible], ['0'])
        self.assertEqual([p['id'] for p in excluded], ['1'])
        self.assertEqual([p['id'] for p in uncertain], ['2'])
        self.assertEqual(call.call_count, 3)

    def test_verifier_failure_never_passes_hard_constraint(self):
        eligible, uncertain, _ = verify_candidates([PRODUCT], [{'requirement': 'not blue'}], call=Mock(side_effect=ModelError('offline')))
        self.assertEqual(eligible, [])
        self.assertEqual(len(uncertain), 1)

    def test_no_preferences_makes_no_model_calls(self):
        call = Mock()
        eligible, _, _ = verify_candidates([PRODUCT], [], call=call)
        self.assertEqual(len(eligible), 1)
        call.assert_not_called()

    def test_soft_preference_violation_is_not_exclusion(self):
        eligible, uncertain, excluded = verify_candidates([PRODUCT], [{'requirement': 'black', 'strength': 'soft'}],
            call=Mock(return_value=Decision(verdict='violates', evidence='Red shirt')))
        self.assertEqual(len(eligible), 1)
        self.assertEqual(uncertain + excluded, [])

    @patch('agent.workflow.list_categories', return_value=CATEGORIES)
    def test_new_session_and_reset_preserve_cart(self, _):
        with patch('agent.workflow.complete') as model:
            state = build_recommendation_agent().invoke({'turn': {}, 'cart': [{'id': 'one', 'title': 'Shirt'}]})
            self.assertEqual(state['status'], 'categories')
            reset = build_recommendation_agent().invoke({**state, 'turn': {'query': '#buynew'}})
            self.assertEqual(reset['cart'], state['cart'])
            self.assertEqual(reset['selected_product_ids'], [])
            model.assert_not_called()

    @patch('agent.workflow.list_categories', return_value=CATEGORIES)
    def test_filters_only_bypass_verification(self, _):
        plan = Plan(intent='recommend', filters=Filters(brand='Nike', is_secondhand=False))
        with patch('agent.workflow.complete', side_effect=[plan, ModelError('summary offline')]), \
             patch('agent.workflow.retrieve_products', return_value=[PRODUCT]) as retrieve, \
             patch('agent.workflow.verify_candidates') as verify:
            state = build_recommendation_agent().invoke({'turn': {'query': 'new Nike products'}})
            self.assertEqual(state['recommendations'][0]['id'], 'one')
            self.assertEqual(retrieve.call_args.kwargs['filters']['is_secondhand'], False)
            self.assertEqual(retrieve.call_args.kwargs['query_context'], '')
            verify.assert_not_called()

    @patch('agent.workflow.list_categories', return_value=CATEGORIES)
    def test_empty_verified_pool_does_not_refill(self, _):
        plan = Plan(intent='recommend', preferences=[{'requirement': 'not blue'}])
        with patch('agent.workflow.complete', return_value=plan), \
             patch('agent.workflow.retrieve_products', return_value=[PRODUCT]) as retrieve, \
             patch('agent.workflow.verify_candidates', return_value=([], [PRODUCT], [])) as verify:
            state = build_recommendation_agent().invoke({'turn': {'query': 'no blue'}})
            self.assertEqual(state['status'], 'no_candidates')
            retrieve.assert_called_once()
            verify.assert_called_once()

    @patch('agent.workflow.list_categories', return_value=CATEGORIES)
    def test_explicit_cart_does_not_call_model(self, _):
        state = {'plan': Plan(intent='recommend').model_dump(), 'recommendations': [PRODUCT],
                 'turn': {'action': 'cart_add', 'product_ids': ['one']}}
        with patch('agent.workflow.complete') as model:
            result = build_recommendation_agent().invoke(state)
            self.assertEqual(result['cart'][0]['id'], 'one')
            model.assert_not_called()
        with self.assertRaises(ValueError):
            build_recommendation_agent().invoke({**state, 'turn': {'action': 'cart_add', 'product_ids': ['invented']}})


if __name__ == '__main__':
    unittest.main()
