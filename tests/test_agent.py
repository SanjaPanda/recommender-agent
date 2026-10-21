"""Behavioral checks for constraints, routing, retrieval and custom-model failures."""
import unittest
import json
from unittest.mock import patch, Mock

from agent.models import ExplanationRequest, Summary, Filters, Plan, ProductFacts, TurnEdits
from agent.llm import ModelError
from agent.config import Settings, settings
from pydantic import ValidationError
from agent.workflow import _is_plain_search, build_recommendation_agent, sanitize_apparel_products
from rag.retrieval import filter_clause

CATEGORIES = [{'name': 'Apparel & Accessories', 'count': 10}]
PRODUCT = {
    'id': 'one', 'title': 'Red shirt', 'description': 'Cotton shirt', 'score': 1.0,
    'text_similarity': 0.65, 'image_similarity': 0.30, 'example_similarity': None,
}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.request = ExplanationRequest(plan=Plan(intent='recommend'), products=[ProductFacts(id='one')])

    def test_model_uses_supplied_timeout(self):
        from agent.llm import complete
        response = Mock()
        response.read.return_value = json.dumps({"message": {"content": json.dumps({"message": "Red garment"})}}).encode()
        with patch.object(settings, 'agent_model', 'test'), \
             patch.object(settings, 'ollama_url', 'http://127.0.0.1:11434'), \
             patch('agent.llm.urllib.request.urlopen') as request:
            request.return_value.__enter__.return_value = response
            self.assertEqual(complete(Summary, 'Check', self.request, timeout=7).message, 'Red garment')
            self.assertEqual(request.call_args.kwargs['timeout'], 7)
            request.reset_mock()
            with self.assertRaises(ModelError):
                complete(Summary, 'Check', self.request, timeout=0)
            request.assert_not_called()

    def test_malformed_model_output_fails_without_echoing_untrusted_content(self):
        from agent.llm import complete
        response = Mock()
        response.read.return_value = json.dumps({"message": {"content": '{"unexpected":"raw text"}'}}).encode()
        with patch.object(settings, 'agent_model', 'test'), \
             patch.object(settings, 'ollama_url', 'http://127.0.0.1:11434'), \
             patch('agent.llm.urllib.request.urlopen') as request:
            request.return_value.__enter__.return_value = response
            with self.assertRaisesRegex(ModelError, 'invalid Summary fields') as error:
                complete(Summary, 'Check', self.request, timeout=7)
        self.assertNotIn('raw text', str(error.exception))

    def test_thinking_is_disabled_by_default_and_configurable(self):
        from agent.llm import complete
        with patch.object(settings, 'agent_model', 'qwen3:1.7b'), \
             patch.object(settings, 'agent_think', False), \
             patch('agent.llm.urllib.request.urlopen', side_effect=TimeoutError) as request:
            for value, expected in [(None, False), ('false', False), ('true', True)]:
                with patch.object(settings, 'agent_think', expected):
                    with self.assertRaises(ModelError):
                        complete(Summary, 'Check', self.request, timeout=7)
                payload = json.loads(request.call_args.args[0].data)
                self.assertIs(payload['think'], expected)
                self.assertEqual(payload['options']['num_ctx'], 4096)
    def test_cpu_override_and_timeout_diagnostic(self):
        from agent.llm import complete
        with patch.object(settings, 'agent_model', 'test'), \
             patch.object(settings, 'agent_num_gpu', 0), \
             patch.object(settings, 'ollama_url', 'http://127.0.0.1:11434'), \
             patch('agent.llm.urllib.request.urlopen', side_effect=TimeoutError) as request:
            with self.assertRaisesRegex(ModelError, 'timed out after 7 seconds'):
                complete(Summary, 'Check', self.request, timeout=7)
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload['options']['num_gpu'], 0)
            self.assertEqual(request.call_count, 1)

    def test_settings_reject_invalid_hardware_and_timeout_values(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None)
        for values in ({'agent_num_gpu': -1}, {'agent_router_timeout_seconds': 0},
                       {'ollama_url': 'https://example.com'}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Settings(_env_file=None, database_url='postgresql://user:pass@localhost/db', **values)

    def test_saved_session_hides_non_apparel_product_references(self):
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = [('shirt',)]
        previous = {
            'recommendations': [{'id': 'shirt'}, {'id': 'garden-tool'}],
            'selected_product_ids': ['garden-tool'],
            'cart': [{'id': 'garden-tool'}, {'id': 'shirt'}],
            'messages': [{'role': 'user', 'content': 'find garden tools'}],
        }
        state = sanitize_apparel_products(connection, previous)
        self.assertEqual(state['recommendations'], [{'id': 'shirt'}])
        self.assertEqual(state['selected_product_ids'], [])
        self.assertEqual(state['cart'], [{'id': 'shirt'}])
        self.assertEqual(state['messages'], previous['messages'])

    @patch('agent.workflow.list_categories', return_value=[{
        'name': 'Apparel & Accessories > Clothing', 'count': 10,
    }])
    def test_ui_category_selection_uses_deterministic_apparel_filter(self, _):
        with patch('agent.workflow.complete') as model, \
             patch('agent.workflow.retrieve_products', return_value=[PRODUCT]) as retrieve:
            state = build_recommendation_agent().invoke({
                'turn': {'query': 'Show products in Apparel & Accessories > Clothing'},
            })
        model.assert_not_called()
        self.assertEqual(state['recommendations'], [PRODUCT])
        self.assertEqual(retrieve.call_args.kwargs['filters']['category_prefixes'],
                         ['Apparel & Accessories > Clothing'])
        self.assertEqual([event.node for event in state['trace']],
                         ['router', 'retrieve', 'respond'])
        self.assertEqual(state['trace'][1].text_count, 1)
        self.assertEqual(state['trace'][1].clip_count, 1)
        self.assertEqual(state['trace'][1].example_count, 0)
        self.assertEqual(state['trace'][1].result_count, 1)
        self.assertTrue(all(event.duration_ms >= 0 for event in state['trace']))

    @patch('agent.workflow.list_categories', return_value=CATEGORIES)
    def test_plain_product_search_skips_router_and_preserves_context(self, _):
        previous = Plan(
            intent='recommend',
            filters=Filters(category_prefixes=['Apparel & Accessories > Clothing']),
            preferences=[{'requirement': 'cotton'}],
        )
        displayed = [PRODUCT, {**PRODUCT, 'id': 'two', 'title': 'Blue shirt'}]
        with patch('agent.workflow.complete') as model, \
             patch('agent.workflow.retrieve_products', return_value=displayed) as retrieve:
            state = build_recommendation_agent().invoke({
                'plan': previous.model_dump(), 'recommendations': displayed,
                'selected_product_ids': ['one'],
                'turn': {'query': 'blue clothes', 'selected_product_ids': ['one']},
            })

        model.assert_not_called()
        self.assertEqual(state['plan']['query'], 'blue clothes')
        self.assertEqual(state['plan']['filters']['category_prefixes'],
                         ['Apparel & Accessories > Clothing'])
        self.assertEqual(retrieve.call_args.kwargs['query_context'], 'blue clothes cotton')
        self.assertEqual(retrieve.call_args.kwargs['selected_product_ids'], ['one'])
        self.assertEqual([event.node for event in state['trace']],
                         ['router', 'retrieve', 'respond'])
        self.assertEqual(state['trace'][1].text_count, 2)
        self.assertEqual(state['trace'][1].clip_count, 2)
        self.assertEqual(state['trace'][1].example_count, 0)

    def test_preference_edit_cues_keep_the_model_router(self):
        self.assertTrue(_is_plain_search('blue clothes'))
        self.assertTrue(_is_plain_search('show me a blue jacket'))
        self.assertFalse(_is_plain_search('no blue'))
        self.assertFalse(_is_plain_search('remove the black filter'))
        self.assertFalse(_is_plain_search('blue clothes without stripes'))

    @patch('agent.workflow.list_categories', return_value=[])
    def test_ui_category_selection_rejects_non_apparel_path(self, _):
        with self.assertRaisesRegex(ValueError, 'not in the apparel catalogue'):
            build_recommendation_agent().invoke({
                'turn': {'query': 'Show products in Home & Garden'},
            })

    def test_router_timeout_does_not_expand_summary_timeout(self):
        from agent.llm import complete
        with patch.object(settings, 'agent_model', 'test'), \
             patch.object(settings, 'agent_router_timeout_seconds', 300), \
             patch.object(settings, 'ollama_url', 'http://127.0.0.1:11434'), \
             patch('agent.llm.urllib.request.urlopen', side_effect=TimeoutError) as request:
            for schema, supplied, expected in [(Plan, {}, 300), (TurnEdits, {}, 300), (Summary, {}, 120),
                                               (Summary, {'timeout': 7}, 7)]:
                with self.assertRaises(ModelError):
                    complete(schema, 'Check', self.request, **supplied)
                self.assertEqual(request.call_args.kwargs['timeout'], expected)
            request.reset_mock()

    @patch('agent.workflow.list_categories', return_value=CATEGORIES)
    def test_replacement_plan_overwrites_previous_constraints(self, _):
        old = Plan(intent='recommend', filters=Filters(brand='Nike'), preferences=[{'requirement': 'not blue'}])
        new = TurnEdits(intent='recommend', edits=[{'field':'brand','operation':'replace','value':'Adidas','evidence':'Adidas'}, {'field':'preference','operation':'remove','target':'not blue','evidence':'blue is fine'}])
        with patch('agent.workflow.complete', return_value=new) as router, \
             patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            state = build_recommendation_agent().invoke({'plan': old.model_dump(),
                'turn': {'query': 'Actually Adidas, and blue is fine'}})
            self.assertEqual(router.call_args.args[2].previous_plan.filters.brand, 'Nike')
            self.assertEqual(state['plan']['preferences'], [])
            self.assertEqual(retrieve.call_args.kwargs['filters']['brand'], 'Adidas')

    def test_filters_are_parameterized_and_false_is_preserved(self):
        sql, params = filter_clause({'brand': "Nike' OR true --", 'is_secondhand': False}, ['Apparel & Accessories'])
        self.assertNotIn('Nike', sql)
        self.assertIn(False, params)
        self.assertIn("Nike' OR true --", params)
        self.assertNotIn('ILIKE', sql)
        self.assertIn("starts_with(p.raw_source->>'ground_truth_category'", sql)
        with self.assertRaisesRegex(ValueError, 'Apparel'):
            filter_clause({'category_prefixes': ['Health & Beauty']}, [])
        with self.assertRaises(ValueError):
            filter_clause({'price': 100}, [])

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
    def test_filters_only_use_sql_retrieval(self, _):
        plan = TurnEdits(intent='recommend', edits=[{'field':'brand','operation':'add','value':'Nike','evidence':'Nike'}, {'field':'is_secondhand','operation':'add','value':'new','evidence':'new'}])
        with patch('agent.workflow.complete', side_effect=[plan, ModelError('summary offline')]), \
             patch('agent.workflow.retrieve_products', return_value=[PRODUCT]) as retrieve:
            state = build_recommendation_agent().invoke({'turn': {'query': 'new Nike products'}})
            self.assertEqual(state['recommendations'][0]['id'], 'one')
            self.assertEqual(retrieve.call_args.kwargs['filters']['is_secondhand'], False)
            self.assertEqual(retrieve.call_args.kwargs['query_context'], '')

    @patch('agent.workflow.list_categories', return_value=CATEGORIES)
    @patch.dict('os.environ', {'AGENT_GENERATE_SUMMARIES': 'false'})
    def test_preferences_use_retrieval_without_reranker(self, _):
        plan = TurnEdits(intent='recommend', edits=[
            {'field': 'preference', 'operation': 'add', 'value': 'not blue',
             'evidence': 'no blue', 'visual': True}])
        products = [{**PRODUCT, 'id': str(i), 'score': 5 - i} for i in range(5)]
        with patch('agent.workflow.complete', return_value=plan) as model, \
             patch('agent.workflow.retrieve_products', return_value=products) as retrieve:
            state = build_recommendation_agent().invoke({'turn': {'query': 'no blue'}})
        model.assert_called_once()
        retrieve.assert_called_once()
        self.assertEqual(retrieve.call_args.kwargs['query_context'], 'not blue')
        self.assertEqual(retrieve.call_args.kwargs['limit'], 20)
        self.assertEqual(state['recommendations'], products)
        self.assertEqual(state['status'], 'ok')
        self.assertIn('not verified', state['message'])
        self.assertNotIn('verify', build_recommendation_agent().get_graph().nodes)

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
