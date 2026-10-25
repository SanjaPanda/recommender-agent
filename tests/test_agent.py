"""Agent state, explicit actions, typed router output and local model failures."""
import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from agent.config import Settings, settings
from agent.llm import ModelError
from agent.models import ExplanationRequest, Filters, Plan, ProductFacts, RouterDecision, SearchIntent, Summary
from agent.workflow import build_recommendation_agent, sanitize_apparel_products
from rag.retrieval import filter_clause

CATEGORIES = [{'name': 'Apparel & Accessories > Clothing', 'count': 10}]
PRODUCT = {'id': 'one', 'title': 'Blue Dress', 'description': 'Cotton dress',
           'text_similarity': .8, 'image_similarity': .3, 'example_similarity': None}


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
        self.assertEqual(state['recommendations'], [])
        self.assertEqual(state['selected_product_ids'], [])
        self.assertEqual(state['cart'], [{'id': 'shirt'}])
        self.assertEqual(state['messages'], previous['messages'])

    def test_filters_are_parameterized_and_condition_is_rejected(self):
        sql, params = filter_clause({'brand': "Nike' OR true --"}, ['Apparel & Accessories'])
        self.assertNotIn('Nike', sql)
        self.assertIn("Nike' OR true --", params)
        self.assertNotIn('ILIKE', sql)
        self.assertIn("starts_with(p.raw_source->>'ground_truth_category'", sql)
        with self.assertRaisesRegex(ValueError, 'Apparel'):
            filter_clause({'category_prefixes': ['Health & Beauty']}, [])
        for unsupported in ({'price': 100}, {'is_secondhand': False}, {'is_secondhand': None}):
            with self.assertRaises(ValueError):
                filter_clause(unsupported, [])

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
    def test_explicit_cart_does_not_call_model(self, _):
        state = {'plan': Plan(intent='recommend').model_dump(), 'recommendations': [PRODUCT],
                 'turn': {'action': 'cart_add', 'product_ids': ['one']}}
        with patch('agent.workflow.complete') as model:
            result = build_recommendation_agent().invoke(state)
            self.assertEqual(result['cart'][0]['id'], 'one')
            model.assert_not_called()
        with self.assertRaises(ValueError):
            build_recommendation_agent().invoke({**state, 'turn': {'action': 'cart_add', 'product_ids': ['invented']}})
