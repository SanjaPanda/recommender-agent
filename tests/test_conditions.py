"""Unsupported condition requests and safe loading of saved shopping state."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from agent.models import Filters, Plan, RouterDecision
from agent.conditions import requests_condition, UNSUPPORTED_CONDITION
from agent.workflow import run_turn, _retrieve
from agent.sessions import sanitize_apparel_products
from agent.tools.retrieval import retrieve_products


class ConditionTests(unittest.TestCase):
    def test_condition_fields_are_not_part_of_active_contracts(self):
        for value in (True, False, None):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Filters(is_secondhand=value)
        with self.assertRaises(ValidationError):
            RouterDecision(intent='recommend', filters={'is_secondhand': False})
        self.assertNotIn('is_secondhand', str(RouterDecision.model_json_schema()))

    def test_explicit_conditions_are_recognized_without_confusing_names_or_commands(self):
        for text in ('new dresses', 'only used shoes', 'no second-hand clothes', 'preowned bags',
                     'pre-owned jewelry', 'brand new shirts', 'unused socks', 'good condition',
                     'exclude used items', 'second hand coats', 'never worn dress', 'unworn shirts',
                     'worn once jacket', 'pre-loved bags', 'thrifted coats'):
            with self.subTest(text=text):
                self.assertTrue(requests_condition(text))
        for text in ('New Balance shoes', 'New York shirt', 'New Era cap', 'new search',
                     'new arrivals', '#buynew', 'vintage style dress', 'a belt used to hold trousers',
                     'air conditioning print shirt'):
            with self.subTest(text=text):
                self.assertFalse(requests_condition(text))

    @patch('agent.workflow.list_categories', return_value=[])
    def test_condition_requests_preserve_snapshot_without_model_or_retrieval(self, _):
        previous = {'plan': Plan(intent='recommend', query='dress').model_dump(),
                    'recommendations': [{'id': 'dress', 'title': 'Dress'}],
                    'cart': [{'id': 'cart'}], 'selected_product_ids': ['dress']}
        for field in ('query', 'refinement'):
            with self.subTest(field=field), patch('agent.workflow.complete') as model, \
                 patch('agent.workflow.extract_search_intent') as extract, \
                 patch('agent.workflow.prepare_query', return_value=None), patch('agent.workflow.retrieve_products') as retrieve:
                result = run_turn({**previous, 'turn': {field: 'only new dresses'}})
                self.assertEqual(result['status'], 'clarify')
                self.assertEqual(result['message'], UNSUPPORTED_CONDITION)
                for key in previous:
                    self.assertEqual(result[key], previous[key])
                model.assert_not_called()
                extract.assert_not_called()
                retrieve.assert_not_called()

    def test_initial_condition_request_and_saved_preference_do_not_search(self):
        with patch('agent.workflow.list_categories', return_value=[]), \
             patch('agent.workflow.complete') as model, patch('agent.workflow.prepare_query', return_value=None), patch('agent.workflow.retrieve_products') as retrieve:
            result = run_turn({'turn': {'query': 'used shoes'}})
            self.assertEqual(result['message'], UNSUPPORTED_CONDITION)
            plan = Plan(intent='recommend', query='shoes', preferences=[{'requirement': 'only new'}])
            result = _retrieve({'plan': plan.model_dump(), 'selected_product_ids': ['seed']})
            self.assertEqual(result['route'], 'respond')
            self.assertEqual(result['message'], UNSUPPORTED_CONDITION)
            model.assert_not_called()
            retrieve.assert_not_called()

    def test_direct_retrieval_rejects_conditions_before_encoders_or_database(self):
        with patch('agent.tools.retrieval.embed_texts') as encode, patch('agent.tools.retrieval.psycopg.connect') as connect:
            with self.assertRaisesRegex(ValueError, 'Condition filtering is not supported'):
                retrieve_products(query_context='used shoes')
            encode.assert_not_called()
            connect.assert_not_called()

    def test_saved_sessions_drop_condition_without_mutating_history_or_cart(self):
        for condition in (True, False, None):
            with self.subTest(condition=condition):
                card = {'id': 'dress', 'title': 'Old title', 'is_secondhand': condition,
                        'source_record_sha256': 'source', 'image_sha256': 'image', 'text_similarity': .8}
                original = {'plan': {**Plan(intent='recommend', query='dress').model_dump(),
                                    'filters': {'brand': 'Brand', 'is_secondhand': condition}},
                            'recommendations': [card], 'cart': [card], 'selected_product_ids': ['dress'],
                            'messages': [{'role': 'user', 'content': 'historical request'}]}
                before = deepcopy(original)
                connection = MagicMock()
                connection.execute.return_value.fetchall.return_value = [
                    ('dress', 'Description', [], {}, {}, 'Current title', 'Brand',
                     'Apparel & Accessories', None, 'source', 'image')]
                result = sanitize_apparel_products(connection, original)
                Plan.model_validate(result['plan'])
                self.assertNotIn('is_secondhand', result['plan']['filters'])
                self.assertNotIn('is_secondhand', result['cart'][0])
                self.assertEqual(result['cart'][0]['title'], 'Current title')
                self.assertEqual(result['messages'], before['messages'])
                self.assertEqual(original, before)
                self.assertNotIn('is_secondhand', connection.execute.call_args.args[0])
                if condition is None:
                    self.assertEqual(result['recommendations'], [])  # Already in the cart.
                    self.assertEqual(result['selected_product_ids'], [])
                    self.assertNotEqual(result.get('status'), 'clarify')
                else:
                    self.assertEqual(result['recommendations'], [])
                    self.assertEqual(result['selected_product_ids'], [])
                    self.assertIn('not supported', result['message'])

    def test_resume_saved_session_through_api_and_continue_with_cart(self):
        from agent.api import SessionRequest, session
        saved = {'plan': {'intent': 'recommend', 'filters': {'is_secondhand': False}},
                  'messages': [{'role': 'user', 'content': 'new clothes'}]}
        with patch('agent.api.psycopg.connect') as connect, \
             patch('agent.tools.retrieval.list_categories', return_value=[]):
            connect.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = (saved,)
            response = session(SessionRequest(thread_id='saved'))
        self.assertEqual(response['status'], 'clarify')
        self.assertEqual(response['message'], UNSUPPORTED_CONDITION)
        self.assertNotIn('is_secondhand', response['filters'])
        with patch('agent.workflow.list_categories', return_value=[]), patch('agent.workflow.complete') as model:
            state = run_turn({
                'plan': {**saved['plan'], 'filters': {}}, 'turn': {'action': 'cart_show'}})
        self.assertEqual(state['cart'], [])
        model.assert_not_called()

    def test_cli_rejects_retired_flag_and_condition_query(self):
        from rag.search import main
        for argv in (['search', 'dress', '--secondhand', 'false'], ['search', 'used shoes']):
            with self.subTest(argv=argv), patch('sys.argv', argv), \
                 patch('sys.stderr'), patch('rag.search.retrieve_products') as retrieve:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                retrieve.assert_not_called()

    def test_images_outside_catalog_are_not_served(self):
        from agent.api import _find_image_path
        from rag.catalog import local_asset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / '.data' / 'outside-catalog'
            saved.mkdir(parents=True)
            (saved / 'shirt.jpg').write_bytes(b'outside fixture')
            with patch('agent.api.ROOT', root):
                self.assertIsNone(_find_image_path({'local_path': 'shirt.jpg'}))
            with self.assertRaises(ValueError):
                local_asset(root / '.data' / 'catalog', '../outside-catalog/shirt.jpg')
