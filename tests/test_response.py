"""Product cards should not wait for optional generated prose by default."""
import unittest
from unittest.mock import patch

from agent.llm import ModelError
from agent.models import Plan, Summary
from agent.config import settings
from agent.workflow import _respond


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.products = [{'id': 'a', 'title': 'Red shirt'}, {'id': 'b', 'title': 'White shirt'}]
        self.state = {'route': 'retrieve', 'plan': Plan(intent='recommend', preferences=[{'requirement': 'not blue'}]).model_dump(),
                      'candidates': self.products, 'uncertain': [{'id': 'unverified'}]}

    def test_default_returns_cards_and_preference_notice_without_model_call(self):
        with patch.object(settings, 'agent_generate_summaries', False), \
             patch('agent.workflow.complete') as model:
            result = _respond(self.state)
        model.assert_not_called()
        self.assertEqual(result['recommendations'], self.products)
        self.assertIn('not verified', result['message'])
        self.assertIn('may not satisfy every preference or exclusion', result['message'])
        self.assertEqual(result['messages'][-1]['content'], result['message'])

    def test_opt_in_summary_is_bounded_and_keeps_cards_on_timeout(self):
        with patch.object(settings, 'agent_generate_summaries', True), \
             patch('agent.workflow.complete', side_effect=ModelError('timeout')) as model:
            result = _respond(self.state)
        self.assertEqual(model.call_args.kwargs['timeout'], 15)
        self.assertEqual(result['recommendations'], self.products)
        self.assertEqual(result['status'], 'ok')

    def test_successful_optional_summary_preserves_order_and_eligibility(self):
        with patch.object(settings, 'agent_generate_summaries', True), \
             patch('agent.workflow.complete', return_value=Summary(message='Two shirts.')):
            result = _respond(self.state)
        self.assertEqual(result['recommendations'], self.products)
        self.assertTrue(result['message'].endswith('Two shirts.'))

    def test_empty_pool_never_calls_summary(self):
        with patch.object(settings, 'agent_generate_summaries', True), \
             patch('agent.workflow.complete') as model:
            result = _respond({**self.state, 'candidates': []})
        model.assert_not_called()
        self.assertEqual(result['status'], 'no_candidates')

    def test_title_color_refinement_with_no_text_query_is_explained_clearly(self):
        plan = Plan(intent='recommend', filters={'title_color': 'black'})
        with patch.object(settings, 'agent_generate_summaries', False):
            result = _respond({**self.state, 'plan': plan.model_dump(),
                               'candidates': self.products})
        self.assertIn("Filtered to titles containing 'black'", result['message'])
        self.assertNotIn("Matching ''", result['message'])
