"""Regression checks for atomic edits and unchanged shopping context."""
import unittest
from agent.models import SearchIntent
from unittest.mock import patch

from agent.edits import AmbiguousEdit, apply_edits
from agent.models import Filters, Plan, TurnEdits
from agent.workflow import build_recommendation_agent


class EditTests(unittest.TestCase):
    def setUp(self):
        # Routing/relevance tests isolate the separately tested LLM extractor.
        intent_patch = patch('agent.workflow.extract_search_intent', return_value=SearchIntent())
        intent_patch.start()
        self.addCleanup(intent_patch.stop)
        self.old = Plan(intent='recommend', query='dress', filters=Filters(brand='vidaXL'),
                        preferences=[{'requirement': 'not blue'},
                                     {'requirement': 'lightweight', 'strength': 'soft'}])

    def apply(self, edits, message='Switch to Trendsi; blue is fine'):
        return apply_edits(self.old, TurnEdits(intent='recommend', edits=edits), message)

    def test_brand_replacement_and_removal_preserve_other_context(self):
        result = self.apply([
            dict(field='brand', operation='replace', value='Trendsi', evidence='Trendsi'),
            dict(field='preference', operation='remove', target='not blue', evidence='blue is fine')])
        self.assertEqual(result.filters.brand, 'Trendsi')
        self.assertIsNone(result.filters.is_secondhand)
        self.assertEqual(result.query, 'dress')
        self.assertEqual([p.requirement for p in result.preferences], ['lightweight'])
        self.assertEqual(self.old.filters.brand, 'vidaXL')
        self.assertEqual(len(self.old.preferences), 2)

    def test_bad_later_edit_does_not_apply_earlier_brand_change(self):
        with self.assertRaises(AmbiguousEdit):
            self.apply([dict(field='brand', operation='replace', value='Trendsi', evidence='Trendsi'),
                        dict(field='preference', operation='remove', target='nonexistent', evidence='blue is fine')])
        self.assertEqual(self.old.filters.brand, 'vidaXL')

    def test_unrequested_condition_is_rejected(self):
        with self.assertRaises(AmbiguousEdit):
            self.apply([dict(field='is_secondhand', operation='replace', value='new', evidence='Trendsi')])

    def test_missing_edits_or_ignored_withdrawal_require_clarification(self):
        with self.assertRaises(AmbiguousEdit):
            self.apply([])
        with self.assertRaises(AmbiguousEdit):
            self.apply([dict(field='brand', operation='replace', value='Trendsi', evidence='Trendsi')])

    def test_permission_is_not_positive_preference(self):
        with self.assertRaises(AmbiguousEdit):
            self.apply([dict(field='preference', operation='add', value='blue', evidence='blue is fine')])

    def test_conflicting_preferences_are_rejected(self):
        with self.assertRaises(AmbiguousEdit):
            self.apply([dict(field='preference', operation='add', value='blue', evidence='blue')], 'blue')

    def test_add_multiple_and_replace_strength(self):
        result = self.apply([
            dict(field='preference', operation='add', value='cotton', evidence='cotton'),
            dict(field='preference', operation='add', value='halter', evidence='halter'),
            dict(field='preference', operation='replace', target='lightweight', value='lightweight',
                 strength='hard', evidence='must be lightweight')], 'cotton halter; must be lightweight')
        self.assertEqual(len(result.preferences), 4)
        self.assertEqual(result.preferences[-1].strength, 'hard')

    def test_remove_filter_and_set_condition_false(self):
        result = self.apply([
            dict(field='brand', operation='remove', evidence='any brand'),
            dict(field='is_secondhand', operation='add', value='new', evidence='brand new')],
            'any brand, brand new')
        self.assertIsNone(result.filters.brand)
        self.assertIs(result.filters.is_secondhand, False)

    def test_category_and_query_changes(self):
        result = self.apply([
            dict(field='category', operation='add', value='Apparel & Accessories', evidence='apparel'),
            dict(field='query', operation='replace', value='shirt', evidence='shirt')], 'apparel shirt')
        self.assertEqual(result.filters.category_prefixes, ['Apparel & Accessories'])
        self.assertEqual(result.query, 'shirt')

    def test_unknown_evidence_and_duplicate_edits_rejected(self):
        edit = dict(field='brand', operation='replace', value='Trendsi', evidence='Trendsi')
        with self.assertRaises(AmbiguousEdit):
            self.apply([edit], 'Adidas')
        with self.assertRaises(AmbiguousEdit):
            self.apply([edit, edit])

    @patch('agent.workflow.list_categories', return_value=[])
    def test_clear_brand_retrieves_with_other_constraints_unchanged(self, _):
        turn = TurnEdits(intent='recommend', edits=[
            dict(field='brand', operation='remove', evidence='Any brand is fine')])
        with patch('agent.workflow.complete', return_value=turn), \
             patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            result = build_recommendation_agent().invoke({'plan': self.old.model_dump(),
                'turn': {'query': 'Any brand is fine. Keep all other preferences.'}})
        self.assertIsNone(result['plan']['filters']['brand'])
        self.assertEqual(result['plan']['preferences'], self.old.model_dump()['preferences'])
        self.assertEqual(retrieve.call_args.kwargs['query_context'], 'dress not blue lightweight')
        self.assertIsNone(retrieve.call_args.kwargs['filters']['brand'])

    @patch('agent.workflow.list_categories', return_value=[])
    def test_soft_add_preserves_exclusion_and_does_not_inherit_old_intent(self, _):
        previous = self.old.model_copy(deep=True)
        previous.intent = 'clarify'
        previous.preferences = previous.preferences[:1]
        turn = TurnEdits(intent='recommend', edits=[dict(field='preference', operation='add',
            value='lightweight', strength='soft', evidence='prefer lightweight')])
        with patch('agent.workflow.complete', return_value=turn) as model, \
             patch('agent.workflow.retrieve_products', return_value=[]) as retrieve:
            result = build_recommendation_agent().invoke({'plan': previous.model_dump(),
                'turn': {'query': 'Also prefer lightweight.'}})
        self.assertEqual(result['plan']['preferences'], self.old.model_dump()['preferences'])
        self.assertEqual(result['plan']['filters'], previous.filters.model_dump())
        self.assertEqual(model.call_args.args[2].previous_plan.intent, 'clarify')
        retrieve.assert_called_once()

    @patch('agent.workflow.list_categories', return_value=[])
    def test_recorded_mixed_clarify_and_edits_stays_atomic(self, _):
        turn = TurnEdits(intent='clarify', edits=[
            dict(field='brand', operation='remove', evidence='Any brand is fine'),
            dict(field='is_secondhand', operation='remove', evidence='keep all other preferences')])
        with patch('agent.workflow.complete', return_value=turn), \
             patch('agent.workflow.retrieve_products') as retrieve:
            result = build_recommendation_agent().invoke({'plan': self.old.model_dump(),
                'turn': {'query': 'Any brand is fine; keep all other preferences'}})
        self.assertEqual(result['status'], 'clarify')
        self.assertEqual(result['plan'], self.old.model_dump())
        retrieve.assert_not_called()

    @patch('agent.workflow.list_categories', return_value=[])
    def test_ambiguous_turn_preserves_snapshot_and_skips_retrieval(self, _):
        edits = TurnEdits(intent='recommend', edits=[dict(field='preference', operation='remove',
                                                       target='unknown', evidence='remove it')])
        with patch('agent.workflow.complete', return_value=edits), \
             patch('agent.workflow.retrieve_products') as retrieve:
            result = build_recommendation_agent().invoke({'plan': self.old.model_dump(),
                'recommendations': [{'id': 'existing', 'title': 'Shirt'}], 'cart': [{'id': 'cart-item'}],
                'turn': {'query': 'remove it'}})
        self.assertEqual(result['status'], 'clarify')
        self.assertEqual(result['plan'], self.old.model_dump())
        self.assertEqual(result['recommendations'], [{'id': 'existing', 'title': 'Shirt'}])
        self.assertEqual(result['cart'], [{'id': 'cart-item'}])
        retrieve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
