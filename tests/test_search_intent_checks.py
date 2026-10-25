"""Authored evaluation must flag broader products and missed exclusions."""
import unittest

from agent.models import SearchIntent
from scripts.check_search_intent import check_intent


class SearchIntentCheckTests(unittest.TestCase):
    def test_compound_cannot_be_replaced_by_its_head(self):
        case = {'product_phrase': 'bow tie', 'attributes': []}
        self.assertFalse(check_intent(case, SearchIntent(product_phrase='tie'))['passed'])
        self.assertTrue(check_intent(case, SearchIntent(product_phrase='bow ties'))['passed'])

    def test_wrong_exclusion_flag_fails_required_attribute_check(self):
        case = {'product_phrase': 'dress', 'attributes': [{'name': 'color', 'value': 'blue', 'excluded': True}]}
        intent = SearchIntent(product_phrase='dress', attributes=[{'name': 'color', 'value': 'blue', 'evidence': 'blue'}])
        self.assertFalse(check_intent(case, intent)['passed'])

    def test_exclusion_only_case_has_no_positive_product(self):
        case = {'product_phrase': '', 'attributes': []}
        self.assertFalse(check_intent(case, SearchIntent(product_phrase='not wool'))['passed'])
        self.assertTrue(check_intent(case, SearchIntent())['passed'])
