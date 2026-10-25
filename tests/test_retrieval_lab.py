"""Protect the experiment's arithmetic, fair seed exclusions and honest proxy labels."""
import unittest
import json
from pathlib import Path

import numpy as np

from notebooks.retrieval_lab import candidate_coverage, centroid, evidence, exclude_examples, fuse_components, ranked, summaries, unit_rows, visual_review_summary


def product(pid, image_hash='unique', title='Blue Dress'):
    """Return an authored lightweight candidate, without accessing the database."""
    return {'id': pid, 'title': title, 'description': '', 'features': [], 'brand': '',
            'image': None, 'image_sha256': image_hash, 'category': 'Apparel & Accessories > Clothing'}


class RetrievalLabTests(unittest.TestCase):
    def test_normalized_average_is_not_the_sum_of_cosine_magnitudes(self):
        direction = centroid(np.array([[3., 0.], [0., 7.]]))
        np.testing.assert_allclose(direction, np.array([1., 1.]) / np.sqrt(2))
        for invalid in ([[0., 0.]], [[float('nan'), 1.]], [[float('inf'), 1.]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                unit_rows(invalid)

    def test_stable_ties_and_exclusion_preserve_rank_order(self):
        products = [product('a'), product('b'), product('c')]
        self.assertEqual(ranked(np.array([.5, .5, .9]), products, {'c'}), [(0, .5), (1, .5)])
        with self.assertRaises(ValueError):
            ranked(np.array([.5, .5]), products)

    def test_every_fixed_seed_and_identical_photo_is_excluded_for_all_counts(self):
        products = [product('a', 'photo-a'), product('b', 'photo-b'),
                    product('copy-a', 'photo-a'), product('other', 'photo-other')]
        self.assertEqual(exclude_examples(products, ['a', 'b']), {'a', 'b', 'copy-a'})
        with self.assertRaises(ValueError):
            exclude_examples(products, ['missing'])

    def test_rrf_uses_ranks_rather_than_adding_incomparable_scores(self):
        products = [product('a'), product('b')]
        methods = {'bm25': [(0, 1000.), (1, .01)], 'nomic_text': [(1, .8), (0, .7)]}
        result = fuse_components(methods, products)
        self.assertEqual({index for index, _ in result}, {0, 1})
        self.assertAlmostEqual(result[0][1], .5 / 61 + .5 / 62)
        self.assertAlmostEqual(result[0][1], result[1][1])

    def test_missing_color_is_not_proof_of_successful_negation(self):
        case = {'id': 'exclude_blue', 'family': r'\bdress\b', 'reject': '', 'requirements': []}
        for title in ('Dress', 'Blue Dress'):
            self.assertEqual(evidence(product('a', title=title), case), (True, None))
        result = summaries({'nomic': [(0, .8)]}, [product('a')], case)[0]
        self.assertIsNone(result['listing_support_at_5'])
        self.assertEqual(result['family_at_5'], .2)  # Fixed denominator; no padding or score inflation.

    def test_incidental_dress_words_are_rejected_by_the_declared_proxy(self):
        cases = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/cases.json').read_text())
        case = next(case for case in cases['text_cases'] if case['id'] == 'blue_dress')
        self.assertEqual(evidence(product('a', title='Blue Dress Shoes'), case), (False, False))
        self.assertEqual(evidence(product('b', title='Blue Dress'), case), (True, True))
        self.assertEqual(evidence(product('c', title='Blue T-Shirt Dress'), case), (True, True))

    def test_absent_targets_and_unscored_exclusions_are_visible_in_coverage(self):
        cases = [{'id': 'bow_tie', 'family': r'\bbow tie\b', 'reject': '', 'requirements': ['pink']},
                 {'id': 'exclude_blue', 'family': r'\bdress\b', 'reject': '', 'requirements': []}]
        coverage = candidate_coverage([product('a')], cases)
        self.assertEqual(coverage['bow_tie'], {'title_family_count': 0, 'literal_listing_count': 0})
        self.assertIsNone(coverage['exclude_blue']['literal_listing_count'])

    def test_unjudged_or_ambiguous_photos_remain_unknown_in_visual_review(self):
        results = {'method': [{'id': pid} for pid in ('yes', 'no', 'ambiguous', 'unjudged', 'no-even-if-unknown')]}
        review = {'labels': {'yes': {'mostly_white': True, 'low_top': True},
                             'no': {'mostly_white': False, 'low_top': True},
                             'ambiguous': {'mostly_white': None, 'low_top': True},
                             'no-even-if-unknown': {'mostly_white': None, 'low_top': False}}}
        row = visual_review_summary(results, review)[0]
        self.assertEqual(row['supported'], 1)
        self.assertEqual(row['unknown'], 2)
        self.assertEqual(row['support_at_5_bounds'], [.2, .6])
