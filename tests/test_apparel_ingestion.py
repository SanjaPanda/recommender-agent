"""The local ingestion contract accepts Apparel records only."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.db.ingest_sample import DEFAULT_INPUTS, is_apparel, read_apparel_rows
from scripts.data.fetch_sample import load_product_ids


class ApparelIngestionTests(unittest.TestCase):
    def test_pinned_source_count_is_approximately_two_thousand(self):
        if not all(path.is_file() for path in DEFAULT_INPUTS):
            self.skipTest("Download the pinned apparel train/test samples to validate their count.")
        self.assertEqual(len(read_apparel_rows(DEFAULT_INPUTS)), 2173)

    def test_pinned_apparel_identity_map_preserves_source_database_ids(self):
        ids = load_product_ids()
        self.assertEqual(len(ids), 2173)
        self.assertEqual(len(set(ids.values())), 2173)
        self.assertTrue(all(len(key) == 64 and value for key, value in ids.items()))

    def test_only_apparel_root_and_descendants_are_accepted(self):
        self.assertTrue(is_apparel({'ground_truth_category': 'Apparel & Accessories'}))
        self.assertTrue(is_apparel({'ground_truth_category': 'Apparel & Accessories > Clothing > Dresses'}))
        for category in ('Health & Beauty', 'Apparel & Accessories Extra', None):
            with self.subTest(category=category):
                self.assertFalse(is_apparel({'ground_truth_category': category}))

    def test_mixed_files_are_filtered_deduplicated_and_limited_by_apparel_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'train.jsonl', Path(directory) / 'test.jsonl'
            first.write_text('\n'.join(json.dumps(row) for row in (
                {'id': 'shirt', 'ground_truth_category': 'Apparel & Accessories > Clothing'},
                {'id': 'face-cream', 'ground_truth_category': 'Health & Beauty'},
                {'id': 'socks', 'ground_truth_category': 'Apparel & Accessories > Clothing Accessories'},
            )) + '\n', encoding='utf-8')
            second.write_text('\n'.join(json.dumps(row) for row in (
                {'id': 'shirt', 'ground_truth_category': 'Apparel & Accessories > Clothing'},
                {'id': 'coat', 'ground_truth_category': 'Apparel & Accessories > Clothing'},
            )) + '\n', encoding='utf-8')
            rows = read_apparel_rows((first, second))
            self.assertEqual([row['id'] for row in rows], ['shirt', 'socks', 'coat'])
            self.assertEqual([row['id'] for row in read_apparel_rows((first, second), limit=2)],
                             ['shirt', 'socks'])
