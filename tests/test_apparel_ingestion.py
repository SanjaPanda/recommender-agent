"""The local ingestion contract accepts Apparel records only."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.db.ingest_sample import is_apparel, read_apparel_rows
from scripts.data.fetch_sample import load_product_ids


class ApparelIngestionTests(unittest.TestCase):
    def test_identity_map_is_optional_and_reads_authored_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identities.json'
            self.assertEqual(load_product_ids(path), {})
            expected = {'a' * 64: 'authored-shirt'}
            path.write_text(json.dumps(expected), encoding='utf-8')
            self.assertEqual(load_product_ids(path), expected)
            path.write_text('[]', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_product_ids(path)

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
