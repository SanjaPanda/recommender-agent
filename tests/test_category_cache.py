"""Cross-process invalidation and snapshot correctness for category counts."""
import os
import unittest
from uuid import uuid4
from unittest.mock import MagicMock, patch

import psycopg
from psycopg.types.json import Jsonb
from agent.config import settings
import agent.tools.retrieval as retrieval


class CategoryCacheTests(unittest.TestCase):
    def setUp(self):
        retrieval._category_cache.clear()

    def test_revision_change_invalidates_and_returned_values_are_independent(self):
        connection = MagicMock()
        revision = [1]
        queries = []
        def execute(sql, *args):
            if sql.startswith('SELECT revision'):
                return MagicMock(fetchone=lambda: (revision[0],))
            if 'GROUP BY 1' in sql:
                queries.append(sql)
                return MagicMock(fetchall=lambda: [('Apparel & Accessories > Clothing', revision[0])])
            return MagicMock()
        connection.execute.side_effect = execute
        with patch('agent.tools.retrieval.psycopg.connect') as connect:
            connect.return_value.__enter__.return_value = connection
            first = retrieval.list_categories()
            first[0]['count'] = -1
            self.assertEqual(retrieval.list_categories()[0]['count'], 1)
            self.assertEqual(len(queries), 1)
            revision[0] = 2
            self.assertEqual(retrieval.list_categories()[0]['count'], 2)
            self.assertEqual(len(queries), 2)
        self.assertTrue(any('REPEATABLE READ, READ ONLY' in call.args[0]
                            for call in connection.execute.call_args_list))


@unittest.skipUnless(os.environ.get('RUN_DATABASE_TESTS') == '1', 'Opt-in PostgreSQL integration tests')
class CategoryCacheDatabaseTests(unittest.TestCase):
    def test_commits_in_other_connection_refresh_cache_and_rollback_does_not(self):
        identifier = 'cache-test-' + uuid4().hex
        category = 'Apparel & Accessories > CacheFixture' + uuid4().hex
        original = retrieval.list_categories()
        with psycopg.connect(settings.database_url.get_secret_value(), autocommit=True) as writer:
            try:
                writer.execute('BEGIN')
                writer.execute('INSERT INTO products(id,title,raw_source,source_name) VALUES (%s,%s,%s,%s)',
                               (identifier, 'Authored Shirt', Jsonb({'ground_truth_category': category}), 'authored-test'))
                self.assertEqual(retrieval.list_categories(), original)
                writer.execute('ROLLBACK')
                self.assertEqual(retrieval.list_categories(), original)
                writer.execute('INSERT INTO products(id,title,raw_source,source_name) VALUES (%s,%s,%s,%s)',
                               (identifier, 'Authored Shirt', Jsonb({'ground_truth_category': category}), 'authored-test'))
                self.assertIn({'name': category, 'count': 1}, retrieval.list_categories())
                changed = category + 'Changed'
                writer.execute('UPDATE products SET raw_source=%s WHERE id=%s',
                               (Jsonb({'ground_truth_category': changed}), identifier))
                refreshed = retrieval.list_categories()
                self.assertNotIn({'name': category, 'count': 1}, refreshed)
                self.assertIn({'name': changed, 'count': 1}, refreshed)
                writer.execute('DELETE FROM products WHERE id=%s', (identifier,))
                self.assertEqual(retrieval.list_categories(), original)
            finally:
                writer.execute('ROLLBACK')
                writer.execute('DELETE FROM products WHERE id=%s', (identifier,))
