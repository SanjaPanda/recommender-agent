"""Schema initialization and upgrades; database tests use a disposable database."""
from contextlib import redirect_stdout
from io import StringIO
import os
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from agent.config import settings
from scripts.db import apply_migrations as migrations


class MigrationPlanTests(unittest.TestCase):
    def test_new_database_uses_current_schema_without_replaying_archive(self):
        self.assertEqual([p.name for p in migrations.migration_plan(set())],
                         ['schema.sql', 'require_source_name.sql'])

    def test_initialized_database_never_replays_numbered_history(self):
        self.assertEqual([p.name for p in migrations.migration_plan({'schema.sql'})],
                         ['require_source_name.sql'])
        self.assertEqual(migrations.migration_plan({'schema.sql', 'require_source_name.sql'}), [])

    def test_existing_database_keeps_recorded_ids_and_only_applies_missing_changes(self):
        applied = {p.name for p in migrations.ARCHIVE.glob('*.sql')}
        self.assertEqual([p.name for p in migrations.migration_plan(applied)], ['require_source_name.sql'])
        self.assertEqual(migrations.migration_plan(applied | {'require_source_name.sql'}), [])

    def test_unknown_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unrecognized migration history'):
            migrations.migration_plan({'unknown.sql'})

    def test_failed_commit_never_reports_success(self):
        connection = MagicMock()
        connection.__exit__.side_effect = psycopg.OperationalError('commit failed')
        output = StringIO()
        with patch.object(migrations.psycopg, 'connect', return_value=connection), \
             patch.object(migrations, 'apply_migrations', return_value=['schema.sql']), redirect_stdout(output):
            with self.assertRaises(psycopg.OperationalError):
                migrations.main()
        self.assertEqual(output.getvalue(), '')


@unittest.skipUnless(os.environ.get('RUN_DATABASE_TESTS') == '1', 'Opt-in PostgreSQL integration tests')
class MigrationDatabaseTests(unittest.TestCase):
    """Replay SQL away from the application database and roll back each fixture."""

    @classmethod
    def setUpClass(cls):
        cls.database = 'codex_migrations_' + uuid4().hex
        cls.admin_url = settings.database_url.get_secret_value()
        with psycopg.connect(cls.admin_url, autocommit=True, connect_timeout=5) as admin:
            admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(cls.database)))
        cls.addClassCleanup(cls.remove_test_database)
        cls.test_url = make_conninfo(cls.admin_url, dbname=cls.database)

    @classmethod
    def remove_test_database(cls):
        """Drop only the uniquely named disposable database created by this class."""
        assert cls.database.startswith('codex_migrations_') and len(cls.database) == 49
        with psycopg.connect(cls.admin_url, autocommit=True, connect_timeout=5) as admin:
            assert admin.execute('SELECT current_database()').fetchone()[0] != cls.database
            admin.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(cls.database)))

    def setUp(self):
        self.connection = psycopg.connect(self.test_url, connect_timeout=5)
        self.addCleanup(self.connection.close)
        self.addCleanup(self.connection.rollback)

    def replay_history(self, *, initial_only=False):
        paths = sorted(migrations.ARCHIVE.glob('*.sql'))
        for path in paths[:1] if initial_only else paths:
            self.connection.execute(path.read_text(encoding='utf-8'), prepare=False)
            self.connection.execute('INSERT INTO schema_migrations(version) VALUES (%s)', (path.name,))

    def schema_contract(self):
        return {
            'columns': self.connection.execute(
                "SELECT table_name,column_name,data_type,udt_name,is_nullable,column_default,"
                "is_identity,is_generated,generation_expression FROM information_schema.columns "
                "WHERE table_schema='public' ORDER BY table_name,column_name").fetchall(),
            'constraints': self.connection.execute(
                "SELECT c.relname,p.conname,pg_get_constraintdef(p.oid) FROM pg_constraint p "
                "JOIN pg_class c ON c.oid=p.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname='public' ORDER BY c.relname,p.conname").fetchall(),
            'indexes': self.connection.execute(
                "SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY indexname").fetchall(),
            'triggers': self.connection.execute(
                "SELECT tgname,pg_get_triggerdef(oid) FROM pg_trigger WHERE NOT tgisinternal "
                "AND tgrelid='products'::regclass ORDER BY tgname").fetchall(),
            'revision_function': self.connection.execute(
                "SELECT pg_get_functiondef(oid) FROM pg_proc WHERE proname='bump_catalog_revision' "
                "AND pronamespace='public'::regnamespace").fetchall(),
        }

    def test_fresh_schema_and_archived_upgrade_produce_the_same_contract(self):
        self.assertEqual(migrations.apply_migrations(self.connection), ['schema.sql', 'require_source_name.sql'])
        self.connection.execute('INSERT INTO products(id,source_name,raw_source) VALUES (%s,%s,%s)',
                                ('fixture', 'authored', Jsonb({'ground_truth_category': 'Apparel & Accessories'})))
        self.assertEqual(self.connection.execute('SELECT revision FROM catalog_revision').fetchone()[0], 1)
        fresh = self.schema_contract()
        self.assertEqual(migrations.apply_migrations(self.connection), [])
        self.connection.rollback()
        self.replay_history()
        self.assertEqual(migrations.apply_migrations(self.connection), ['require_source_name.sql'])
        self.assertEqual(self.schema_contract(), fresh)

    def test_existing_products_vectors_sessions_and_history_are_preserved(self):
        self.replay_history()
        self.connection.execute(
            'INSERT INTO products(id,title,source_name,raw_source) VALUES (%s,%s,%s,%s)',
            ('fixture', 'Fixture Shirt', 'authored', Jsonb({'ground_truth_category': 'Apparel & Accessories'})))
        self.connection.execute("INSERT INTO product_embeddings(product_id,modality,model_name,embedding) "
                                "VALUES ('fixture','text','authored','[1,0]')")
        self.connection.execute("INSERT INTO agent_sessions(thread_id,state) VALUES ('fixture','{\"cart\": []}')")
        queries = ['SELECT row_to_json(p) FROM products p',
                   'SELECT embedding::text,model_name FROM product_embeddings',
                   'SELECT thread_id,state FROM agent_sessions',
                   'SELECT version,applied_at FROM schema_migrations ORDER BY version']
        before = [self.connection.execute(query).fetchall() for query in queries]
        self.assertEqual(migrations.apply_migrations(self.connection), ['require_source_name.sql'])
        for query, expected in zip(queries[:3], before[:3]):
            self.assertEqual(self.connection.execute(query).fetchall(), expected)
        self.assertTrue(set(before[3]).issubset(set(self.connection.execute(queries[3]).fetchall())))
        self.assertIsNone(self.connection.execute(
            "SELECT column_default FROM information_schema.columns WHERE table_name='products' "
            "AND column_name='source_name'").fetchone()[0])
        with self.assertRaises(psycopg.errors.NotNullViolation), self.connection.transaction():
            self.connection.execute("INSERT INTO products(id,raw_source) VALUES ('missing-source',%s)",
                                    (Jsonb({'ground_truth_category': 'Apparel & Accessories'}),))

    def test_partial_database_upgrade_preserves_eligible_products(self):
        self.replay_history(initial_only=True)
        self.connection.execute('INSERT INTO products(id,source_name,raw_source) VALUES (%s,%s,%s)',
                                ('fixture', 'authored', Jsonb({'ground_truth_category': 'Apparel & Accessories'})))
        executed = migrations.apply_migrations(self.connection)
        self.assertNotIn('schema.sql', executed)
        self.assertEqual(executed[-1], 'require_source_name.sql')
        self.assertEqual(self.connection.execute('SELECT id,source_name FROM products').fetchall(),
                         [('fixture', 'authored')])

    def test_invalid_taxonomy_stops_upgrade_without_deleting_records(self):
        self.replay_history(initial_only=True)
        for category in (None, 'Home & Garden'):
            with self.subTest(category=category), self.connection.transaction(force_rollback=True):
                self.connection.execute('INSERT INTO products(id,raw_source) VALUES (%s,%s)',
                                        ('fixture', Jsonb({'ground_truth_category': category})))
                with self.assertRaisesRegex(ValueError, 'no products were deleted'):
                    migrations.apply_migrations(self.connection)
                self.assertEqual(self.connection.execute('SELECT count(*) FROM products').fetchone()[0], 1)
                self.assertEqual(self.connection.execute('SELECT version FROM schema_migrations').fetchall(),
                                 [('001_initial_schema.sql',)])

    def test_database_without_history_is_not_overwritten(self):
        self.connection.execute('CREATE TABLE products(id text)')
        self.connection.execute("INSERT INTO products VALUES ('fixture')")
        with self.assertRaisesRegex(ValueError, 'without migration history'):
            migrations.apply_migrations(self.connection)
        self.assertEqual(self.connection.execute('SELECT id FROM products').fetchall(), [('fixture',)])

    def test_migration_lock_lasts_until_the_transaction_ends(self):
        migrations.apply_migrations(self.connection)
        with psycopg.connect(self.test_url, autocommit=True, connect_timeout=5) as other:
            self.assertFalse(other.execute('SELECT pg_try_advisory_xact_lock(%s)',
                                           (migrations.MIGRATION_LOCK,)).fetchone()[0])
            self.connection.rollback()
            self.assertTrue(other.execute('SELECT pg_try_advisory_xact_lock(%s)',
                                          (migrations.MIGRATION_LOCK,)).fetchone()[0])
