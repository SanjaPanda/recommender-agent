"""Behavioral regressions from the repository audit; no model downloads."""
from contextlib import contextmanager
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4
from unittest.mock import MagicMock, patch

from pgvector import Vector
from agent.models import Plan, SearchIntent
from agent.refinements import color_refinement
from agent.relevance import is_relevant_product
from agent.search_intent import extract_search_intent
from agent.product_evidence import listing_checks
from agent.llm import ModelError
from agent.workflow import SessionBusyError, run_recommendation_turn, sanitize_apparel_products
from rag import embed_text
from rag.retrieval import _search_embedding, retrieve_products
from scripts import expand_catalog
import psycopg
from psycopg.types.json import Jsonb
from agent.config import settings


class AuditRegressionTests(unittest.TestCase):
    def test_color_override_and_clear_remove_old_query_and_preference_colors(self):
        previous = Plan(intent='recommend', query='blue dress', preferences=[
            {'requirement': 'not black'}, {'requirement': 'soft cotton'}])
        product = {'id': 'dress', 'title': 'Black Cotton Dress', 'description': 'Soft cotton',
                   'text_similarity': .9}
        for followup in ('black', 'any color'):
            updated = color_refinement(previous, followup)
            self.assertEqual(updated.query, 'dress')
            self.assertEqual([p.requirement for p in updated.preferences], ['soft cotton'])
            self.assertTrue(is_relevant_product(product, updated.query,
                            required_title_color=updated.filters.title_color, product_phrase='dress'))
        self.assertEqual(previous.query, 'blue dress')

    def test_color_override_reextracts_saved_interpretation(self):
        from agent.workflow import _retrieve
        plan = color_refinement(Plan(intent='recommend', query='blue dress'), 'black')
        with patch('agent.workflow.extract_search_intent', return_value=SearchIntent(product_phrase='dress')) as extract, \
             patch('agent.workflow.retrieve_products', return_value=[{'id': 'd', 'title': 'Black Dress'}]):
            result = _retrieve({'plan': plan.model_dump(), 'intent_query': 'blue dress',
                                'search_intent': SearchIntent(product_phrase='dress').model_dump()})
        extract.assert_called_once_with('dress')
        self.assertEqual(result['result_kind'], 'matches')

    def test_truncated_compounds_and_negated_products_fail_closed(self):
        for query, phrase in [('blue bow tie', 'tie'), ('silver ear cuff', 'cuff'),
                              ('not wool', 'wool'), ('dress without shoes', 'shoes')]:
            with self.subTest(query=query), patch('agent.search_intent.complete',
                    return_value=SearchIntent(product_phrase=phrase)):
                with self.assertRaises(ModelError):
                    extract_search_intent(query)

    def test_incidental_product_word_is_neither_match_nor_tick(self):
        for title in ('Blue Dress Shoes', 'Dress Shoe Accessories'):
            self.assertFalse(is_relevant_product({'title': title, 'text_similarity': .99},
                                                'blue dress', product_phrase='dress'))
            self.assertEqual(listing_checks(title, {}, 'dress', {}, 'dress')[-1]['status'], 'unknown')
        self.assertTrue(is_relevant_product({'title': 'Blue Dress, Size 8'}, 'blue dress', product_phrase='dress'))

    def test_exact_model_tags_select_the_correct_digest(self):
        metadata = {'models': [{'name': 'nomic-embed-text:other', 'digest': 'other'},
                               {'name': 'nomic-embed-text:latest', 'digest': 'latest'}]}
        for configured, expected in [('nomic-embed-text', 'latest'),
                                     ('nomic-embed-text:latest', 'latest'),
                                     ('nomic-embed-text:other', 'other')]:
            with self.subTest(configured=configured), patch.object(embed_text, 'MODEL', configured), \
                 patch.object(embed_text.urllib.request, 'urlopen', return_value=BytesIO(json.dumps(metadata).encode())):
                self.assertEqual(embed_text.model_digest(), expected)

    def test_vector_query_requires_revision_dimensions_and_apparel_scope(self):
        connection = MagicMock()
        _search_embedding(connection, vector=Vector([1., 0.]), modality='text', model='model',
                          model_version='pinned', category_clause=' AND p.category=%s',
                          category_params=['Apparel & Accessories'], excluded_ids=[], limit=5)
        sql, values = connection.execute.call_args.args
        self.assertIn('e.model_version=%s', sql)
        self.assertIn('e.dimensions=%s', sql)
        self.assertEqual(values[3:6], ('pinned', 2, 'Apparel & Accessories'))

    def test_changed_encoder_during_query_fails_before_database_search(self):
        with patch('rag.retrieval.text_model_version', side_effect=['old', 'new']), \
             patch('rag.retrieval.embed_texts', return_value=[[1., 0.]]), \
             patch('rag.retrieval.psycopg.connect') as connect:
            with self.assertRaisesRegex(RuntimeError, 'changed during inference'):
                retrieve_products(query_context='dress')
        connect.assert_not_called()

    def test_mixed_example_revisions_are_rejected(self):
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [
            ('a', [1., 0.], 'Dress', 'old'), ('b', [1., 0.], 'Dress', 'new')]
        with patch('rag.retrieval.psycopg.connect') as connect, patch('rag.retrieval.register_vector'):
            connect.return_value.__enter__.return_value = connection
            with self.assertRaisesRegex(ValueError, 'incompatible CLIP revisions'):
                retrieve_products(selected_product_ids=['a', 'b'])

    def test_changed_snapshot_clears_recommendations_but_refreshes_cart(self):
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [
            ('dress', 'New Dress', 'New description', 'Brand', 'Apparel & Accessories',
             {'local_path': 'new.jpg'}, False, ['New feature'], 'new-source', 'new-image')]
        old = {'id': 'dress', 'title': 'Old Dress', 'source_record_sha256': 'old-source',
               'image_sha256': 'old-image', 'text_similarity': .9}
        result = sanitize_apparel_products(connection, {'recommendations': [old], 'cart': [old],
                                                      'selected_product_ids': ['dress']})
        self.assertEqual(result['recommendations'], [])
        self.assertEqual(result['selected_product_ids'], [])
        self.assertEqual(result['cart'][0]['title'], 'New Dress')
        self.assertIsNone(result['cart'][0]['text_similarity'])
        self.assertEqual(result['status'], 'catalog_changed')

    def test_serialization_refreshes_listing_image_and_clears_obsolete_scores(self):
        from agent.api import _serialize_products
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [
            ('dress', 'New description', ['New feature'], {'average_rating': 4.5}, {},
             'New Dress', 'Brand', 'Apparel & Accessories', {'local_path': 'new.jpg'}, False, 'new', 'new-image')]
        with patch('agent.api.psycopg.connect') as connect, patch('agent.api._find_image_path', return_value=Path('new.jpg')) as image:
            connect.return_value.__enter__.return_value = connection
            products = _serialize_products([{'id': 'dress', 'title': 'Old Dress',
                                            'source_record_sha256': 'old', 'text_similarity': .9}])
        self.assertEqual(products[0]['title'], 'New Dress')
        self.assertEqual(products[0]['listing']['description'], 'New description')
        self.assertIsNone(products[0]['text_similarity'])
        image.assert_called_once_with({'local_path': 'new.jpg'})

    def test_inference_runs_outside_transactions_and_lock_releases_on_failure(self):
        connection = MagicMock()
        depth = 0
        @contextmanager
        def transaction():
            nonlocal depth
            depth += 1
            try:
                yield
            finally:
                depth -= 1
        connection.transaction.side_effect = transaction
        def execute(sql, params=None):
            if 'pg_try_advisory_lock' in sql:
                return MagicMock(fetchone=lambda: (True,))
            if 'SELECT state' in sql:
                return MagicMock(fetchone=lambda: ({},))
            if 'SELECT revision' in sql:
                return MagicMock(fetchone=lambda: (1,))
            return MagicMock(rowcount=1)
        connection.execute.side_effect = execute
        graph = MagicMock()
        def invoke(*args, **kwargs):
            self.assertEqual(depth, 0)
            raise RuntimeError('inference failed')
        graph.invoke.side_effect = invoke
        with patch('agent.workflow.psycopg.connect') as connect:
            connect.return_value.__enter__.return_value = connection
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                run_recommendation_turn(graph, thread_id='test', turn={})
        self.assertTrue(any('pg_advisory_unlock' in c.args[0] for c in connection.execute.call_args_list))
        self.assertTrue(connect.call_args.kwargs['autocommit'])

    def test_busy_session_is_rejected_before_graph_and_api_returns_409(self):
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = (False,)
        graph = MagicMock()
        with patch('agent.workflow.psycopg.connect') as connect:
            connect.return_value.__enter__.return_value = connection
            with self.assertRaises(SessionBusyError):
                run_recommendation_turn(graph, thread_id='busy', turn={})
        graph.invoke.assert_not_called()
        from agent.api import discover, DiscoverRequest
        from fastapi import HTTPException
        with patch('agent.api.run_recommendation_turn', side_effect=SessionBusyError('Busy')):
            with self.assertRaises(HTTPException) as error:
                discover(DiscoverRequest(query='dress'))
        self.assertEqual(error.exception.status_code, 409)

    def test_catalog_change_or_external_session_edit_prevents_snapshot_save(self):
        for catalog_changes in (False, True):
            with self.subTest(catalog_changes=catalog_changes):
                connection = MagicMock()
                revisions = iter((1, 2 if catalog_changes else 1))
                def execute(sql, params=None):
                    if 'pg_try_advisory_lock' in sql:
                        return MagicMock(fetchone=lambda: (True,))
                    if 'SELECT state' in sql:
                        return MagicMock(fetchone=lambda: ({},))
                    if 'SELECT revision' in sql:
                        return MagicMock(fetchone=lambda: (next(revisions),))
                    return MagicMock(rowcount=0)
                connection.execute.side_effect = execute
                graph = MagicMock()
                graph.invoke.return_value = {'message': 'Must not persist'}
                with patch('agent.workflow.psycopg.connect') as connect:
                    connect.return_value.__enter__.return_value = connection
                    with self.assertRaises(SessionBusyError):
                        run_recommendation_turn(graph, thread_id='conflicting', turn={})
                updates = [c for c in connection.execute.call_args_list if c.args[0].startswith('UPDATE')]
                self.assertEqual(len(updates), 0 if catalog_changes else 1)
                if updates:
                    self.assertIn('AND state=%s', updates[0].args[0])
                self.assertTrue(any('pg_advisory_unlock' in c.args[0] for c in connection.execute.call_args_list))

    def test_catalog_promotion_recovers_after_committed_import_and_partial_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / '.data/catalog'
            stage = root / '.data/expansions/run'
            for path in (active / 'images', stage / 'images'):
                path.mkdir(parents=True)
            for name in expand_catalog.PROMOTED_FILES:
                (active / name).write_text('old', encoding='utf-8')
                (stage / name).write_text(json.dumps({'accepted': 2}) if name == 'pilot-report.json' else 'new', encoding='utf-8')
            original_replace = Path.replace
            def interrupted(path, target):
                if path.name == 'manifest.json.next':
                    raise OSError('publication interrupted')
                return original_replace(path, target)
            connection = MagicMock()
            imported = {'accepted': 2, 'rejected': {}}
            with patch.object(expand_catalog, 'ROOT', root), \
                 patch.object(expand_catalog.psycopg, 'connect') as connect, \
                 patch.object(expand_catalog, 'ingest', return_value=imported) as ingest:
                connect.return_value.__enter__.return_value = connection
                with patch.object(Path, 'replace', interrupted):
                    with self.assertRaises(OSError):
                        expand_catalog.promote_catalog(active, stage)
                self.assertTrue((active / '.promotion.json').exists())
                self.assertEqual((active / 'records.jsonl').read_text(), 'new')
                self.assertEqual((active / 'manifest.json').read_text(), 'old')
                self.assertEqual(expand_catalog.promote_catalog(active), imported)
                self.assertEqual(ingest.call_count, 2)
            self.assertFalse((active / '.promotion.json').exists())
            for name in expand_catalog.PROMOTED_FILES:
                self.assertEqual((active / name).read_bytes(), (stage / name).read_bytes())

    def test_changed_recovery_stage_is_rejected_before_import(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / '.data/catalog'
            stage = root / '.data/expansions/run'
            active.mkdir(parents=True)
            stage.mkdir(parents=True)
            for name in expand_catalog.PROMOTED_FILES:
                (stage / name).write_text('{}', encoding='utf-8')
            (active / '.promotion.json').write_text(json.dumps({
                'stage': '.data/expansions/run', 'hashes': {name: 'old' for name in expand_catalog.PROMOTED_FILES}}))
            with patch.object(expand_catalog, 'ROOT', root), \
                 patch.object(expand_catalog.psycopg, 'connect'), patch.object(expand_catalog, 'ingest') as ingest:
                with self.assertRaisesRegex(ValueError, 'metadata changed'):
                    expand_catalog.promote_catalog(active)
            ingest.assert_not_called()


@unittest.skipUnless(os.environ.get('RUN_DATABASE_TESTS') == '1', 'Opt-in PostgreSQL integration tests')
class SessionConcurrencyDatabaseTests(unittest.TestCase):
    """Use isolated session rows to verify PostgreSQL lock/transaction behavior."""

    def test_populated_snapshot_refresh_uses_real_provenance_columns(self):
        identifier = 'audit-product-' + uuid4().hex
        with psycopg.connect(settings.database_url.get_secret_value(), autocommit=True) as connection:
            with connection.transaction(force_rollback=True):
                connection.execute(
                    'INSERT INTO products(id,title,raw_source,source_name,source_record_sha256,image_sha256) '
                    'VALUES (%s,%s,%s,%s,%s,%s)',
                    (identifier, 'Authored Dress', Jsonb({'ground_truth_category': 'Apparel & Accessories'}),
                     'authored-test', 'a'*64, 'b'*64))
                card = {'id': identifier, 'title': 'Authored Dress', 'text_similarity': .9}
                stamped = sanitize_apparel_products(connection, {'recommendations': [card], 'cart': [card]},
                                                     invalidate_changed=False)
                self.assertEqual(stamped['recommendations'][0]['image_sha256'], 'b'*64)
                unchanged = sanitize_apparel_products(connection, stamped)
                self.assertEqual(unchanged['recommendations'][0]['text_similarity'], .9)
                connection.execute('UPDATE products SET title=%s,source_record_sha256=%s WHERE id=%s',
                                   ('Changed Dress', 'c'*64, identifier))
                refreshed = sanitize_apparel_products(connection, stamped)
                self.assertEqual(refreshed['recommendations'], [])
                self.assertEqual(refreshed['cart'][0]['title'], 'Changed Dress')
                self.assertIsNone(refreshed['cart'][0]['text_similarity'])

    def test_same_thread_is_busy_while_inference_has_no_open_transaction(self):
        identifier = 'audit-session-' + uuid4().hex
        application = 'session-test-' + uuid4().hex
        original_connect = psycopg.connect
        def connect(*args, **kwargs):
            return original_connect(*args, **kwargs, application_name=application)
        with original_connect(settings.database_url.get_secret_value(), autocommit=True) as monitor:
            graph = MagicMock()
            def invoke(state, **kwargs):
                with self.assertRaises(SessionBusyError):
                    run_recommendation_turn(MagicMock(), thread_id=identifier, turn={})
                # A just-closed competing connection can briefly remain in
                # pg_stat_activity. Inspect the actual advisory-lock owner.
                rows = monitor.execute("SELECT state,xact_start FROM pg_stat_activity a WHERE application_name=%s "
                                       "AND EXISTS (SELECT 1 FROM pg_locks l WHERE l.pid=a.pid AND l.locktype='advisory')",
                                       (application,)).fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0], ('idle', None))
                return {**state, 'message': 'Saved'}
            graph.invoke.side_effect = invoke
            try:
                with patch('agent.workflow.psycopg.connect', side_effect=connect):
                    result = run_recommendation_turn(graph, thread_id=identifier, turn={})
                self.assertEqual(result['message'], 'Saved')
                saved = monitor.execute('SELECT state FROM agent_sessions WHERE thread_id=%s', (identifier,)).fetchone()[0]
                self.assertEqual(saved['message'], 'Saved')
                # Normal completion releases the lock, permitting another turn.
                with patch('agent.workflow.psycopg.connect', side_effect=connect):
                    run_recommendation_turn(graph, thread_id=identifier, turn={})
            finally:
                monitor.execute('DELETE FROM agent_sessions WHERE thread_id=%s', (identifier,))


if __name__ == '__main__':
    unittest.main()
