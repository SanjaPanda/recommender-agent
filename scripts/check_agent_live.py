"""Exercise the real API, local model, and database using an isolated test session.

Run with .venv/Scripts/python.exe scripts/check_agent_live.py.
Starts its own loopback API process, restarts it to check durability, then removes
only the session created by this run. Does not modify products or embeddings.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb
from agent.config import settings
from agent.models import Plan, Filters
from agent.workflow import sanitize_apparel_products
from rag.retrieval import APPAREL_CATEGORY, retrieve_products

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Exercise the real API/database lifecycle and optionally the local router.

    Run ``.venv/Scripts/python.exe scripts/check_agent_live.py --infrastructure-only``
    to validate persistence and UI payloads without inference.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infrastructure-only', action='store_true',
        help='Use a catalogue-backed session fixture to check API/database behavior without model inference.')
    args = parser.parse_args()
    report = {'checks': [], 'status': 'running',
              'mode': 'infrastructure' if args.infrastructure_only else 'full',
              'model': settings.agent_model,
              'num_gpu': settings.agent_num_gpu if settings.agent_num_gpu is not None else 'automatic',
              'router_timeout_seconds': settings.agent_router_timeout_seconds}
    thread = 'acceptance-' + str(uuid4())
    process = None
    artifacts = ROOT / '.codex-work'
    artifacts.mkdir(exist_ok=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    def check(name, passed):
        """Record a named assertion or stop the live check on failure.

        Args:
            name: Description included in the report and console output.
            passed: Condition that must be true for the check to succeed.

        Raises:
            AssertionError: If ``passed`` is false.
        """
        if not passed:
            raise AssertionError(name)
        report['checks'].append(name)
        print('PASS:', name, flush=True)

    def post(path, body):
        """POST JSON to the temporary API and record response timing.

        Args:
            path: API path relative to the temporary loopback server.
            body: JSON-serializable request payload.

        Returns:
            The decoded JSON response object.

        Raises:
            urllib.error.URLError: If the request fails or times out.
            json.JSONDecodeError: If the response is not valid JSON.
        """
        request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        report.setdefault('turns', []).append({'path': path, 'seconds': round(time.monotonic() - started, 2),
            **{k: result.get(k) for k in ('status', 'message', 'filters', 'preferences', 'query_context')},
            'product_count': len(result.get('products', []))})
        print('TURN:', json.dumps(report['turns'][-1]), flush=True)
        return result

    def start(log):
        """Start the test API and poll its health endpoint until ready.

        Args:
            log: Open file-like object used for child stdout and stderr.

        Returns:
            The running child process.

        Raises:
            RuntimeError: If the process exits early or fails to become ready
                within the bounded polling period. Startup failures terminate it.
        """
        child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'agent.api:app',
                                  '--host', '127.0.0.1', '--port', str(port)],
                                 cwd=ROOT, stdout=log, stderr=log)
        try:
            for _ in range(60):
                if child.poll() is not None:
                    raise RuntimeError('Test API exited; see live-agent-api.log')
                try:
                    with urllib.request.urlopen(base + '/api/health', timeout=1):
                        return child
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    time.sleep(0.5)
            raise RuntimeError('Test API did not become ready')
        except BaseException:
            child.terminate()
            child.wait(timeout=10)
            raise

    try:
        with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
            report['products'] = connection.execute('SELECT count(*) FROM products').fetchone()[0]
            roots = [row[0] for row in connection.execute(
                "SELECT DISTINCT split_part(raw_source->>'ground_truth_category',' > ',1) "
                "FROM products ORDER BY 1").fetchall()]
            vectors_without_catalog_items = connection.execute(
                'SELECT count(*) FROM product_embeddings e LEFT JOIN products p ON p.id=e.product_id '
                'WHERE p.id IS NULL').fetchone()[0]
            products_missing_core_vectors = connection.execute('''
                SELECT count(*) FROM products p WHERE NOT EXISTS (
                    SELECT 1 FROM product_embeddings e WHERE e.product_id=p.id
                    AND e.modality='text' AND e.status='ready') OR NOT EXISTS (
                    SELECT 1 FROM product_embeddings e WHERE e.product_id=p.id
                    AND e.modality='image' AND e.status='ready')
            ''').fetchone()[0]
            report['embeddings'] = connection.execute(
                'SELECT modality,status,count(*) FROM product_embeddings GROUP BY 1,2 ORDER BY 1,2').fetchall()
            brands = [row[0] for row in connection.execute(
                "SELECT brand FROM products WHERE NULLIF(brand,'') IS NOT NULL "
                'GROUP BY brand ORDER BY count(*) DESC,brand LIMIT 2').fetchall()]
            seed = connection.execute(
                "SELECT p.id FROM products p JOIN product_embeddings e ON e.product_id=p.id "
                "WHERE e.modality='image' AND e.status='ready' "
                "AND starts_with(p.raw_source->>'ground_truth_category', 'Apparel & Accessories > ') "
                'ORDER BY p.id LIMIT 1').fetchone()
        if report['products'] == 0:
            raise RuntimeError('Live checks require an imported catalog with compatible text and image vectors.')
        check('Catalog is populated for live checks', report['products'] > 0)
        check('Catalog contains only Apparel & Accessories rows', roots == [APPAREL_CATEGORY])
        check('Every apparel product has text and CLIP vectors', products_missing_core_vectors == 0)
        check('No embeddings are orphaned from products', vectors_without_catalog_items == 0)
        check('Catalogue contains two brands for replacement checks', len(brands) == 2)
        check('Catalogue contains an indexed apparel seed', seed is not None)
        similar = retrieve_products(selected_product_ids=[seed[0]],
            filters={'category_prefixes': ['Apparel & Accessories']}, limit=5)
        check('Image retrieval returns five unique products and excludes seed', len(similar) == 5
              and len({p['id'] for p in similar}) == 5 and seed[0] not in {p['id'] for p in similar})
        with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
            categories = connection.execute(
                "SELECT raw_source->>'ground_truth_category' FROM products WHERE id=ANY(%s)",
                ([p['id'] for p in similar],)).fetchall()
        check('Image retrieval preserves apparel scope', all(
            category.startswith('Apparel & Accessories > ') for (category,) in categories))
        log_name = 'live-agent-infrastructure-api.log' if args.infrastructure_only else 'live-agent-api.log'
        with (artifacts / log_name).open('w', encoding='utf-8') as log:
            process = start(log)
            state = post('/api/session', {'thread_id': thread})
            check('New session shows categories', state['status'] == 'categories' and bool(state['categories']))
            if args.infrastructure_only:
                fixture = {'plan': Plan(intent='recommend', filters=Filters(brand=brands[0]),
                                       preferences=[{'requirement': 'lightweight', 'strength': 'soft'}]).model_dump(),
                           'recommendations': retrieve_products(filters={'brand': brands[0]}, limit=5),
                           'cart': [], 'selected_product_ids': [], 'status': 'ok'}
                with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
                    fixture = sanitize_apparel_products(connection, fixture, invalidate_changed=False)
                    connection.execute('UPDATE agent_sessions SET state=%s WHERE thread_id=%s',
                                       (Jsonb(fixture), thread))
                state = post('/api/session', {'thread_id': thread})
            else:
                state = post('/api/discover', {'thread_id': thread,
                    'query': f'Show products from the brand {brands[0]}. No other filters or preferences.'})
            check('Session returns bounded catalogue cards', bool(state['products']) and len(state['products']) <= 5)
            check('Brand filter is enforced', all(p['brand'].casefold() == brands[0].casefold() for p in state['products']))
            product = state['products'][0]['id']
            state = post('/api/cart', {'thread_id': thread, 'action': 'cart_add', 'product_ids': [product]})
            check('Explicit add persists cart item', [p['id'] for p in state['cart']] == [product])
            check('Cart item leaves recommendations and selections',
                  product not in {p['id'] for p in state['products']} and product not in state['selected_product_ids'])
            saved = state
            process.terminate()
            process.wait(timeout=10)
            process = start(log)
            state = post('/api/session', {'thread_id': thread})
            check('API restart preserves cart, filters, preferences and cards', all(
                state[key] == saved[key] for key in ('cart', 'filters', 'preferences', 'products')))
            if not args.infrastructure_only:
                state = post('/api/discover', {'thread_id': thread,
                    'query': f'Actually, replace the brand with {brands[1]}. No other filters or preferences.'})
                check('Follow-up replaces the old brand', state['filters']['brand'].casefold() == brands[1].casefold()
                      and bool(state['products']) and all(p['brand'].casefold() == brands[1].casefold() for p in state['products']))
            state = post('/api/discover', {'thread_id': thread, 'query': '#buynew'})
            check('Reset clears shopping context and keeps cart', state['status'] == 'categories'
                  and not state['products'] and not state['preferences'] and not state['selected_product_ids']
                  and state['filters']['brand'] is None and [p['id'] for p in state['cart']] == [product])
            state = post('/api/cart', {'thread_id': thread, 'action': 'cart_remove', 'product_ids': [product]})
            check('Explicit removal empties cart', state['cart'] == [])
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        if isinstance(exc, urllib.error.HTTPError):
            report['http_status'] = exc.code
            report['api_error'] = exc.read().decode('utf-8', errors='replace')[:2000]
            print('API error:', report['api_error'], flush=True)
        if isinstance(exc, AssertionError):
            report['failed_check'] = str(exc)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        try:
            with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
                connection.execute('DELETE FROM agent_sessions WHERE thread_id=%s', (thread,))
        except Exception:
            report['cleanup'] = 'Could not remove test session: ' + thread
        report_name = 'live-agent-infrastructure-report.json' if args.infrastructure_only else 'live-agent-report.json'
        (artifacts / report_name).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
