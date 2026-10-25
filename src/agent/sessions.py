"""Load, normalize and save shopping sessions with PostgreSQL concurrency guards.

``run_recommendation_turn`` owns short load/save transactions and a connection
advisory lock. Model and retrieval work runs between those transactions.
Read-only resume uses ``sanitize_apparel_products`` without saving normalized JSON.
Run regression checks from the repository root with
``.venv/Scripts/python.exe -m unittest discover -s tests -v``.
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .config import settings
from .workflow import run_turn
from .conditions import UNSUPPORTED_CONDITION
from .product_evidence import load_products, refresh_product_facts
from rag import validate_category_prefixes


DURABLE_FIELDS = frozenset({
    'plan', 'messages', 'selected_product_ids', 'removed_filters',
    'recommendations', 'cart', 'status', 'message', 'result_kind',
    'search_intent', 'intent_query',
})


class SessionBusyError(RuntimeError):
    """Another turn owns the session, or its inputs changed before saving."""


def sanitize_apparel_products(connection: psycopg.Connection, state: dict[str, Any],
                             *, invalidate_changed: bool = True) -> dict[str, Any]:
    """Refresh apparel facts and hide recommendations from changed source records.

    The session row and its conversation history remain intact. Only product
    references outside apparel are omitted. Cart facts are refreshed. With
    ``invalidate_changed=True`` (the default), changed/unknown source hashes
    discard historical recommendations and scores. False stamps newly ranked
    cards after the caller checks that the catalog revision has not changed.

    Args:
        connection: Open PostgreSQL connection used for one apparel-scoped read.
        state: Saved or newly produced state; the input is not modified.
        invalidate_changed: Whether changed or missing snapshot hashes invalidate
            old recommendations. Defaults to True; False stamps new results.

    Returns:
        A new state mapping with current facts and eligible product references.

    Raises:
        psycopg.Error: If the current catalog cannot be read.
    """
    result = {key: value for key, value in state.items() if key not in {'uncertain', 'excluded_count'}}
    plan = state.get('plan')
    saved_condition = None
    if isinstance(plan, dict):
        result['query_context'] = plan.get('query', '')
        filters = dict(plan.get('filters', {}))
        saved_condition = filters.pop('is_secondhand', None)
        saved_categories = filters.get('category_prefixes', [])
        valid_categories = []
        for prefix in saved_categories:
            try:
                valid_categories.extend(validate_category_prefixes([prefix]))
            except ValueError:
                continue  # Preserve supported filters when resuming stored state.
        if valid_categories != saved_categories:
            filters['category_prefixes'] = valid_categories
            result.update(recommendations=[], selected_product_ids=[], search_intent=None,
                          intent_query='', result_kind='none', status='catalog_changed',
                          message='Saved category filters are unavailable. Search again to refresh recommendations.')
        result['plan'] = {key: value for key, value in {**plan, 'filters': filters}.items()
                          if key != 'product_ids'}
        if plan.get('intent') in {'cart_add', 'cart_remove', 'cart_show'}:
            result['plan']['intent'] = 'recommend'
        if saved_condition is not None:
            result.update(recommendations=[], selected_product_ids=[], search_intent=None,
                          intent_query='', result_kind='none', status='clarify',
                          message=UNSUPPORTED_CONDITION)
    state = result
    referenced = {
        product.get('id')
        for key in ('recommendations', 'cart')
        for product in state.get(key, [])
        if isinstance(product, dict) and isinstance(product.get('id'), str)
    }
    referenced.update(pid for pid in state.get('selected_product_ids', []) if isinstance(pid, str))
    if not referenced:
        result.update(recommendations=[], selected_product_ids=[], cart=[])
        return result
    current, _ = load_products(connection, list(referenced))
    changed = set()
    def refresh(product: dict[str, Any]) -> dict[str, Any]:
        """Refresh source facts and discard scores tied to another snapshot."""
        facts = current[product['id']]
        updated, stale = refresh_product_facts(product, facts, invalidate_changed=invalidate_changed)
        if stale:
            changed.add(product['id'])
        return updated

    refreshed = [refresh(product) for product in state.get('recommendations', [])
                 if isinstance(product, dict) and product.get('id') in current]
    result['recommendations'] = [product for product in refreshed if product['id'] not in changed]
    result['cart'] = [
        refresh(product) for product in state.get('cart', [])
        if isinstance(product, dict) and product.get('id') in current
    ]
    cart_ids = {product['id'] for product in result['cart']}
    result['recommendations'] = [p for p in result['recommendations'] if p['id'] not in cart_ids]
    result['selected_product_ids'] = [
        pid for pid in state.get('selected_product_ids', [])
        if pid in current and pid not in changed and pid not in cart_ids
    ]
    if changed and saved_condition is None:
        result.update(status='catalog_changed', message='The catalog changed. Search again to refresh recommendations.')
    return result


def run_recommendation_turn(*, thread_id: str, turn: dict[str, Any]) -> dict[str, Any]:
    """Run a turn with a nonblocking session advisory lock and short transactions.

    The connection is idle outside transactions while inference runs. A second
    turn for the same thread fails immediately; unrelated sessions can proceed.
    A compare-and-save rejects external snapshot changes. Persisted references are checked
    against the apparel catalog before invocation. Transient request, candidate,
    category and trace fields are omitted from the durable JSONB.

    Args:
        thread_id: Nonblank persistent session identifier, at most 128 characters.
        turn: Current request fields consumed by the router stage.

    Returns:
        The full turn state for formatting into the API/CLI response.

    Raises:
        ValueError: If ``thread_id`` is invalid or a turn validation rejects
            the requested action.
        psycopg.Error: If PostgreSQL cannot create, lock, read, or save the session.
        SessionBusyError: If another turn owns the session or its snapshot changed.
    """
    if not thread_id.strip() or len(thread_id) > 128:
        raise ValueError('Invalid session ID.')
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5, autocommit=True) as connection:
        lock_name = 'recommender-turn:' + thread_id
        acquired = connection.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0))', (lock_name,)).fetchone()[0]
        if not acquired:
            raise SessionBusyError('This session is processing another request. Retry after it finishes.')
        try:
            with connection.transaction():
                connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
                connection.execute("SET LOCAL lock_timeout = '5s'")
                revision = connection.execute('SELECT revision FROM catalog_revision WHERE singleton=true').fetchone()[0]
                connection.execute('INSERT INTO agent_sessions(thread_id) VALUES (%s) ON CONFLICT DO NOTHING', (thread_id,))
                snapshot = connection.execute('SELECT state FROM agent_sessions WHERE thread_id=%s', (thread_id,)).fetchone()[0]
                previous = sanitize_apparel_products(connection, snapshot)
            state = run_turn({**previous, 'turn': turn})
            # Stamp new cards with source identities before persisting them.
            with connection.transaction():
                connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
                connection.execute("SET LOCAL lock_timeout = '5s'")
                current_revision = connection.execute('SELECT revision FROM catalog_revision WHERE singleton=true').fetchone()[0]
                if current_revision != revision:
                    raise SessionBusyError('The catalog changed during this request. Retry your search.')
                state = sanitize_apparel_products(connection, state, invalidate_changed=False)
                durable = {k: v for k, v in state.items()
                           if k in DURABLE_FIELDS}
                saved = connection.execute(
                    'UPDATE agent_sessions SET state=%s,updated_at=now() WHERE thread_id=%s AND state=%s',
                    (Jsonb(durable), thread_id, Jsonb(snapshot)))
                if saved.rowcount != 1:
                    raise SessionBusyError('This session changed during the request. Reload and retry.')
        finally:
            connection.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))', (lock_name,))
    return state
