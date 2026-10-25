"""Initialize or upgrade the PostgreSQL/pgvector application schema.

Run ``uv run python scripts/db/apply_migrations.py`` from the repository root.
Reads DATABASE_URL, migrations/schema.sql and ordered changes; writes schema and
migration records in one locked transaction. See docs/database.md. Regression
checks: ``.venv/Scripts/python.exe -m unittest discover -s tests -v``.
"""
from __future__ import annotations

from pathlib import Path

import psycopg

from agent.config import ROOT, settings

MIGRATIONS = ROOT / 'migrations'
ARCHIVE = ROOT / 'archive' / 'migrations'
BASELINE = MIGRATIONS / 'schema.sql'
# Explicit application order allows descriptive filenames without numeric prefixes.
CHANGES = (MIGRATIONS / 'require_source_name.sql',)
MIGRATION_LOCK = 0x4D494752


def migration_plan(applied: set[str]) -> list[Path]:
    """Choose a fresh schema or an upgrade path without renaming recorded IDs.

    Args:
        applied: Existing schema_migrations.version values.

    Returns:
        Unapplied SQL paths in dependency order. New databases use schema.sql;
        databases with numbered history finish that chain from the archive.

    Raises:
        ValueError: If SQL files are unregistered or database history is unknown.
    """
    if set(MIGRATIONS.glob('*.sql')) != {BASELINE, *CHANGES}:
        raise ValueError('Register each SQL change in apply_migrations.CHANGES in application order.')
    archived = sorted(ARCHIVE.glob('*.sql'))
    if BASELINE.name in applied:
        candidates = CHANGES
    elif not applied:
        candidates = (BASELINE, *CHANGES)
    elif any(path.name in applied for path in archived):
        candidates = (*archived, *CHANGES)
    else:
        raise ValueError('Unrecognized migration history; inspect schema_migrations before upgrading.')
    return [path for path in candidates if path.name not in applied]


def apply_migrations(connection: psycopg.Connection) -> list[str]:
    """Apply the plan in the caller's transaction, rolling back on caller failure.

    Args:
        connection: PostgreSQL connection with a transaction owned by the caller.

    Returns:
        Executed filenames. They are durable only after the caller commits.

    Raises:
        ValueError: For unmanaged schemas, unknown history or unsafe taxonomy.
        psycopg.Error: For connection, SQL or constraint errors.

    Holds a transaction-scoped advisory lock through SQL and ledger writes.
    Existing products, embeddings, sessions and migration IDs are preserved.
    """
    connection.execute('SELECT pg_advisory_xact_lock(%s)', (MIGRATION_LOCK,))
    connection.execute('CREATE TABLE IF NOT EXISTS schema_migrations ('
                       'version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())')
    applied = {row[0] for row in connection.execute('SELECT version FROM schema_migrations')}
    plan = migration_plan(applied)
    has_products = connection.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables "
        "WHERE table_schema=current_schema() AND table_name='products')"
    ).fetchone()[0]
    if BASELINE in plan and has_products:
        raise ValueError('Products exist without migration history; inspect the database before initialization.')
    if has_products and any(path.name in {'006_apparel_catalog_scope.sql', '010_curated_catalog.sql'} for path in plan):
        connection.execute('LOCK TABLE products IN SHARE ROW EXCLUSIVE MODE')
        invalid = connection.execute(
            "SELECT count(*) FROM products WHERE raw_source->>'ground_truth_category' IS NULL OR NOT ("
            "raw_source->>'ground_truth_category'='Apparel & Accessories' OR "
            "starts_with(raw_source->>'ground_truth_category','Apparel & Accessories > '))"
        ).fetchone()[0]
        if invalid:
            raise ValueError(f'{invalid} products have missing or unsupported taxonomy. '
                             'Review those records before upgrading; no products were deleted.')
    for path in plan:
        connection.execute(path.read_text(encoding='utf-8'), prepare=False)
        connection.execute('INSERT INTO schema_migrations(version) VALUES (%s)', (path.name,))
    return [path.name for path in plan]


def main() -> None:
    """Commit the complete locked plan before reporting applied filenames."""
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
        executed = apply_migrations(connection)
    for version in executed:
        print(f'applied {version}')
    if not executed:
        print('Database schema is current.')


if __name__ == '__main__':
    main()
