"""Apply ordered SQL migrations to the configured PostgreSQL database.

Run ``uv run python scripts/db/apply_migrations.py`` from the repository root.
Requires configured PostgreSQL; writes schema/migration records. See docs/SETUP.md."""
from __future__ import annotations

import psycopg

from agent.config import ROOT, settings

MIGRATIONS = ROOT / "migrations"


def main() -> None:
    """Apply each unapplied SQL migration in filename order."""
    with psycopg.connect(settings.database_url.get_secret_value()) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        for migration_path in sorted(MIGRATIONS.glob("*.sql")):
            version = migration_path.name
            applied = connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s", (version,)
            ).fetchone()
            if applied:
                print(f"skip {version} (already applied)")
                continue

            sql = migration_path.read_text(encoding="utf-8")
            with connection.transaction():
                # Let PostgreSQL parse statements, including dollar-quoted
                # function bodies containing their own semicolons.
                connection.execute(sql, prepare=False)
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
                )
            print(f"applied {version}")


if __name__ == "__main__":
    main()
