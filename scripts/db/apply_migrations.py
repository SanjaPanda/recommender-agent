"""Apply ordered SQL migrations to the configured PostgreSQL database."""
from __future__ import annotations

import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
MIGRATIONS = ROOT / "migrations"


def main() -> None:
    """Apply each unapplied SQL migration in filename order."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Set DATABASE_URL to a PostgreSQL database with pgvector available.")

    with psycopg.connect(database_url) as connection:
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
            statements = [part.strip() for part in re.split(r";\s*", sql) if part.strip()]
            with connection.transaction():
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
                )
            print(f"applied {version}")


if __name__ == "__main__":
    main()
