"""Inspect local schema readiness without loading models or changing data.

Run: ``.venv/Scripts/python.exe scripts/db/inspect_database.py``.
Uses the existing validated DATABASE_URL and never prints credentials.
"""
from __future__ import annotations

import json

import psycopg

from agent.config import settings


def main() -> None:
    """Print database version, vector extension, migration history and row counts.

    Raises:
        psycopg.Error: If connectivity or an expected schema object is missing.
    """
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        report = {
            "database": connection.execute("SELECT current_database()").fetchone()[0],
            "postgresql": connection.execute("SHOW server_version").fetchone()[0],
            "pgvector": connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0],
            "migrations": [row[0] for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )],
            "rows": {},
        }
        for table in ("dataset_sources", "products", "product_embeddings", "agent_sessions",
                      "product_feedback", "recommendation_sessions"):
            # Identifiers are a fixed application-owned allowlist, never user input.
            report["rows"][table] = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
