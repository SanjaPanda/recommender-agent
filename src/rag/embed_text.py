"""Generate and store Ollama text embeddings for catalogue products."""
from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request
import json
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pgvector import Vector

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one JSON request to the configured local Ollama endpoint."""
    request = urllib.request.Request(
        f"{OLLAMA_URL.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL}: {exc}") from exc


def model_digest() -> str | None:
    """Return the installed embedding model digest or explain how to install it."""
    request = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            tags = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not read Ollama model metadata: {exc}") from exc
    for model in tags.get("models", []):
        if model.get("name", "").split(":", 1)[0] == MODEL:
            return model.get("digest")
    raise SystemExit(f"Model {MODEL!r} is not installed. Run: ollama pull {MODEL}")


def document_text(title: str | None, description: str | None) -> str:
    """Format available product text using Nomic's document instruction prefix."""
    parts = [part.strip() for part in (title, description) if part and part.strip()]
    return "search_document: " + "\n".join(parts)


def main() -> None:
    """Embed catalogue product text and upsert model-versioned vectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum products to process")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Recompute all product embeddings")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.batch_size < 1 or args.batch_size > 32:
        parser.error("--batch-size must be between 1 and 32")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Set DATABASE_URL to the PostgreSQL database.")
    digest = model_digest()
    limit_sql = " LIMIT %s" if args.limit is not None else ""
    existing_filter = "" if args.force else """
        AND NOT EXISTS (
            SELECT 1 FROM product_embeddings e
            WHERE e.product_id = p.id AND e.modality = 'text'
              AND e.model_name = %s AND e.model_version = %s AND e.status = 'ready'
        )
    """
    select_params: list[Any] = []
    if not args.force:
        select_params.extend([MODEL, digest])
    if args.limit is not None:
        select_params.append(args.limit)
    query = """
        SELECT p.id, p.title, p.description
        FROM products p
        WHERE (NULLIF(trim(p.title), '') IS NOT NULL
               OR NULLIF(trim(p.description), '') IS NOT NULL)
    """ + existing_filter + " ORDER BY p.id" + limit_sql

    started = time.perf_counter()
    processed = 0
    dimensions: int | None = None
    with psycopg.connect(database_url, autocommit=True) as connection:
        register_vector(connection)
        products = connection.execute(query, select_params).fetchall()
        if not products:
            print(f"No products need embeddings for {MODEL} ({digest or 'unknown version'}).")
            return

        for offset in range(0, len(products), args.batch_size):
            batch = products[offset : offset + args.batch_size]
            texts = [document_text(title, description) for _, title, description in batch]
            result = post_json("/api/embed", {"model": MODEL, "input": texts})
            vectors = result.get("embeddings", [])
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(vectors)} vectors for {len(batch)} products."
                )
            for vector in vectors:
                if dimensions is None:
                    dimensions = len(vector)
                elif len(vector) != dimensions:
                    raise RuntimeError("Ollama returned inconsistent embedding dimensions.")
            with connection.transaction():
                for (product_id, _, _), vector in zip(batch, vectors, strict=True):
                    connection.execute(
                        """
                        INSERT INTO product_embeddings
                            (product_id, modality, model_name, model_version, embedding, status, error)
                        VALUES (%s, 'text', %s, %s, %s, 'ready', NULL)
                        ON CONFLICT (product_id, modality, model_name) DO UPDATE SET
                            model_version = EXCLUDED.model_version,
                            embedding = EXCLUDED.embedding,
                            status = 'ready', error = NULL, updated_at = now()
                        """,
                        (product_id, MODEL, digest, Vector(vector)),
                    )
            processed += len(batch)
            print(f"Embedded {processed}/{len(products)} products")

    elapsed = time.perf_counter() - started
    print(
        f"Stored {processed} {MODEL} embeddings ({dimensions} dimensions, "
        f"model digest {digest or 'unknown'}) in {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
