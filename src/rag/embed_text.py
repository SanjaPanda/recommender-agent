"""Generate revisioned Nomic product vectors through local Ollama.

Run ``uv run python -m rag.embed_text --help`` from the repository root.
Normal execution reads PostgreSQL products, calls Ollama and writes vectors in
batches. Configure services and private assets as documented in docs/SETUP.md.
"""
from __future__ import annotations

import argparse
import time

import psycopg
from pgvector.psycopg import register_vector

from agent.config import settings
from agent.tools.embeddings import MODEL, document_text, embed_texts, model_digest
from . import CATALOG_WRITE_LOCK
from .embedding_store import pending_clause, store_batch

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

    database_url = settings.database_url.get_secret_value()
    digest = model_digest()
    limit_sql = " LIMIT %s" if args.limit is not None else ""
    existing_filter, select_params = pending_clause("text", MODEL, digest, args.force)
    if args.limit is not None:
        select_params.append(args.limit)
    query = """
        SELECT p.id, p.title, p.description, p.features
        FROM products p
        WHERE (NULLIF(trim(p.title), '') IS NOT NULL
               OR NULLIF(trim(p.description), '') IS NOT NULL)
    """ + existing_filter + " ORDER BY p.id" + limit_sql

    started = time.perf_counter()
    processed = 0
    dimensions: int | None = None
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(%s)", (CATALOG_WRITE_LOCK,))
        register_vector(connection)
        products = connection.execute(query, select_params).fetchall()
        if not products:
            print(f"No products need embeddings for {MODEL} ({digest or 'unknown version'}).")
            return

        for offset in range(0, len(products), args.batch_size):
            batch = products[offset : offset + args.batch_size]
            texts = [document_text(title, description, features) for _, title, description, features in batch]
            vectors = embed_texts(texts)
            for vector in vectors:
                if dimensions is None:
                    dimensions = len(vector)
                elif len(vector) != dimensions:
                    raise RuntimeError("Ollama returned inconsistent embedding dimensions.")
            store_batch(connection, (row[0] for row in batch), vectors,
                        modality="text", model=MODEL, revision=digest)
            processed += len(batch)
            print(f"Embedded {processed}/{len(products)} products")

    elapsed = time.perf_counter() - started
    print(
        f"Stored {processed} {MODEL} embeddings ({dimensions} dimensions, "
        f"model digest {digest or 'unknown'}) in {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
