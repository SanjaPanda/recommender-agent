"""Shared revision selection and atomic vector storage for offline encoders.

Text formatting, vector dimensions and image normalization belong to each job.
These helpers only select missing revisions and persist one completed batch.
"""
from __future__ import annotations

from collections.abc import Iterable

from pgvector import Vector
import psycopg


def pending_clause(modality: str, model: str, revision: str, force: bool) -> tuple[str, list[str]]:
    """Return an SQL eligibility suffix and parameters for alias ``products p``.

    A ready vector skips work only for the requested modality, model and immutable
    revision. Force includes every otherwise eligible product.
    """
    if force:
        return "", []
    return """
        AND NOT EXISTS (
            SELECT 1 FROM product_embeddings e
            WHERE e.product_id = p.id AND e.modality = %s
              AND e.model_name = %s AND e.model_version = %s AND e.status = 'ready'
        )
    """, [modality, model, revision]


def store_batch(connection: psycopg.Connection, product_ids: Iterable[str],
                vectors: Iterable[list[float]], *, modality: str, model: str,
                revision: str) -> None:
    """Atomically upsert one encoder batch without changing vector values.

    Args:
        connection: Job connection holding the catalog maintenance lock.
        product_ids: Product IDs in encoder input order.
        vectors: Corresponding raw text or normalized image vectors.
        modality: Stored modality (text or image).
        model: Encoder model name.
        revision: Immutable revision used to generate this batch.

    A length mismatch or database error rolls back the entire batch. Successful
    writes replace the same product/modality/model row and clear stale errors.
    """
    with connection.transaction():
        for product_id, vector in zip(product_ids, vectors, strict=True):
            connection.execute(
                """
                INSERT INTO product_embeddings
                    (product_id, modality, model_name, model_version, embedding, status, error)
                VALUES (%s, %s, %s, %s, %s, 'ready', NULL)
                ON CONFLICT (product_id, modality, model_name) DO UPDATE SET
                    model_version = EXCLUDED.model_version,
                    embedding = EXCLUDED.embedding,
                    status = 'ready', error = NULL, updated_at = now()
                """,
                (product_id, modality, model, revision, Vector(vector)),
            )
