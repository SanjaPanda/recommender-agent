"""Catalogue-backed text and image retrieval used by the recommendation agent."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

from .embed_images import MODEL as IMAGE_MODEL
from .embed_text import MODEL as TEXT_MODEL, post_json

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def retrieve_products(
    *,
    query_context: str = "",
    selected_product_ids: list[str] | None = None,
    category_prefixes: list[str] | None = None,
    limit: int = 5,
    image_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Retrieve catalogue candidates using accumulated text and selected examples.

    If examples and query context are both present, image-seed and text-query
    rankings are fused with weighted reciprocal-rank fusion. This preserves the
    selected-product signal while allowing follow-up preferences to shift results.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not 0 <= image_weight <= 1:
        raise ValueError("image_weight must be between 0 and 1")
    refs = selected_product_ids or []
    if len(refs) > 3:
        raise ValueError("at most three selected product examples are supported")
    if len(set(refs)) != len(refs):
        raise ValueError("selected product IDs must be unique")
    if not query_context.strip() and not refs:
        return []

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL to the PostgreSQL database.")

    category_clause = ""
    category_params: list[Any] = []
    if category_prefixes:
        category_clause = " AND (" + " OR ".join(
            "p.raw_source ->> 'ground_truth_category' ILIKE %s"
            for _ in category_prefixes
        ) + ")"
        category_params = [prefix + "%" for prefix in category_prefixes]

    candidate_limit = max(limit * 10, 50)
    image_rows: list[tuple[Any, ...]] = []
    text_rows: list[tuple[Any, ...]] = []
    with psycopg.connect(database_url, connect_timeout=5) as connection:
        register_vector(connection)
        if refs:
            placeholders = ", ".join("%s" for _ in refs)
            reference_rows = connection.execute(
                f"""
                SELECT e.product_id, e.embedding::real[]
                FROM product_embeddings e
                WHERE e.product_id IN ({placeholders})
                  AND e.model_name = %s AND e.modality = 'image' AND e.status = 'ready'
                """,
                (*refs, IMAGE_MODEL),
            ).fetchall()
            found = {row[0] for row in reference_rows}
            missing = [product_id for product_id in refs if product_id not in found]
            if missing:
                raise ValueError(
                    "Image embeddings are missing for selected product IDs: "
                    + ", ".join(missing)
                )

            dimension = len(reference_rows[0][1])
            mean_vector = [0.0] * dimension
            for _, values in reference_rows:
                for index, value in enumerate(values):
                    mean_vector[index] += float(value) / len(reference_rows)
            norm = sum(value * value for value in mean_vector) ** 0.5
            if not norm:
                raise ValueError("Could not form an image vector from selected examples.")
            image_vector = Vector([value / norm for value in mean_vector])
            image_rows = connection.execute(
                f"""
                SELECT p.id, p.title, p.description, p.brand, p.category, p.image,
                       1 - (e.embedding <=> %s) AS similarity
                FROM product_embeddings e JOIN products p ON p.id = e.product_id
                WHERE e.model_name = %s AND e.modality = 'image' AND e.status = 'ready'
                  AND e.product_id NOT IN ({placeholders})
                """ + category_clause + " ORDER BY e.embedding <=> %s LIMIT %s",
                (image_vector, IMAGE_MODEL, *refs, *category_params,
                 image_vector, candidate_limit),
            ).fetchall()

        if query_context.strip():
            query_vector = post_json(
                "/api/embed",
                {"model": TEXT_MODEL, "input": "search_query: " + query_context.strip()},
            )["embeddings"][0]
            exclude_clause = ""
            exclude_params: list[str] = []
            if refs:
                placeholders = ", ".join("%s" for _ in refs)
                exclude_clause = f" AND e.product_id NOT IN ({placeholders})"
                exclude_params = refs
            text_rows = connection.execute(
                """
                SELECT p.id, p.title, p.description, p.brand, p.category, p.image,
                       1 - (e.embedding <=> %s) AS similarity
                FROM product_embeddings e JOIN products p ON p.id = e.product_id
                WHERE e.model_name = %s AND e.modality = 'text' AND e.status = 'ready'
                """ + exclude_clause + category_clause
                + " ORDER BY e.embedding <=> %s LIMIT %s",
                (Vector(query_vector), TEXT_MODEL, *exclude_params, *category_params,
                 Vector(query_vector), candidate_limit),
            ).fetchall()

    if refs and query_context.strip():
        image_rank_weight, text_rank_weight = image_weight, 1 - image_weight
    elif refs:
        image_rank_weight, text_rank_weight = 1.0, 0.0
    else:
        image_rank_weight, text_rank_weight = 0.0, 1.0

    fused: dict[str, dict[str, Any]] = {}
    for rows, modality, weight in (
        (image_rows, "image", image_rank_weight),
        (text_rows, "text", text_rank_weight),
    ):
        for rank, row in enumerate(rows, start=1):
            product_id, title, description, brand, category, image_meta, similarity = row
            item = fused.setdefault(product_id, {
                "id": product_id,
                "title": title,
                "description": description,
                "brand": brand,
                "category": category,
                "image_meta": image_meta,
                "image_similarity": None,
                "text_similarity": None,
                "score": 0.0,
            })
            item[f"{modality}_similarity"] = float(similarity)
            item["score"] += weight / (60 + rank)

    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:limit]
