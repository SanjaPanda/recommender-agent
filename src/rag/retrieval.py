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
    filters: dict[str, Any] | None = None,
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

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL to the PostgreSQL database.")

    category_clause, category_params = filter_clause(filters or {}, category_prefixes or [])

    candidate_limit = max(limit * 10, 50)
    image_rows: list[tuple[Any, ...]] = []
    text_rows: list[tuple[Any, ...]] = []
    with psycopg.connect(database_url, connect_timeout=5) as connection:
        register_vector(connection)
        if not query_context.strip() and not refs:
            rows = connection.execute(
                "SELECT p.id,p.title,p.description,p.brand,p.category,p.image,p.is_secondhand "
                "FROM products p WHERE true" + category_clause + " ORDER BY p.id LIMIT %s",
                (*category_params, limit),
            ).fetchall()
            return [dict(zip(["id","title","description","brand","category","image_meta","is_secondhand"], row),
                         score=0.0, image_similarity=None, text_similarity=None) for row in rows]
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


def filter_clause(filters: dict, category_prefixes: list[str]) -> tuple[str, list]:
    """Compile the allowlisted filter contract to parameterized SQL."""
    allowed = {"brand", "is_secondhand", "category_prefixes"}
    if set(filters) - allowed:
        raise ValueError("Unsupported catalogue filter")
    clauses, params = [], []
    prefixes = filters.get("category_prefixes") or category_prefixes
    if prefixes:
        terms = []
        for prefix in prefixes:
            # Compare exact category or a descendant; user wildcards are literal.
            terms.append("(p.raw_source ->> 'ground_truth_category' = %s OR starts_with(p.raw_source ->> 'ground_truth_category', %s))")
            params.extend([prefix, prefix + " > "])
        clauses.append("(" + " OR ".join(terms) + ")")
    if filters.get("brand") is not None:
        clauses.append("lower(p.brand) = lower(%s)")
        params.append(filters["brand"])
    if filters.get("is_secondhand") is not None:
        if not isinstance(filters["is_secondhand"], bool):
            raise ValueError("is_secondhand must be a boolean")
        clauses.append("p.is_secondhand = %s")
        params.append(filters["is_secondhand"])
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def list_categories() -> list[dict]:
    """Return locally available top-level categories and product counts."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=5) as connection:
        rows = connection.execute(
            "SELECT split_part(raw_source->>'ground_truth_category', ' > ', 1),count(*) "
            "FROM products WHERE NULLIF(raw_source->>'ground_truth_category','') IS NOT NULL "
            "GROUP BY 1 ORDER BY 1"
        ).fetchall()
    return [{"name": name, "count": count} for name, count in rows]
