"""Search products by semantic similarity with optional catalogue filters."""
from __future__ import annotations

import argparse
import os
import re
import urllib.request
import json
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

from .embed_text import MODEL, OLLAMA_URL, post_json

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
PRICE_REQUEST = re.compile(
    r"\b(budget|price|priced|cost)\b|"
    r"\b(?:under|below|less than)\s*(?:(?:₹|rs\.?|inr|usd|eur|gbp|dollars?|rupees?)\s*)?\d+|"
    r"(?:₹|\$|£|€)\s*\d+",
    re.IGNORECASE,
)


def main() -> None:
    """Search products by text with supported filters and optional image ranking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language product search")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", choices=("text", "hybrid"), default="text")
    parser.add_argument("--image-weight", type=float, default=0.1,
                        help="Weight for image rank in hybrid mode (0 to 1)")
    parser.add_argument(
        "--min-similarity", type=float, default=0.65,
        help="Hide top matches below this cosine similarity (default: 0.65)",
    )
    parser.add_argument("--brand")
    parser.add_argument("--category", help="Substring match on catalogue category")
    parser.add_argument(
        "--ground-truth-category-prefix", action="append", default=[],
        help="Restrict to a ground-truth category prefix; repeat to combine categories",
    )
    parser.add_argument("--secondhand", choices=("true", "false"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0 <= args.min_similarity <= 1:
        parser.error("--min-similarity must be between 0 and 1")
    if not 0 <= args.image_weight <= 1:
        parser.error("--image-weight must be between 0 and 1")

    if PRICE_REQUEST.search(args.query):
        print(json.dumps({
            "query": args.query,
            "results": [],
            "warning": "Price and budget requests cannot be enforced: this catalogue has no structured price field.",
        }, ensure_ascii=True))
        return

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Set DATABASE_URL to the PostgreSQL database.")
    query_vector = post_json(
        "/api/embed", {"model": MODEL, "input": "search_query: " + args.query}
    )["embeddings"][0]

    filters = ["e.model_name = %s", "e.modality = 'text'", "e.status = 'ready'"]
    filter_params: list[Any] = [MODEL]
    if args.brand:
        filters.append("p.brand ILIKE %s")
        filter_params.append(args.brand)
    if args.category:
        filters.append("p.category ILIKE %s")
        filter_params.append(f"%{args.category}%")
    if args.ground_truth_category_prefix:
        filters.append("(" + " OR ".join(
            "p.raw_source ->> 'ground_truth_category' ILIKE %s"
            for _ in args.ground_truth_category_prefix
        ) + ")")
        filter_params.extend(prefix + "%" for prefix in args.ground_truth_category_prefix)
    if args.secondhand is not None:
        filters.append("p.is_secondhand = %s")
        filter_params.append(args.secondhand == "true")
    sql = f"""
        SELECT p.id, p.title, p.description, p.brand, p.category,
               p.is_secondhand, 1 - (e.embedding <=> %s) AS similarity
        FROM product_embeddings e
        JOIN products p ON p.id = e.product_id
        WHERE {' AND '.join(filters)}
        ORDER BY e.embedding <=> %s
        LIMIT %s
    """
    # The distance operator needs the query vector once for the score and once for ranking.
    candidate_limit = max(args.limit * 5, 50) if args.mode == "hybrid" else args.limit
    params = [Vector(query_vector), *filter_params, Vector(query_vector), candidate_limit]
    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        text_rows = connection.execute(sql, params).fetchall()
        if args.mode == "hybrid":
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as exc:
                raise SystemExit(
                    "Hybrid search dependencies are optional. Install them with: "
                    "uv sync --extra image"
                ) from exc
            clip_model_name = os.environ.get(
                "IMAGE_EMBEDDING_MODEL", "openai/clip-vit-base-patch32"
            )
            clip_model = CLIPModel.from_pretrained(clip_model_name).to("cpu").eval()
            clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            clip_inputs = clip_processor(text=[args.query], return_tensors="pt", padding=True)
            with torch.inference_mode():
                clip_vector = clip_model.get_text_features(**clip_inputs)[0].cpu().tolist()
            clip_norm = sum(value * value for value in clip_vector) ** 0.5
            clip_vector = [value / clip_norm for value in clip_vector]
            image_filters = [
                "e.model_name = %s", "e.modality = 'image'", "e.status = 'ready'",
            ]
            image_params: list[Any] = [clip_model_name]
            # Reuse all deterministic catalogue filters from the text query.
            image_filters.extend(filters[3:])
            image_params.extend(filter_params[1:])
            image_sql = f"""
                SELECT p.id, p.title, p.description, p.brand, p.category,
                       p.is_secondhand, 1 - (e.embedding <=> %s) AS similarity
                FROM product_embeddings e JOIN products p ON p.id = e.product_id
                WHERE {' AND '.join(image_filters)}
                ORDER BY e.embedding <=> %s LIMIT %s
            """
            image_rows = connection.execute(
                image_sql,
                [Vector(clip_vector), *image_params, Vector(clip_vector), max(args.limit * 5, 50)],
            ).fetchall()
            # Weighted reciprocal-rank fusion avoids assuming that Nomic and CLIP
            # cosine scores have the same calibration.
            combined: dict[str, dict[str, Any]] = {}
            for modality_rows, modality in ((text_rows, "text"), (image_rows, "image")):
                for rank, row in enumerate(modality_rows, start=1):
                    product_id, title, description, brand, category, secondhand, similarity = row
                    item = combined.setdefault(product_id, {
                        "id": product_id, "title": title, "description": description,
                        "brand": brand, "category": category, "is_secondhand": secondhand,
                        "text_similarity": None, "image_similarity": None, "hybrid_score": 0.0,
                    })
                    item[f"{modality}_similarity"] = float(similarity)
                    weight = (1 - args.image_weight) if modality == "text" else args.image_weight
                    item["hybrid_score"] += weight / (60 + rank)
            rows = sorted(combined.values(), key=lambda item: item["hybrid_score"], reverse=True)
            rows = [row for row in rows if max(
                row["text_similarity"] or 0.0, row["image_similarity"] or 0.0
            ) >= args.min_similarity][:args.limit]
        else:
            rows = [
                {"id": row[0], "title": row[1], "description": row[2], "brand": row[3],
                 "category": row[4], "is_secondhand": row[5], "text_similarity": float(row[6]),
                 "image_similarity": None, "hybrid_score": None}
                for row in text_rows if float(row[6]) >= args.min_similarity
            ]
    for row in rows:
        print(json.dumps({
            "id": row["id"],
            "title": row["title"],
            "description": row["description"][:240] if row["description"] else None,
            "brand": row["brand"],
            "category": row["category"],
            "is_secondhand": row["is_secondhand"],
            "text_similarity": round(row["text_similarity"], 4)
            if row["text_similarity"] is not None else None,
            "image_similarity": round(row["image_similarity"], 4)
            if row["image_similarity"] is not None else None,
            "hybrid_score": round(row["hybrid_score"], 6)
            if row["hybrid_score"] is not None else None,
        }, ensure_ascii=True))
    if not rows:
        print(json.dumps({
            "query": args.query,
            "results": [],
            "message": (
                "No products reached the minimum similarity. Try broader search terms."
                if args.min_similarity > 0
                else "No matching products. Generate embeddings first with recommender-embed-products."
            ),
            "minimum_similarity": args.min_similarity,
        }, ensure_ascii=True))


if __name__ == "__main__":
    main()
