"""Compare text-only and hybrid retrieval on a labeled category slice."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import psycopg
import torch
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector
from transformers import CLIPModel, CLIPProcessor

from .embed_images import MODEL as IMAGE_MODEL
from .embed_text import MODEL as TEXT_MODEL, post_json

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DEFAULT_QUERIES = ROOT / "data" / "outdoor_garden_queries.jsonl"


def read_queries(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL cases for multimodal retrieval comparison."""
    with path.open(encoding="utf-8") as source:
        queries = [json.loads(line) for line in source if line.strip()]
    if not queries:
        raise SystemExit(f"No evaluation queries found in {path}")
    return queries


def normalized_text_features(model: CLIPModel, processor: CLIPProcessor, query: str) -> list[float]:
    """Encode a query with CLIP and return its unit-normalized text features."""
    tokens = processor(text=[query], return_tensors="pt", padding=True)
    with torch.inference_mode():
        vector = model.get_text_features(**tokens)[0].cpu().tolist()
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector]


def metrics(ranks: list[int | None], k: int) -> dict[str, float]:
    """Compute hit rate at k and mean reciprocal rank for labeled cases."""
    return {
        f"hit_rate_at_{k}": round(sum(rank is not None and rank <= k for rank in ranks) / max(1, len(ranks)), 4),
        "mean_reciprocal_rank": round(
            sum(1 / rank if rank is not None else 0.0 for rank in ranks) / max(1, len(ranks)), 4
        ),
    }


def main() -> None:
    """Compare text-only results with weighted text and image rank fusion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--image-weight", type=float, default=0.1)
    parser.add_argument(
        "--ground-truth-category-prefix", action="append",
        default=None,
        help="Restrict the candidate set to these categories",
    )
    args = parser.parse_args()
    if args.k < 1 or args.candidate_count < args.k:
        parser.error("candidate-count must be at least k, and k must be positive")
    if not 0 <= args.image_weight <= 1:
        parser.error("--image-weight must be between 0 and 1")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Set DATABASE_URL to the PostgreSQL database.")

    queries = read_queries(args.queries)
    clip_model = CLIPModel.from_pretrained(IMAGE_MODEL).to("cpu").eval()
    clip_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
    post_json("/api/embed", {"model": TEXT_MODEL, "input": "search_query: retrieval warmup"})

    category_prefixes = args.ground_truth_category_prefix or ["Home & Garden > Lawn & Garden"]
    category_clause = " AND (" + " OR ".join(
        "p.raw_source ->> 'ground_truth_category' ILIKE %s"
        for _ in category_prefixes
    ) + ")"
    category_params = [prefix + "%" for prefix in category_prefixes]
    text_ranks: list[int | None] = []
    hybrid_ranks: list[int | None] = []
    details: list[dict[str, Any]] = []
    started = time.perf_counter()

    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        for item in queries:
            query = item["query"]
            relevant = set(item["relevant_ids"])
            nomic_vector = post_json(
                "/api/embed", {"model": TEXT_MODEL, "input": "search_query: " + query}
            )["embeddings"][0]
            clip_vector = normalized_text_features(clip_model, clip_processor, query)
            text_rows = connection.execute(
                """
                SELECT p.id, e.embedding <=> %s AS distance
                FROM product_embeddings e JOIN products p ON p.id = e.product_id
                WHERE e.model_name = %s AND e.modality = 'text' AND e.status = 'ready'
                """ + category_clause + " ORDER BY e.embedding <=> %s LIMIT %s",
                (Vector(nomic_vector), TEXT_MODEL, *category_params,
                 Vector(nomic_vector), args.candidate_count),
            ).fetchall()
            image_rows = connection.execute(
                """
                SELECT p.id, e.embedding <=> %s AS distance
                FROM product_embeddings e JOIN products p ON p.id = e.product_id
                WHERE e.model_name = %s AND e.modality = 'image' AND e.status = 'ready'
                """ + category_clause + " ORDER BY e.embedding <=> %s LIMIT %s",
                (Vector(clip_vector), IMAGE_MODEL, *category_params,
                 Vector(clip_vector), args.candidate_count),
            ).fetchall()

            text_ids = [row[0] for row in text_rows]
            fused: dict[str, float] = {}
            for rows, weight in (
                (text_rows, 1 - args.image_weight),
                (image_rows, args.image_weight),
            ):
                for rank, (product_id, _) in enumerate(rows, start=1):
                    fused[product_id] = fused.get(product_id, 0.0) + weight / (60 + rank)
            hybrid_ids = sorted(fused, key=fused.get, reverse=True)

            text_rank = next((i for i, pid in enumerate(text_ids, 1) if pid in relevant), None)
            hybrid_rank = next((i for i, pid in enumerate(hybrid_ids, 1) if pid in relevant), None)
            text_ranks.append(text_rank)
            hybrid_ranks.append(hybrid_rank)
            details.append({
                "query": query,
                "text_rank": text_rank,
                "hybrid_rank": hybrid_rank,
                "text_top5": text_ids[:5],
                "hybrid_top5": hybrid_ids[:5],
            })

    print(json.dumps({
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
        "category_prefixes": category_prefixes,
        "query_count": len(queries),
        "candidate_count_per_modality": args.candidate_count,
        "image_weight": args.image_weight,
        "text_only": metrics(text_ranks, args.k),
        "hybrid": metrics(hybrid_ranks, args.k),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "queries": details,
        "caveat": "Hand-labeled one-to-two relevant products per query on a 55-product category subset; exploratory only.",
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
