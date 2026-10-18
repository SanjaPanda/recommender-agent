"""Run a small labeled retrieval evaluation and report ranking and latency."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

from .embed_text import MODEL, post_json

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DEFAULT_QUERIES = ROOT / "data" / "phase2_queries.jsonl"


def read_queries(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL retrieval cases from the supplied evaluation file."""
    with path.open(encoding="utf-8") as source:
        queries = [json.loads(line) for line in source if line.strip()]
    if not queries:
        raise SystemExit(f"No evaluation queries found in {path}")
    return queries


def main() -> None:
    """Evaluate text retrieval ranks and latency against labeled product IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--ground-truth-category-prefix", action="append", default=[],
        help="Restrict every query to these ground-truth category prefixes",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Set DATABASE_URL to the PostgreSQL database.")

    queries = read_queries(args.queries)
    category_clause = ""
    category_params: list[str] = []
    if args.ground_truth_category_prefix:
        category_clause = " AND (" + " OR ".join(
            "p.raw_source ->> 'ground_truth_category' ILIKE %s"
            for _ in args.ground_truth_category_prefix
        ) + ")"
        category_params = [prefix + "%" for prefix in args.ground_truth_category_prefix]
    # Keep model startup time out of per-query latency; report warm search latency.
    post_json("/api/embed", {"model": MODEL, "input": "search_query: evaluation warmup"})
    results: list[dict[str, Any]] = []
    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        for item in queries:
            query = item["query"]
            relevant_ids = set(item["relevant_ids"])
            expected_empty = item.get("expected_empty", False)
            embed_start = time.perf_counter()
            vector = post_json(
                "/api/embed", {"model": MODEL, "input": "search_query: " + query}
            )["embeddings"][0]
            embed_ms = (time.perf_counter() - embed_start) * 1000

            search_start = time.perf_counter()
            rows = connection.execute(
                """
                SELECT p.id, p.title, 1 - (e.embedding <=> %s) AS similarity
                FROM product_embeddings e JOIN products p ON p.id = e.product_id
                WHERE e.model_name = %s AND e.modality = 'text' AND e.status = 'ready'
                """ + category_clause + """
                ORDER BY e.embedding <=> %s LIMIT %s
                """,
                (Vector(vector), MODEL, *category_params, Vector(vector), args.k),
            ).fetchall()
            search_ms = (time.perf_counter() - search_start) * 1000
            ranked_ids = [row[0] for row in rows]
            rank = next(
                (index for index, product_id in enumerate(ranked_ids, start=1)
                 if product_id in relevant_ids),
                None,
            )
            top_similarity = float(rows[0][2]) if rows else None
            results.append({
                "query": query,
                "expected_empty": expected_empty,
                "unsupported_constraint": item.get("unsupported_constraint"),
                "rank": rank,
                "hit_at_k": rank is not None,
                "reciprocal_rank": 1 / rank if rank else 0.0,
                "top_similarity": round(top_similarity, 4) if top_similarity is not None else None,
                "returned_for_empty_query": expected_empty and bool(rows),
                "embedding_ms": round(embed_ms, 2),
                "database_search_ms": round(search_ms, 2),
                "top_results": [
                    {"title": title, "similarity": round(float(score), 4)}
                    for _, title, score in rows[:3]
                ],
            })

    in_catalogue = [row for row in results if not row["expected_empty"]]
    out_of_catalogue = [row for row in results if row["expected_empty"]]
    hit_rate = sum(row["hit_at_k"] for row in in_catalogue) / max(1, len(in_catalogue))
    mrr = sum(row["reciprocal_rank"] for row in in_catalogue) / max(1, len(in_catalogue))
    mean_embed = sum(row["embedding_ms"] for row in results) / len(results)
    mean_search = sum(row["database_search_ms"] for row in results) / len(results)
    threshold_sweep = []
    for threshold in (0.55, 0.60, 0.65, 0.70):
        true_accepts = sum(
            row["rank"] == 1 and row["top_similarity"] is not None
            and row["top_similarity"] >= threshold
            for row in in_catalogue
        )
        false_accepts = sum(
            row["top_similarity"] is not None and row["top_similarity"] >= threshold
            for row in out_of_catalogue
        )
        threshold_sweep.append({
            "minimum_top_similarity": threshold,
            "in_catalogue_correct_top1_rate": round(true_accepts / max(1, len(in_catalogue)), 4),
            "out_of_catalogue_accept_rate": round(false_accepts / max(1, len(out_of_catalogue)), 4),
        })
    print(json.dumps({
        "model": MODEL,
        "ground_truth_category_prefixes": args.ground_truth_category_prefix,
        "query_count": len(results),
        "in_catalogue_query_count": len(in_catalogue),
        "out_of_catalogue_query_count": len(out_of_catalogue),
        "k": args.k,
        "in_catalogue_hit_rate_at_k": round(hit_rate, 4),
        "in_catalogue_mean_reciprocal_rank": round(mrr, 4),
        "unfiltered_out_of_catalogue_return_rate": round(
            sum(row["returned_for_empty_query"] for row in out_of_catalogue)
            / max(1, len(out_of_catalogue)), 4
        ),
        "out_of_catalogue_top_similarities": [
            {"query": row["query"], "top_similarity": row["top_similarity"]}
            for row in out_of_catalogue
        ],
        "exploratory_similarity_threshold_sweep": threshold_sweep,
        "mean_embedding_ms": round(mean_embed, 2),
        "mean_database_search_ms": round(mean_search, 2),
        "queries": results,
        "caveat": "One hand-selected relevant product per in-catalogue query; this small set is not a representative benchmark. Raw search always returns candidates; threshold sweep is exploratory, not a calibrated abstention policy.",
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
