"""Inspect recommendations from selected examples and optional preferences."""
from __future__ import annotations

import argparse
import json

from .retrieval import retrieve_products


def main() -> None:
    """Print recommendations for one to three product examples and refinements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-product-id", action="append", required=True,
        help="Selected product example; provide one, two, or three IDs",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--refinement",
        help="Optional natural-language preference to blend with visual similarity",
    )
    parser.add_argument(
        "--image-weight", type=float, default=0.5,
        help="Image-rank contribution when --refinement is supplied (0 to 1)",
    )
    parser.add_argument(
        "--ground-truth-category-prefix", action="append", default=[],
        help="Optional category scope for recommendations; can be repeated",
    )
    args = parser.parse_args()
    if not 1 <= len(args.reference_product_id) <= 3:
        parser.error("provide one, two, or three --reference-product-id values")
    if len(set(args.reference_product_id)) != len(args.reference_product_id):
        parser.error("reference product IDs must be unique")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0 <= args.image_weight <= 1:
        parser.error("--image-weight must be between 0 and 1")

    rows = retrieve_products(
        query_context=args.refinement or "",
        selected_product_ids=args.reference_product_id,
        category_prefixes=args.ground_truth_category_prefix,
        limit=args.limit,
        image_weight=args.image_weight,
    )
    for row in rows:
        print(json.dumps({
            **row,
            "image_similarity": round(row["image_similarity"], 4)
            if row["image_similarity"] is not None else None,
            "text_similarity": round(row["text_similarity"], 4)
            if row["text_similarity"] is not None else None,
            "score": round(row["score"], 6),
            "based_on_examples": args.reference_product_id,
            "refinement": args.refinement,
        }, ensure_ascii=True))


if __name__ == "__main__":
    main()
