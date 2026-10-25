"""Inspect the same apparel retrieval pipeline used by the recommendation agent."""
from __future__ import annotations

import argparse
import json

from . import validate_category_prefixes
from .retrieval import retrieve_products


def main() -> None:
    """Print hybrid apparel matches for a query and/or selected product examples.

    Run ``uv run recommender-search "blue cotton dress"`` or include repeated
    ``--reference-product-id`` flags to add CLIP image examples.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Natural-language apparel need or refinement")
    parser.add_argument("--reference-product-id", action="append", default=[],
                        help="Displayed apparel product to use as a visual example; repeat up to three times")
    parser.add_argument("--category-prefix", action="append", default=[],
                        help="Optional apparel category path; repeat to include more paths")
    parser.add_argument("--brand", help="Exact catalogue brand filter")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if not args.query and not args.reference_product_id:
        parser.error("provide a query or at least one --reference-product-id")
    if not 1 <= len(args.reference_product_id) <= 3 and args.reference_product_id:
        parser.error("provide one to three --reference-product-id values")
    if len(set(args.reference_product_id)) != len(args.reference_product_id):
        parser.error("reference product IDs must be unique")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        validate_category_prefixes(args.category_prefix)
    except ValueError as exc:
        parser.error(str(exc))

    from agent.conditions import requests_condition, UNSUPPORTED_CONDITION
    if requests_condition(args.query or ""):
        parser.error(UNSUPPORTED_CONDITION)

    filters = {
        "brand": args.brand,
        "category_prefixes": args.category_prefix,
    }
    rows = retrieve_products(
        query_context=args.query or "",
        selected_product_ids=args.reference_product_id,
        filters=filters,
        limit=args.limit,
    )
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != "image_meta"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
