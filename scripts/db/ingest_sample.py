"""Idempotently import the local JSONL sample into PostgreSQL."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DEFAULT_INPUT = ROOT / ".data" / "shopify-catalogue-sample" / "products.jsonl"
UPSERT_PRODUCT = """
INSERT INTO products (
    id, title, description, brand, category, is_secondhand,
    image, raw_source, source_name
) VALUES (
    %(id)s, %(title)s, %(description)s, %(brand)s, %(category)s,
    %(is_secondhand)s, %(image)s, %(raw_source)s, 'Shopify/product-catalogue'
)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    brand = EXCLUDED.brand,
    category = EXCLUDED.category,
    is_secondhand = EXCLUDED.is_secondhand,
    image = EXCLUDED.image,
    raw_source = EXCLUDED.raw_source,
    source_name = EXCLUDED.source_name
"""


def source_image_meta(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the source image metadata object when the row has one."""
    image = row.get("product_image")
    return image if isinstance(image, dict) else None


def main() -> None:
    """Read sample JSONL and idempotently upsert catalogue products."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.input.is_file():
        raise SystemExit(
            f"Sample file not found: {args.input}. Fetch it with scripts/data/fetch_sample.py."
        )

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Set DATABASE_URL to the target PostgreSQL database.")

    count = 0
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor, args.input.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if args.limit is not None and count >= args.limit:
                    break
                row = json.loads(line)
                image_meta = source_image_meta(row)
                values = {
                    "id": row["id"],
                    "title": row.get("product_title"),
                    "description": row.get("product_description"),
                    "brand": row.get("ground_truth_brand"),
                    "category": " | ".join(row.get("potential_product_categories") or [])
                    or row.get("ground_truth_category"),
                    "is_secondhand": row.get("ground_truth_is_secondhand"),
                    "image": Jsonb(image_meta) if image_meta is not None else None,
                    "raw_source": Jsonb(row),
                }
                try:
                    cursor.execute(UPSERT_PRODUCT, values)
                except Exception as exc:
                    raise RuntimeError(f"Failed to import source line {line_number}") from exc
                count += 1

    print(f"Upserted {count} product rows from {args.input}")
    print("Running the importer again updates those same rows by deterministic ID.")


if __name__ == "__main__":
    main()
