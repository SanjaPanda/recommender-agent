"""Idempotently import the local JSONL sample into PostgreSQL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.types.json import Jsonb

from agent.config import ROOT, settings

DEFAULT_INPUTS = (
    ROOT / ".data" / "shopify-selected-train" / "products.jsonl",
    ROOT / ".data" / "shopify-selected-test" / "products.jsonl",
)
APPAREL_CATEGORY = "Apparel & Accessories"
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


def is_apparel(row: dict[str, Any]) -> bool:
    """Return whether a dataset row belongs to the apparel taxonomy."""
    category = row.get("ground_truth_category")
    return isinstance(category, str) and (
        category == APPAREL_CATEGORY or category.startswith(APPAREL_CATEGORY + " > ")
    )


def read_apparel_rows(paths: Iterable[Path], limit: int | None = None) -> list[dict[str, Any]]:
    """Read validated apparel rows from the pinned train and test samples.

    Non-apparel rows are ignored, duplicate source IDs are loaded once, and
    ``limit`` counts apparel rows only.
    """
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Apparel sample is missing: {path}. See the documented fetch commands."
            )
        with path.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                row = json.loads(line)
                if not is_apparel(row):
                    continue
                product_id = row.get("id")
                if not isinstance(product_id, str) or not product_id.strip():
                    raise ValueError(f"Missing product ID in {path}, line {line_number}.")
                if product_id in seen_ids:
                    continue
                seen_ids.add(product_id)
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def main() -> None:
    """Import only apparel from the pinned train/test samples into PostgreSQL.

    Run with ``uv run python scripts/db/ingest_sample.py``. The importer skips
    non-apparel rows and upserts by stable source ID, so reruns are idempotent.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    rows = read_apparel_rows(DEFAULT_INPUTS, args.limit)
    with psycopg.connect(settings.database_url.get_secret_value()) as connection:
        with connection.cursor() as cursor:
            for row in rows:
                image_meta = source_image_meta(row)
                cursor.execute(UPSERT_PRODUCT, {
                    "id": row["id"],
                    "title": row.get("product_title"),
                    "description": row.get("product_description"),
                    "brand": row.get("ground_truth_brand"),
                    "category": row.get("ground_truth_category"),
                    "is_secondhand": row.get("ground_truth_is_secondhand"),
                    "image": Jsonb(image_meta) if image_meta is not None else None,
                    "raw_source": Jsonb(row),
                })
    print(f"Upserted {len(rows)} unique Apparel & Accessories products from the train/test samples.")
    print("Running the importer again updates those same rows by deterministic ID.")


if __name__ == "__main__":
    main()
