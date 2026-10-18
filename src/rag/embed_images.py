"""Generate local CLIP image embeddings for products with saved image files."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
MODEL = os.environ.get("IMAGE_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
DEFAULT_IMAGE_ROOTS = [
    ROOT / ".data" / "shopify-catalogue-sample",
    ROOT / ".data" / "shopify-outdoor-garden",
    ROOT / ".data" / "shopify-outdoor-garden-1000",
    ROOT / ".data" / "shopify-selected-train",
    ROOT / ".data" / "shopify-selected-test",
]


def resolve_image(local_path: str, roots: list[Path]) -> Path | None:
    """Resolve a saved image path against allowed roots, preserving absolute paths."""
    candidate = Path(local_path)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for root in roots:
        resolved = root / candidate
        if resolved.is_file():
            return resolved
    return None


def main() -> None:
    """Embed local product images with CLIP and store normalized vectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum products to process")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Recompute existing image vectors")
    parser.add_argument(
        "--image-root", type=Path, action="append", default=None,
        help="Directory containing image local_path values; can be repeated",
    )
    parser.add_argument(
        "--ground-truth-category-prefix", action="append", default=[],
        help="Restrict embedding to products in these ground-truth categories",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 1 <= args.batch_size <= 16:
        parser.error("--batch-size must be between 1 and 16")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Set DATABASE_URL to the PostgreSQL database.")
    roots = [path.resolve() for path in (args.image_root or DEFAULT_IMAGE_ROOTS)]

    try:
        import torch
        from PIL import Image, UnidentifiedImageError
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise SystemExit(
            "Image embedding dependencies are optional. Install them with: "
            "uv sync --extra image"
        ) from exc

    started = time.perf_counter()
    model = CLIPModel.from_pretrained(MODEL).to("cpu").eval()
    processor = CLIPProcessor.from_pretrained(MODEL)
    digest = getattr(model.config, "_commit_hash", None) or "main"
    limit_clause = " LIMIT %s" if args.limit is not None else ""
    existing_clause = "" if args.force else """
        AND NOT EXISTS (
            SELECT 1 FROM product_embeddings e
            WHERE e.product_id = p.id AND e.modality = 'image'
              AND e.model_name = %s AND e.model_version = %s AND e.status = 'ready'
        )
    """
    params: list[Any] = []
    category_clause = ""
    if args.ground_truth_category_prefix:
        category_clause = " AND (" + " OR ".join(
            "p.raw_source ->> 'ground_truth_category' ILIKE %s"
            for _ in args.ground_truth_category_prefix
        ) + ")"
        params.extend(prefix + "%" for prefix in args.ground_truth_category_prefix)
    if not args.force:
        params.extend([MODEL, digest])
    if args.limit is not None:
        params.append(args.limit)

    with psycopg.connect(database_url, autocommit=True) as connection:
        register_vector(connection)
        rows = connection.execute(
            """
            SELECT p.id, p.image, p.raw_source
            FROM products p
            WHERE p.image IS NOT NULL
            """ + category_clause + existing_clause + " ORDER BY p.id" + limit_clause,
            params,
        ).fetchall()
        missing = invalid = processed = 0
        dimensions = model.config.projection_dim
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            valid: list[tuple[str, Any]] = []
            for product_id, image_meta, raw_source in batch:
                image_meta = image_meta or {}
                local_path = image_meta.get("local_path")
                if not local_path:
                    local_path = (raw_source or {}).get("product_image", {}).get("local_path")
                image_path = resolve_image(local_path, roots) if local_path else None
                if image_path is None:
                    missing += 1
                    continue
                try:
                    with Image.open(image_path) as image:
                        valid.append((product_id, image.convert("RGB")))
                except (OSError, UnidentifiedImageError):
                    invalid += 1
            if not valid:
                continue
            inputs = processor(images=[image for _, image in valid], return_tensors="pt")
            with torch.inference_mode():
                vectors = model.get_image_features(**inputs).cpu().tolist()
            # Normalize for cosine similarity and make rows durable per batch.
            with connection.transaction():
                for (product_id, image), vector in zip(valid, vectors, strict=True):
                    norm = sum(value * value for value in vector) ** 0.5
                    normalized = [value / norm for value in vector]
                    connection.execute(
                        """
                        INSERT INTO product_embeddings
                            (product_id, modality, model_name, model_version, embedding, status, error)
                        VALUES (%s, 'image', %s, %s, %s, 'ready', NULL)
                        ON CONFLICT (product_id, modality, model_name) DO UPDATE SET
                            model_version = EXCLUDED.model_version,
                            embedding = EXCLUDED.embedding,
                            status = 'ready', error = NULL, updated_at = now()
                        """,
                        (product_id, MODEL, digest, Vector(normalized)),
                    )
                    image.close()
            processed += len(valid)
            print(f"Embedded {processed}/{len(rows)} candidate products")

    elapsed = time.perf_counter() - started
    print(json.dumps({
        "model": MODEL,
        "model_revision": digest,
        "dimensions": dimensions,
        "embedded": processed,
        "missing_image_files": missing,
        "unreadable_images": invalid,
        "elapsed_seconds": round(elapsed, 1),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
