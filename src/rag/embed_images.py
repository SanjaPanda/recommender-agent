"""Generate local CLIP image embeddings for products with saved image files.

Run ``uv run python -m rag.embed_images --help`` from the repository root.
The normal command reads private product images, loads CLIP and writes
revisioned PostgreSQL vectors in batches; see docs/SETUP.md."""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from agent.config import ROOT, settings
from . import CATALOG_WRITE_LOCK
from .source import local_asset
from .embedding_store import pending_clause, store_batch

MODEL = settings.image_embedding_model


def main() -> None:
    """Embed local product images with CLIP and store normalized vectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum products to process")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Recompute existing image vectors")
    parser.add_argument(
        "--ground-truth-category-prefix", action="append", default=[],
        help="Restrict embedding to products in these ground-truth categories",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 1 <= args.batch_size <= 16:
        parser.error("--batch-size must be between 1 and 16")

    database_url = settings.database_url.get_secret_value()

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
    digest = getattr(model.config, "_commit_hash", None)
    if not digest:
        raise RuntimeError('CLIP model has no immutable revision; install a versioned snapshot before embedding.')
    processor = CLIPProcessor.from_pretrained(MODEL, revision=digest)
    limit_clause = " LIMIT %s" if args.limit is not None else ""
    existing_clause, version_params = pending_clause("image", MODEL, digest, args.force)
    params: list[Any] = []
    category_clause = ""
    if args.ground_truth_category_prefix:
        category_clause = " AND (" + " OR ".join(
            "p.raw_source ->> 'ground_truth_category' ILIKE %s"
            for _ in args.ground_truth_category_prefix
        ) + ")"
        params.extend(prefix + "%" for prefix in args.ground_truth_category_prefix)
    params.extend(version_params)
    if args.limit is not None:
        params.append(args.limit)

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(%s)", (CATALOG_WRITE_LOCK,))
        register_vector(connection)
        rows = connection.execute(
            """
            SELECT p.id, p.image
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
            for product_id, image_meta in batch:
                image_meta = image_meta or {}
                local_path = image_meta.get("local_path")
                try:
                    image_path = local_asset(ROOT / ".data" / "catalog", local_path) if isinstance(local_path, str) else None
                except ValueError:
                    image_path = None
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
            try:
                inputs = processor(images=[image for _, image in valid], return_tensors="pt")
                with torch.inference_mode():
                    vectors = model.get_image_features(**inputs).cpu().tolist()
                # Image vectors are normalized; Nomic text vectors remain raw.
                normalized = []
                for vector in vectors:
                    norm = sum(value * value for value in vector) ** 0.5
                    normalized.append([value / norm for value in vector])
                store_batch(connection, (product_id for product_id, _ in valid), normalized,
                            modality="image", model=MODEL, revision=digest)
            finally:
                for _, image in valid:
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
