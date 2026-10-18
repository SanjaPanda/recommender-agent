"""Fetch a reproducible, bounded sample of Shopify catalogue rows and images."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Image, load_dataset

DATASET_ID = "Shopify/product-catalogue"
DATASET_REVISION = "002ca15536bb015ebc47b99cc2d716d7040bb09e"
FIELDS = (
    "product_title",
    "product_description",
    "product_image",
    "potential_product_categories",
    "ground_truth_brand",
    "ground_truth_is_secondhand",
    "ground_truth_category",
)


def is_missing(value: Any) -> bool:
    """Return whether a source value is empty without treating zero as missing."""
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (list, tuple, dict)) and not value
    )


def image_extension(content: bytes) -> str:
    """Guess a local image filename extension from its byte signature."""
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def identity_key(row: dict[str, Any]) -> str:
    """Hash source content without tying identity to a sample-relative image path."""
    source = {key: value for key, value in row.items() if key != "id"}
    image = source.get("product_image")
    if isinstance(image, dict):
        content = image.get("bytes")
        digest = image.get("sha256")
        if digest is None and content:
            digest = hashlib.sha256(content).hexdigest()
        source["product_image"] = {"sha256": digest} if digest else None
    canonical = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def existing_sample_identities(path: Path) -> dict[str, str]:
    """Map content-derived identities to IDs already present in a sample file."""
    if not path.is_file():
        return {}
    identities = {}
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            row = json.loads(line)
            identities[identity_key(row)] = row["id"]
    return identities


def main() -> None:
    """Stream a pinned Shopify dataset sample and save JSONL rows and images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=250, help="Maximum matching rows to fetch")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".data/shopify-catalogue-sample")
    )
    parser.add_argument(
        "--category-prefix", action="append", default=[],
        help="Keep rows whose ground_truth_category equals or starts with this prefix; repeatable",
    )
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be at least 1")

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_root = image_dir.parent
    dataset = load_dataset(DATASET_ID, split=args.split, streaming=True, revision=DATASET_REVISION)
    dataset = dataset.cast_column("product_image", Image(decode=False))
    existing_identities = (
        existing_sample_identities(Path(".data/shopify-catalogue-sample/products.jsonl"))
        if args.category_prefix
        else {}
    )

    missing = Counter()
    titles = Counter()
    image_count = 0
    image_bytes = 0
    rows_read = 0
    rows_scanned = 0
    products_path = output_dir / "products.jsonl"

    with products_path.open("w", encoding="utf-8", newline="\n") as products_file:
        for row in dataset:
            if rows_read >= args.rows:
                break
            rows_scanned += 1
            if rows_scanned % 500 == 0:
                print(f"Scanned {rows_scanned}; saved {rows_read} ({args.split})", flush=True)
            ground_truth_category = row.get("ground_truth_category") or ""
            if args.category_prefix and not any(
                ground_truth_category.casefold() == prefix.casefold()
                or ground_truth_category.casefold().startswith(prefix.casefold() + " > ")
                for prefix in args.category_prefix
            ):
                continue
            rows_read += 1
            for field in FIELDS:
                if is_missing(row.get(field)):
                    missing[field] += 1

            title = row.get("product_title")
            if isinstance(title, str) and title.strip():
                titles[" ".join(title.casefold().split())] += 1

            source = dict(row)
            image = source.get("product_image")
            image_meta = None
            if isinstance(image, dict):
                content = image.get("bytes")
                if content is None and image.get("path"):
                    image_path = Path(image["path"])
                    if image_path.is_file():
                        content = image_path.read_bytes()
                if content:
                    digest = hashlib.sha256(content).hexdigest()
                    filename = f"{rows_read:06d}-{digest[:16]}{image_extension(content)}"
                    (image_dir / filename).write_bytes(content)
                    image_meta = {
                        "local_path": (Path("images") / filename).as_posix(),
                        "sha256": digest,
                        "byte_length": len(content),
                    }
                    image_count += 1
                    image_bytes += len(content)
                else:
                    image_meta = {"available": False}
            source["product_image"] = image_meta

            canonical = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if args.category_prefix:
                product_id = existing_identities.get(identity_key(source)) or identity_key(source)
            else:
                product_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            products_file.write(
                json.dumps({"id": product_id, **source}, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    duplicate_title_candidates = sum(count - 1 for count in titles.values() if count > 1)
    report = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": args.split,
        "rows_requested": args.rows,
        "rows_saved": rows_read,
        "rows_scanned": rows_scanned,
        "images_saved": image_count,
        "image_bytes_saved": image_bytes,
        "missing_counts": {field: missing[field] for field in FIELDS},
        "duplicate_title_candidates": duplicate_title_candidates,
        "files": {
            "products": products_path.relative_to(image_root).as_posix(),
            "images": "images/",
        },
        "source_schema_fields": list(FIELDS),
        "ground_truth_category_prefixes": args.category_prefix,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "The selected split was streamed up to the requested matching-row limit or its end.",
            "Image bytes are saved as local files; product_image contains their relative path and SHA-256.",
            "Duplicate title matches are candidates, not proof that records describe the same product.",
            "The source schema has no documented price field or stable product identifier.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
