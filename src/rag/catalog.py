"""Normalize and import a bounded local Amazon metadata pilot.

Run ``uv run python -m rag.catalog .data/catalog/manifest.json`` after preparing
source metadata and local images. This module never acquires remote data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from agent.config import ROOT, settings
from . import CATALOG_WRITE_LOCK

ROOT_CATEGORY = "Apparel & Accessories"
SOURCE_ROOTS = {"Clothing, Shoes & Jewelry", "Clothing_Shoes_and_Jewelry"}


class Manifest(BaseModel):
    """Versioned local source decision, bounded scan and image inventory.

    Paths are relative to the manifest directory. Image entries are indexed by
    parent_asin and contain local_path and the selected upstream source_url.
    """

    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    source_name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    rights_status: Literal["permitted", "personal_use", "authored"]
    rights_evidence: str = Field(min_length=1)
    records: str
    images: dict[str, dict[str, str]]
    source_lines: dict[str, PositiveInt] = Field(default_factory=dict)
    max_products: int = Field(default=2000, ge=1, le=2000)
    max_lines: int = Field(default=100000, ge=1, le=100000)
    max_bytes: int = Field(default=250_000_000, ge=1, le=250_000_000)
    quality_policy: Literal["evidence-v2"] | None = None
    min_description_words: int = Field(default=50, ge=50)

    @model_validator(mode="after")
    def require_pinned_source(self) -> "Manifest":
        """Reject blank provenance and mutable revision aliases."""
        for value in (self.source_id, self.source_name, self.revision, self.source_url, self.rights_evidence):
            if not value.strip():
                raise ValueError("Source provenance must not be blank")
        if self.revision.lower() in {"main", "master", "latest"}:
            raise ValueError("Use an immutable source revision")
        return self


def local_asset(root: Path, relative: str) -> Path:
    """Resolve a local file within root, rejecting traversal and symlink escapes."""
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root.resolve()):
        raise ValueError("asset_path")
    if not resolved.is_file():
        raise ValueError("missing_asset")
    return resolved


def string_list(value: Any) -> list[str]:
    """Read a source string or string list without coercing objects into text."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    raise ValueError("text_shape")


def image_urls(value: Any) -> set[str]:
    """Read candidate URLs from list-of-objects or dict-of-arrays metadata."""
    urls: set[str] = set()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for key in ("hi_res", "large"):
                    if isinstance(item.get(key), str) and item[key].startswith("https://"):
                        urls.add(item[key])
    elif isinstance(value, dict):
        for key in ("hi_res", "large"):
            items = value.get(key, [])
            if isinstance(items, str):
                items = [items]
            if isinstance(items, list):
                urls.update(item for item in items if isinstance(item, str) and item.startswith("https://"))
    return urls


def historical_price(value: Any) -> Decimal | None:
    """Parse finite nonnegative historical USD amounts, leaving unknowns null."""
    if value is None or isinstance(value, bool):
        return None
    try:
        price = Decimal(str(value))
        if not price.is_finite() or price < 0 or price > Decimal("9999999999.99"):
            return None
        return price.quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def normalize_metadata(row: dict[str, Any], min_description_words: int = 50, quality_policy: str | None = None) -> dict[str, Any]:
    """Validate apparel source fields before spending network or decode work.

    Returns normalized database fields except image and source provenance. Raw
    upstream fields are preserved with the runtime-compatible taxonomy key.
    """
    asin, title = row.get("parent_asin"), row.get("title")
    if not isinstance(asin, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", asin):
        raise ValueError("product_id")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title")
    categories = string_list(row.get("categories", []))
    if row.get("main_category") not in SOURCE_ROOTS and not (
        row.get("main_category") == "AMAZON FASHION" and categories and categories[0] in SOURCE_ROOTS
    ):
        raise ValueError("taxonomy")
    # Only explicit eligible branches are normalized; never map an entire source
    # domain to apparel (it may contain unrelated products).
    branches = {"Clothing": "Clothing", "Shoes": "Shoes", "Jewelry": "Jewelry",
                "Accessories": "Clothing Accessories"}
    branch = next((branches[item] for item in categories if item in branches), None)
    if branch is None:
        raise ValueError("taxonomy_branch")
    category = ROOT_CATEGORY + " > " + branch
    description = "\n\n".join(string_list(row.get("description", [])))
    if len(re.findall(r"\b\w+\b", description)) < min_description_words:
        raise ValueError("short_description")
    features = string_list(row.get("features", []))
    details = row.get("details", {})
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError as exc:
            raise ValueError("details_shape") from exc
    if not isinstance(details, dict) or not any(
        isinstance(value, str) and value.strip() for value in details.values()
    ):
        raise ValueError("missing_attributes")
    if quality_policy == "evidence-v2":
        from .quality import assess
        failures = assess(row, min_description_words)["failures"]
        if failures:
            raise ValueError(failures[0])
    brand = details.get("Brand")
    from .attributes import extract
    return {
        "source_product_id": asin, "title": title.strip(), "description": description,
        "brand": brand.strip() if isinstance(brand, str) and brand.strip() else None,
        "category": category, "raw_source": Jsonb({**row, "ground_truth_category": category}),
        "features": Jsonb(features), "attributes": Jsonb({"schema_version": 1, "attributes": details,
                                                        "extraction": extract(row)}),
        "historical_price_usd": historical_price(row.get("price")),
    }


def normalize(row: dict[str, Any], manifest: Manifest, root: Path) -> dict[str, Any]:
    """Validate source facts and one decoded image; return database parameters.

    Rejects unsupported taxonomy, incomplete descriptions/details and unsafe or
    invalid local images. No model is called and no missing facts are generated.
    """
    from PIL import Image, UnidentifiedImageError

    product = normalize_metadata(row, manifest.min_description_words, manifest.quality_policy)
    asin = product["source_product_id"]
    asset = manifest.images.get(asin, {})
    if asset.get("source_url") not in image_urls(row.get("images")):
        raise ValueError("image_provenance")
    image_path = local_asset(root, asset.get("local_path", ""))
    if image_path.stat().st_size > 10_000_000:
        raise ValueError("image_byte_limit")
    try:
        with Image.open(image_path) as image:
            if min(image.size) < 224 or image.width * image.height > 20_000_000:
                raise ValueError("image_dimensions")
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("image_format")
            image.verify()
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("image_decode") from exc
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return {
        **product, "id": f"{manifest.source_id}:{asin}",
        "image": Jsonb({"local_path": asset["local_path"], "source_url": asset["source_url"],
                        "sha256": image_hash, "width": width, "height": height}),
        "source_name": manifest.source_name, "dataset_source_id": manifest.source_id,
        "image_sha256": image_hash,
    }



def ingest(connection: psycopg.Connection, manifest_path: Path) -> dict[str, Any]:
    """Import a bounded local manifest in the caller's transaction.

    Writes source provenance, idempotent products and a report returned to the
    caller. Changed source records invalidate their stored embeddings. Existing
    source IDs cannot be reused with different revision/curation decisions.
    """
    connection.execute("SELECT pg_advisory_xact_lock(%s)", (CATALOG_WRITE_LOCK,))
    if manifest_path.stat().st_size > 2_000_000:
        raise ValueError("manifest_byte_limit")
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.resolve()
    records = local_asset(root, manifest.records)
    config = manifest.model_dump(exclude={"images", "rights_evidence", "source_lines"}, exclude_none=True)
    provenance = (manifest.source_name, manifest.revision, manifest.source_url,
                  manifest.rights_status, manifest.rights_evidence, config)
    connection.execute(
        "INSERT INTO dataset_sources(id,name,revision,source_url,rights_status,rights_evidence,curation_config) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
        (manifest.source_id, *provenance[:-1], Jsonb(config)),
    )
    stored = connection.execute(
        "SELECT name,revision,source_url,rights_status,rights_evidence,curation_config "
        "FROM dataset_sources WHERE id=%s FOR UPDATE", (manifest.source_id,),
    ).fetchone()
    if stored != provenance:
        operational = {"max_products", "max_lines", "max_bytes"}
        old_config = stored[-1]
        same_policy = ({k: v for k, v in old_config.items() if k not in operational}
                       == {k: v for k, v in config.items() if k not in operational})
        if stored[:-1] != provenance[:-1] or not same_policy or any(
                config[key] < old_config[key] for key in operational):
            raise ValueError("Source ID already has different provenance or curation settings")
        connection.execute("UPDATE dataset_sources SET curation_config=%s WHERE id=%s",
                           (Jsonb(config), manifest.source_id))
    report: dict[str, Any] = {"source_id": manifest.source_id, "scanned": 0, "bytes": 0,
                              "accepted": 0, "rejected": Counter(), "stop_reason": "eof"}
    seen: set[str] = set()
    with records.open("rb") as stream:
        while report["scanned"] < manifest.max_lines and report["accepted"] < manifest.max_products:
            remaining = manifest.max_bytes - report["bytes"]
            if remaining <= 0:
                report["stop_reason"] = "byte_limit"
                break
            line = stream.readline(min(1_000_000, remaining) + 1)
            if not line:
                break
            if len(line) > min(1_000_000, remaining):
                report["stop_reason"] = "line_or_byte_limit"
                break
            report["bytes"] += len(line)
            report["scanned"] += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record_shape")
                product = normalize(row, manifest, root)
                if product["source_product_id"] in seen:
                    raise ValueError("duplicate_id")
            except (ValueError, UnicodeDecodeError) as exc:
                reason = str(exc) if str(exc) in {
                    "product_id", "title", "taxonomy", "taxonomy_branch", "text_shape",
                    "short_description", "details_shape", "missing_attributes", "image_provenance",
                    "asset_path", "missing_asset", "image_byte_limit", "image_dimensions",
                    "image_format", "image_decode", "duplicate_id", "record_shape",
                    "insufficient_product_description", "weak_description_signals", "no_useful_structured_attributes",
                } else "invalid_record"
                report["rejected"][reason] += 1
                continue
            seen.add(product["source_product_id"])
            product["source_line"] = manifest.source_lines.get(product["source_product_id"], report["scanned"])
            product["source_record_sha256"] = hashlib.sha256(line).hexdigest()
            previous = connection.execute(
                "SELECT source_record_sha256,image_sha256 FROM products WHERE id=%s FOR UPDATE",
                (product["id"],),
            ).fetchone()
            if previous and previous != (product["source_record_sha256"], product["image_sha256"]):
                connection.execute("DELETE FROM product_embeddings WHERE product_id=%s", (product["id"],))
            from psycopg import sql
            columns = list(product)
            connection.execute(sql.SQL("INSERT INTO products ({}) VALUES ({}) ON CONFLICT (id) DO UPDATE SET {}").format(
                sql.SQL(",").join(map(sql.Identifier, columns)),
                sql.SQL(",").join(sql.Placeholder(key) for key in columns),
                sql.SQL(",").join(sql.SQL("{}=EXCLUDED.{}").format(sql.Identifier(key), sql.Identifier(key))
                                 for key in columns if key != "id"),
            ), product)
            report["accepted"] += 1
        else:
            report["stop_reason"] = "product_limit" if report["accepted"] >= manifest.max_products else "line_limit"
    report["rejected"] = dict(report["rejected"])
    report["target"] = manifest.max_products
    report["deficit"] = manifest.max_products - report["accepted"]
    return report


def main() -> None:
    """Import local approved metadata; save rejection counts beside the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Local manifest JSON with source decision and image inventory")
    args = parser.parse_args()
    if args.manifest.parent.resolve() != (ROOT / ".data" / "catalog").resolve():
        parser.error("Place the manifest and local assets under .data/catalog for API image serving")
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
        report = ingest(connection, args.manifest)
    output = args.manifest.parent / "ingestion-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
