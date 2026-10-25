"""Acquire bounded Amazon metadata and local images without importing it.

Run ``uv run python -m rag.download_catalog .data/catalog-plan.json``. The plan
uses the catalog Manifest contract with empty images and a recorded source-use
decision. This tool never treats public availability as a license decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import socket
import shutil
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .catalog import Manifest, ROOT, normalize, normalize_metadata, attach_image

METADATA_HOSTS = frozenset({"huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co",
                            "cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co"})
IMAGE_HOSTS = frozenset({"m.media-amazon.com", "images-na.ssl-images-amazon.com"})
IMAGE_BUDGET = 50_000_000
IMAGE_LIMIT = 10_000_000
MAX_IMAGE_REQUESTS = 150
WALL_SECONDS = 1800


def validate_url(url: str, hosts: frozenset[str]) -> None:
    """Require HTTPS on allowed public hosts, including every redirect target."""
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.password
            or parsed.port not in (None, 443)):
        raise ValueError("url_not_allowed")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("nonpublic_address")


class SafeRedirects(HTTPRedirectHandler):
    """Revalidate redirect targets and bound the redirect chain."""

    max_redirections = 3
    max_repeats = 1

    def __init__(self, hosts: frozenset[str]):
        """Keep the allowed HTTPS hosts for every redirect in this handler."""
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Validate the new destination before urllib creates its next request."""
        validate_url(newurl, self.hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_remote(url: str, hosts: frozenset[str]):
    """Open a validated HTTPS stream with no ambient proxy, cookies or credentials.

    Socket operations time out after 15 seconds. Content encoding is deliberately
    identity so the metadata byte budget also bounds decompressed input.
    """
    validate_url(url, hosts)
    opener = build_opener(ProxyHandler({}), SafeRedirects(hosts))
    response = opener.open(Request(url, headers={"Accept-Encoding": "identity",
                                               "User-Agent": "LocalApparelCatalog/1.0"}), timeout=15)
    if response.headers.get("Content-Encoding", "identity").lower() not in {"identity", ""}:
        response.close()
        raise ValueError("unexpected_content_encoding")
    return response


def ordered_images(value: Any) -> list[str]:
    """Select at most three unique MAIN-first, hi-res-first image candidates."""
    rows = value if isinstance(value, list) else []
    if isinstance(value, dict):
        # Some dataset loaders transpose a sequence of structs into arrays.
        lengths = [len(v) for v in value.values() if isinstance(v, list)]
        rows = [{key: vals[i] if isinstance(vals, list) and i < len(vals) else vals
                 for key, vals in value.items()} for i in range(min(max(lengths, default=0), 100))]
    rows = [row for row in rows if isinstance(row, dict)]
    rows.sort(key=lambda row: row.get("variant") != "MAIN")
    urls: list[str] = []
    for row in rows:
        for key in ("hi_res", "large"):
            url = row.get(key)
            if isinstance(url, str) and url not in urls and urlparse(url).hostname in IMAGE_HOSTS:
                urls.append(url)
    return urls[:3]


def fetch_image(url: str, budget: dict[str, Any], deadline: float) -> tuple[bytes, str]:
    """Download one bounded image, charging actual response body bytes to budget."""
    from PIL import Image
    from io import BytesIO

    if time.monotonic() >= deadline:
        raise ValueError("wall_limit")
    remaining = budget.get("image_byte_limit", IMAGE_BUDGET) - budget["image_bytes"]
    if remaining <= 0 or budget["image_requests"] >= budget.get("image_request_limit", MAX_IMAGE_REQUESTS):
        raise ValueError("image_budget")
    budget["image_requests"] += 1
    # Serialized requests with a small minimum interval, and no automatic retry.
    time.sleep(0.1)
    if time.monotonic() >= deadline:
        raise ValueError("wall_limit")
    with open_remote(url, IMAGE_HOSTS) as response:
        mime = response.headers.get("Content-Type", "").split(";")[0].lower()
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("image_mime")
        limit = min(IMAGE_LIMIT, remaining)
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > limit:
            raise ValueError("image_byte_limit")
        chunks: list[bytes] = []
        total = 0
        while total < limit:
            if time.monotonic() >= deadline:
                raise ValueError("wall_limit")
            chunk = response.read(min(65536, limit - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            budget["image_bytes"] += len(chunk)
        if total == limit:
            # Conservatively reject a full budget buffer, avoiding an extra read.
            raise ValueError("image_byte_limit")
    content = b"".join(chunks)
    with Image.open(BytesIO(content)) as image:
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("image_format")
        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image.format]
    return content, extension


def _download_product_image(row: dict[str, Any], product: dict[str, Any], plan: Manifest,
                            root: Path, report: dict[str, Any], deadline: float,
                            seen_hashes: set[str]) -> dict[str, Any] | None:
    """Try ordered image candidates, recording bounded failures and stopping on 429.

    Inputs share the acquisition budget and de-duplication set. Returns a
    normalized product with a validated local image, or None after all candidates
    fail. Only files created by the failed attempt are removed.
    """
    asin = product['source_product_id']
    for url in ordered_images(row.get("images")):
        asset_path = None
        try:
            content, extension = fetch_image(url, report, deadline)
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen_hashes:
                raise ValueError("duplicate_image")
            asset_path = root / "images" / (digest + extension)
            asset_path.write_bytes(content)
            asset = {"local_path": "images/" + asset_path.name, "source_url": url}
            candidate = plan.model_copy(update={"images": {asin: asset}})
            accepted = attach_image(product, row, candidate, root)
            seen_hashes.add(digest)
            return accepted
        except (ValueError, OSError, URLError) as exc:
            if asset_path is not None:
                asset_path.unlink(missing_ok=True)
            error = "http_" + str(exc.code) if isinstance(exc, HTTPError) else type(exc).__name__
            if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)):
                error = str(exc)
            report["image_failures"][error] += 1
            if isinstance(exc, HTTPError) and exc.code == 429:
                report["stop_reason"] = "rate_limited"
                break
    return None


def download_catalog(plan: Manifest, root: Path, *, seed_root: Path | None = None) -> dict[str, Any]:
    """Scan a pinned source and write bounded importer assets, optionally extending a seed.

    Refuses to overwrite an existing catalog. Reports each rejected source line and
    original accepted line numbers. No database connection or model is used.
    """
    expected = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/"
                + plan.revision + "/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl")
    if not re.fullmatch(r"[0-9a-f]{40}", plan.revision) or plan.source_url != expected:
        raise ValueError("Use the exact pinned clothing metadata URL and a 40-character commit")
    if plan.rights_status not in {"permitted", "personal_use"} or plan.images or plan.source_lines:
        raise ValueError("Acquisition requires a permitted or personal-use source decision, empty image/line maps")
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ("manifest.json", "records.jsonl", "pilot-report.json", "records.partial", "rejections.jsonl", "images")):
        raise FileExistsError("Existing catalog artifacts found; use the catalog maintenance command to extend them")
    (root / "images").mkdir()
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    report: dict[str, Any] = {
        "source_url": plan.source_url, "revision": plan.revision,
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "scanned": 0, "metadata_bytes": 0, "image_bytes": 0, "image_requests": 0,
        "accepted": 0, "rejected": Counter(), "image_failures": Counter(),
        "categories": Counter(), "stop_reason": "eof", "target": plan.max_products,
        "image_byte_limit": max(IMAGE_BUDGET, min(500_000_000, plan.max_products * 250_000)),
        "image_request_limit": max(MAX_IMAGE_REQUESTS, plan.max_products * 3), "seeded": 0,
    }
    images: dict[str, dict[str, str]] = {}
    source_lines: dict[str, int] = {}
    seen_ids: set[str] = set()
    seen_titles: set[tuple[str, str | None]] = set()
    seen_hashes: set[str] = set()
    samples: list[dict[str, Any]] = []
    seed_lines: list[bytes] = []
    if seed_root is not None:
        seed = Manifest.model_validate_json((seed_root / "manifest.json").read_text(encoding="utf-8"))
        operational = {"images", "source_lines", "max_products", "max_lines", "max_bytes"}
        if seed.model_dump(exclude=operational) != plan.model_dump(exclude=operational):
            raise ValueError("Seed provenance and curation must match the expansion plan")
        for line in (seed_root / seed.records).read_bytes().splitlines(keepends=True):
            row = json.loads(line)
            product = normalize(row, seed, seed_root)
            asin = product["source_product_id"]
            asset = seed.images[asin]
            source = (seed_root / asset["local_path"]).resolve()
            target = root / "images" / source.name
            shutil.copy2(source, target)
            images[asin] = {**asset, "local_path": "images/" + target.name}
            source_lines[asin] = seed.source_lines[asin]
            seen_ids.add(asin)
            seen_titles.add((" ".join(product["title"].casefold().split()), product["brand"]))
            seen_hashes.add(hashlib.sha256(source.read_bytes()).hexdigest())
            seed_lines.append(line if line.endswith(b"\n") else line + b"\n")
            report["accepted"] += 1
            report["categories"][product["category"]] += 1
        report["seeded"] = len(seed_lines)
        if len(seed_lines) > plan.max_products:
            raise ValueError("Seed exceeds target")
    try:
        with open_remote(plan.source_url, METADATA_HOSTS) as response, (root / "records.partial").open("wb") as records, (root / "rejections.jsonl").open("w", encoding="utf-8") as rejected:
            records.writelines(seed_lines)
            mime = response.headers.get("Content-Type", "").split(";")[0].lower()
            if mime not in {"application/octet-stream", "application/json", "application/x-ndjson",
                            "text/plain", "text/x-ndjson"}:
                raise ValueError("metadata_mime")
            while report["scanned"] < plan.max_lines and report["accepted"] < plan.max_products:
                if time.monotonic() >= deadline:
                    report["stop_reason"] = "wall_limit"
                    break
                if report["image_bytes"] >= report["image_byte_limit"] or report["image_requests"] >= report["image_request_limit"]:
                    report["stop_reason"] = "image_budget"
                    break
                remaining = plan.max_bytes - report["metadata_bytes"]
                if remaining <= 0:
                    report["stop_reason"] = "metadata_byte_limit"
                    break
                line = response.readline(min(1_000_000, remaining))
                report["metadata_bytes"] += len(line)
                if not line:
                    break
                if not line.endswith(b"\n"):
                    if len(line) == 1_000_000 and remaining >= 1_000_000:
                        # Skip an oversized source record without allocating it.
                        # Continue counting its tail against the same byte/time limits.
                        report["scanned"] += 1
                        report["rejected"]["oversized_record"] += 1
                        rejected.write(json.dumps({"line": report["scanned"], "product_id": None,
                                                   "reason": "oversized_record"}) + "\n")
                        tail = line
                        while tail and not tail.endswith(b"\n"):
                            left = plan.max_bytes - report["metadata_bytes"]
                            if left <= 0 or time.monotonic() >= deadline:
                                break
                            tail = response.readline(min(65536, left))
                            report["metadata_bytes"] += len(tail)
                        if tail.endswith(b"\n"):
                            continue
                    report["stop_reason"] = "incomplete_line_or_byte_limit"
                    break
                report["scanned"] += 1
                asin = None
                reason = None
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record_shape")
                    product = normalize_metadata(row, plan.min_description_words, plan.quality_policy)
                    asin = product["source_product_id"]
                    if asin in seen_ids:
                        if seed_root is not None:
                            continue
                        raise ValueError("duplicate_id")
                    title_key = (" ".join(product["title"].casefold().split()), product["brand"])
                    if title_key in seen_titles:
                        raise ValueError("duplicate_title_brand")
                    accepted = _download_product_image(row, product, plan, root, report, deadline, seen_hashes)
                    if accepted is None:
                        raise ValueError("no_valid_image")
                    images[asin] = {key: accepted['image'].obj[key] for key in ('local_path', 'source_url')}
                    seen_ids.add(asin)
                    seen_titles.add(title_key)
                    records.write(line)
                    source_lines[asin] = report["scanned"]
                    report["accepted"] += 1
                    report["categories"][product["category"]] += 1
                    samples.append({"product_id": asin, "source_line": report["scanned"],
                                    "description_words": len(re.findall(r"\b\w+\b", product["description"])),
                                    "attribute_keys": sorted(product["attributes"].obj["attributes"]),
                                    "has_price": product["historical_price_usd"] is not None,
                                    "image_width": accepted["image"].obj["width"],
                                    "image_height": accepted["image"].obj["height"]})
                except (ValueError, UnicodeDecodeError) as exc:
                    reason = str(exc) if re.fullmatch(r"[a-z_]+", str(exc)) else "invalid_record"
                    report["rejected"][reason] += 1
                    rejected.write(json.dumps({"line": report["scanned"], "product_id": asin, "reason": reason}) + "\n")
                if report["stop_reason"] == "rate_limited":
                    break
            else:
                report["stop_reason"] = "product_limit" if report["accepted"] >= plan.max_products else "line_limit"
        (root / "records.partial").replace(root / "records.jsonl")
        final = plan.model_copy(update={"images": images, "source_lines": source_lines})
        (root / "manifest.json").write_text(final.model_dump_json(indent=2) + "\n", encoding="utf-8")
        report["status"] = "complete" if report["accepted"] == plan.max_products else "partial"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        if isinstance(exc, HTTPError):
            report["http_status"] = exc.code
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["deficit"] = plan.max_products - report["accepted"]
        report["samples"] = samples
        (root / "pilot-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Read a local source-use plan and write a bounded catalog under .data/catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Manifest-shaped plan with approved source evidence and empty images")
    args = parser.parse_args()
    if args.plan.stat().st_size > 2_000_000:
        parser.error("Plan exceeds 2 MB")
    plan = Manifest.model_validate_json(args.plan.read_text(encoding="utf-8"))
    report = download_catalog(plan, ROOT / ".data" / "catalog")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
