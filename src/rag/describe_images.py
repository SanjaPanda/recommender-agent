"""Generate cached image captions offline, then embed them with Nomic."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import psycopg
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from tqdm import tqdm

from agent.config import ROOT, settings
from .caption_schema import VisualAttributes
from .embed_text import MODEL as TEXT_MODEL, OLLAMA_URL, document_text, embed_texts, post_json

PROMPT_VERSION = "product-caption-v5"
PROMPT = (
    "Identify only the main item for sale. Return one JSON object matching this schema "
    "exactly, with every key present: "
    '{"product_type": string or null, "colours": [string] or null, "pattern": string or null, '
    '"visible_components": [string] or null, "visible_object_count": integer or null, '
    '"other_visual_details": string or null}. '
    "Use null for unknown fields, including colours or components that are not clearly visible. "
    "Keep values brief and factual. Ignore people, props, other products and background; "
    "count main product objects, not components or sale quantity. Return JSON only."
)
CATEGORY_PREFIX = "Apparel & Accessories"


class CaptionInput(BaseModel):
    """Validated local inputs for one image-caption request."""

    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    image_base64: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0, le=600)


class CaptionMessage(BaseModel):
    """Ollama multimodal user message with one encoded catalog image."""

    role: Literal["user"] = "user"
    content: str = Field(min_length=1)
    images: list[str] = Field(min_length=1, max_length=1)


class CaptionOptions(BaseModel):
    """Bounded generation options for the optional offline vision model."""

    temperature: float = 0
    num_predict: int = Field(default=250, ge=1, le=512)
    num_ctx: int = Field(default=2048, ge=512, le=4096)
    num_gpu: NonNegativeInt | None = None


class OllamaCaptionRequest(BaseModel):
    """Typed Ollama request for a structured apparel image description."""

    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    stream: Literal[False] = False
    keep_alive: str = "5m"
    format: dict[str, object] = Field(description="JSON Schema for VisualAttributes.")
    options: CaptionOptions
    messages: list[CaptionMessage] = Field(min_length=1, max_length=1)


def image_path(meta: dict | None) -> Path | None:
    """Resolve catalogue-relative paths strictly inside local dataset directories."""
    value = (meta or {}).get("local_path")
    if not isinstance(value, str):
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    for root in (ROOT / ".data" / "shopify-selected-train",
                 ROOT / ".data" / "shopify-selected-test"):
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(root.resolve()) and candidate.is_file():
            return candidate
    return None


def describe(image: bytes, model: str, timeout: float, *, diagnostics: dict | None = None) -> str:
    """One offline vision call, never invoked from recommendation requests."""
    if urllib.parse.urlparse(OLLAMA_URL).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Captioning requires local Ollama.")
    caption_input = CaptionInput(
        model=model,
        image_base64=base64.b64encode(image).decode("ascii"),
        timeout_seconds=timeout,
    )
    request_contract = OllamaCaptionRequest(
        model=caption_input.model,
        format=VisualAttributes.model_json_schema(),
        options=CaptionOptions(num_gpu=settings.agent_num_gpu),
        messages=[CaptionMessage(content=PROMPT, images=[caption_input.image_base64])],
    )
    payload = request_contract.model_dump(mode="json", exclude_none=True)
    request = urllib.request.Request(OLLAMA_URL.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=caption_input.timeout_seconds) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            if 500 <= exc.code < 600 and attempt == 0:
                time.sleep(1)
                continue
            raise urllib.error.HTTPError(
                exc.url, exc.code, f"{exc.reason}; response: {detail}", exc.headers, None
            ) from exc
    message = result.get("message") if isinstance(result, dict) else None
    raw_caption = message.get("content") if isinstance(message, dict) else None
    caption = raw_caption.strip() if isinstance(raw_caption, str) else ""
    if diagnostics is not None:
        diagnostics.update({
            "done_reason": result.get("done_reason") if isinstance(result, dict) else None,
            "response_chars": len(raw_caption) if isinstance(raw_caption, str) else None,
            "eval_count": result.get("eval_count") if isinstance(result, dict) else None,
            "eval_duration_ns": result.get("eval_duration") if isinstance(result, dict) else None,
            "response_keys": sorted(result.keys()) if isinstance(result, dict) else None,
        })
    done_reason = result.get("done_reason") if isinstance(result, dict) else None
    response_chars = len(raw_caption) if isinstance(raw_caption, str) else None
    if done_reason == "length":
        raise ValueError(f"Caption was truncated by the model (done_reason='length', response_chars={response_chars}).")
    if not caption:
        return ""
    attributes = VisualAttributes.model_validate_json(caption)
    parts = []
    if attributes.product_type:
        parts.append(f"Product type: {attributes.product_type}.")
    if attributes.colours:
        parts.append("Colours: " + ", ".join(attributes.colours) + ".")
    if attributes.pattern:
        parts.append(f"Pattern: {attributes.pattern}.")
    if attributes.visible_components:
        parts.append("Visible components: " + ", ".join(attributes.visible_components) + ".")
    if attributes.visible_object_count is not None:
        parts.append(f"Visible product objects: {attributes.visible_object_count}.")
    if attributes.other_visual_details:
        parts.append(attributes.other_visual_details)
    return " ".join(parts)


def validate_vector(vector: list[float]) -> list[float]:
    """Reject malformed vectors before saving them as searchable records."""
    if not vector or not all(math.isfinite(v) for v in vector) or not any(vector):
        raise ValueError("Embedding must be finite and nonzero.")
    return vector


def installed_digest(model: str, models: list[dict]) -> str:
    qualified = model if ":" in model else model + ":latest"
    for entry in models:
        if entry["name"] == qualified:
            return entry["digest"]
    raise ValueError(f"Model {model!r} is not installed in Ollama.")


def summarize_run_status(products: list[dict]) -> str:
    """Distinguish a clean run from recorded item errors and empty outputs."""
    statuses = {product.get("status") for product in products}
    if "failed" in statuses:
        return "completed_with_errors"
    if "empty_caption" in statuses:
        return "completed_with_issues"
    return "success"


def caption_matches_source(caption: str | None, provenance: dict, model_digest: str,
                          image_sha256: str) -> bool:
    """A saved empty caption is still a completed inference outcome for this source."""
    return caption is not None and all(provenance.get(key) == value for key, value in {
        "model_digest": model_digest, "image_sha256": image_sha256,
        "prompt_version": PROMPT_VERSION}.items())


def main() -> None:
    """Generate resumable Pydantic-validated apparel captions offline.

    Run ``uv run recommender-describe-images --limit 8`` for a small batch or
    add ``--all`` after reviewing the expected local inference cost.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=settings.image_description_model)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=8, help="Maximum pending products; default is a small pilot")
    group.add_argument("--all", action="store_true", help="Process the complete catalogue offline")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--report", type=Path, default=ROOT / ".data/caption-pilot.json")
    parser.add_argument("--results", type=Path, help="Append per-product outcomes as JSONL (default: report name with .jsonl)")
    args = parser.parse_args()
    if args.results is None:
        args.results = args.report.with_suffix(".jsonl")
    if args.limit < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("limit and timeout must be positive")
    with urllib.request.urlopen(OLLAMA_URL.rstrip("/") + "/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    digest = installed_digest(args.model, models)
    embedding_digest = installed_digest(TEXT_MODEL, models)
    report = {"model": args.model, "model_digest": digest, "embedding_model": TEXT_MODEL,
              "category_prefix": CATEGORY_PREFIX,
              "num_gpu": settings.agent_num_gpu if settings.agent_num_gpu is not None else "auto",
              "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "products": []}
    started = time.perf_counter()
    with psycopg.connect(settings.database_url.get_secret_value(), autocommit=True, connect_timeout=5) as conn:
        register_vector(conn)
        rows = conn.execute("""
            SELECT p.id,p.image,p.image_description,p.image_description_meta,
                   e.model_version,e.status
            FROM products p LEFT JOIN product_embeddings e
              ON e.product_id=p.id AND e.modality='image_description' AND e.model_name=%s
            WHERE p.image IS NOT NULL
              AND (p.raw_source->>'ground_truth_category' = %s
                   OR starts_with(p.raw_source->>'ground_truth_category', %s))
            ORDER BY p.id
        """, (TEXT_MODEL, CATEGORY_PREFIX, CATEGORY_PREFIX + " > ")).fetchall()
        attempted = 0
        for pid, meta, caption, provenance, embedding_version, status in tqdm(rows, desc="Image captions", unit="product"):
            if not args.all and attempted >= args.limit:
                break
            result = {"id": pid, "caption_seconds": 0.0, "embedding_seconds": 0.0,
                      "caption_diagnostics": {}}
            attempt_started = time.perf_counter()
            counted = False
            try:
                path = image_path(meta)
                if path is None:
                    raise ValueError("Image file is missing or its path is unsafe.")
                raw = path.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                provenance = provenance or {}
                current = caption_matches_source(caption, provenance, digest, sha)
                # Empty captions are saved outcomes too; don't spend inference time retrying them.
                if current and caption == "" and not args.force:
                    continue
                if current and embedding_version == embedding_digest and status == "ready" and not args.force:
                    continue
                attempted += 1
                counted = True
                if not current or args.force:
                    tick = time.perf_counter()
                    try:
                        caption = describe(raw, args.model, args.timeout,
                                           diagnostics=result["caption_diagnostics"])
                    finally:
                        result["caption_seconds"] = time.perf_counter() - tick
                    provenance = {"model": args.model, "model_digest": digest,
                        "image_sha256": sha, "prompt_version": PROMPT_VERSION,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "generation_seconds": result["caption_seconds"]}
                    # Save the caption even if embedding fails; never retain a stale vector.
                    with conn.transaction():
                        conn.execute("UPDATE products SET image_description=%s,image_description_meta=%s WHERE id=%s",
                                     (caption, Jsonb(provenance), pid))
                        conn.execute("DELETE FROM product_embeddings WHERE product_id=%s AND modality='image_description'", (pid,))
                if not caption:
                    result.update(status="empty_caption", caption="")
                else:
                    tick = time.perf_counter()
                    vector = validate_vector(embed_texts(document_text(None, caption))[0])
                    result["embedding_seconds"] = time.perf_counter() - tick
                    conn.execute("""
                        INSERT INTO product_embeddings
                            (product_id,modality,model_name,model_version,embedding,status,error)
                        VALUES (%s,'image_description',%s,%s,%s,'ready',NULL)
                        ON CONFLICT (product_id,modality,model_name) DO UPDATE SET
                            model_version=EXCLUDED.model_version,embedding=EXCLUDED.embedding,
                            status='ready',error=NULL,updated_at=now()
                    """, (pid, TEXT_MODEL, embedding_digest, Vector(vector)))
                    result.update(status="ready", caption=caption)
            except Exception as exc:
                if not counted:
                    attempted += 1
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            result["elapsed_seconds"] = time.perf_counter() - attempt_started
            report["products"].append(result)
            report["elapsed_seconds"] = time.perf_counter() - started
            args.results.parent.mkdir(parents=True, exist_ok=True)
            with args.results.open("a", encoding="utf-8") as results_file:
                results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                results_file.flush()
                os.fsync(results_file.fileno())
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures = sum(p["status"] == "failed" for p in report["products"])
    report["status"] = summarize_run_status(report["products"])
    report["elapsed_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Run status: {report['status']}; processed {len(report['products'])} pending products in {report['elapsed_seconds']:.1f}s. Report: {args.report}; results: {args.results}", flush=True)
    if any(p["status"] == "failed" for p in report["products"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
