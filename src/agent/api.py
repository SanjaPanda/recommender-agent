"""HTTP API and local web entry point for the product recommendation agent."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .a2ui import product_gallery_operations
from .workflow import build_recommendation_agent, run_recommendation_turn

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
app = FastAPI(title="Product Recommender Agent", version="0.1.0")
agent = build_recommendation_agent()
IMAGE_PATHS: dict[str, Path] = {}


class DiscoverRequest(BaseModel):
    """Validate a user's initial product-discovery query and optional thread."""

    query: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = None


class RecommendRequest(BaseModel):
    """Validate selected examples and free-text refinement for a follow-up."""

    thread_id: str = Field(min_length=1, max_length=128)
    selected_product_ids: list[str] = Field(default_factory=list, max_length=3)
    refinement: str = Field(default="", max_length=1000)


def _find_image_path(image_meta: Any) -> Path | None:
    """Resolve only local sample image paths under known image directories."""
    if not isinstance(image_meta, dict):
        return None
    relative = image_meta.get("local_path")
    if not isinstance(relative, str):
        return None
    candidate_rel = Path(relative)
    if candidate_rel.is_absolute() or ".." in candidate_rel.parts:
        return None
    roots = [ROOT / ".data" / "shopify-catalogue-sample"]
    roots.extend(path for path in (ROOT / ".data").glob("*/") if path.is_dir())
    for root in roots:
        candidate = (root / candidate_rel).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _product_image_path(product_id: str) -> Path | None:
    """Resolve a cached local image path, with a bounded database fallback."""
    cached = IMAGE_PATHS.get(product_id)
    if cached and cached.is_file():
        return cached
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None
    with psycopg.connect(database_url, connect_timeout=5) as connection:
        row = connection.execute("SELECT image FROM products WHERE id = %s", (product_id,)).fetchone()
    return _find_image_path(row[0]) if row else None


def _serialize_products(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert retrieval rows to API products and register safe local image paths."""
    products = []
    for item in items:
        image_path = _find_image_path(item.get("image_meta"))
        if image_path:
            IMAGE_PATHS[item["id"]] = image_path
        product = {key: value for key, value in item.items() if key != "image_meta"}
        product["title"] = product.get("title") or "Untitled product"
        product["image_url"] = f"/api/products/{item['id']}/image" if image_path else None
        products.append(product)
    return products


def _response(
    state: dict[str, Any], thread_id: str, heading: str, *, create_surface: bool
) -> dict[str, Any]:
    """Build the JSON response and A2UI operations for one agent turn."""
    products = _serialize_products(state.get("recommendations", []))
    surface_id = f"product-recommendations-{thread_id}"
    return {
        "thread_id": thread_id,
        "a2ui_surface_id": surface_id,
        "query_context": state.get("query_context", ""),
        "selected_product_ids": state.get("selected_product_ids", []),
        "products": products,
        "status": state.get("status", "ok"),
        "a2ui_operations": product_gallery_operations(
            products, heading=heading, surface_id=surface_id, create_surface=create_surface
        ),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    """Return a small liveness response for the local API."""
    return {"status": "ok"}


@app.post("/api/discover")
def discover(request: DiscoverRequest) -> dict[str, Any]:
    """Start or continue a recommendation thread from a free-text request."""
    thread_id = request.thread_id or str(uuid4())
    try:
        state = run_recommendation_turn(agent, thread_id=thread_id, turn={"query": request.query, "limit": 5})
        return _response(state, thread_id, "Choose products you like", create_surface=True)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Catalogue database is unavailable. Start PostgreSQL and retry.") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/recommend")
def recommend(request: RecommendRequest) -> dict[str, Any]:
    """Use selected products and refinement text to update recommendations."""
    if len(set(request.selected_product_ids)) != len(request.selected_product_ids):
        raise HTTPException(status_code=422, detail="Selected product IDs must be unique.")
    if not request.selected_product_ids and not request.refinement.strip():
        raise HTTPException(status_code=422, detail="Select a product or add a preference.")
    turn: dict[str, Any] = {"selected_product_ids": request.selected_product_ids, "limit": 5}
    if request.refinement.strip():
        turn["refinement"] = request.refinement.strip()
    try:
        state = run_recommendation_turn(agent, thread_id=request.thread_id, turn=turn)
        return _response(state, request.thread_id, "Recommended for you", create_surface=False)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Catalogue database is unavailable. Start PostgreSQL and retry.") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/products/{product_id}/image")
def product_image(product_id: str) -> FileResponse:
    """Serve a product image only when it resolves inside a local sample root."""
    path = _product_image_path(product_id)
    if not path:
        raise HTTPException(status_code=404, detail="Product image is not available locally.")
    return FileResponse(path)


def main() -> None:
    """Run the local API and built-in web app."""
    import uvicorn

    # The Dockerized web container reaches this host service through host.docker.internal.
    uvicorn.run("agent.api:app", host="0.0.0.0", port=8000, reload=False)
