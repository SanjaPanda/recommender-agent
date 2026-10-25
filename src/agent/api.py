"""Local FastAPI interface; the Next.js web UI runs separately in Compose.

Run ``uv run recommender-agent-api`` from the repository root after database
setup in docs/SETUP.md. Requests persist sessions; GET /api/health is read-only.
Regression checks: ``uv run python -m unittest discover -s tests -v``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4
from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .a2ui import product_gallery_operations
from .config import settings
from .models import CategoryPrefixes
from rag import APPAREL_CATEGORY
from .sessions import SessionBusyError, run_recommendation_turn, sanitize_apparel_products
from .product_evidence import listing_checks, load_products, refresh_product_facts

ROOT = settings.model_config["env_file"].parent
app = FastAPI(title="Product Recommender Agent", version="0.1.0")
logger = logging.getLogger(__name__)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class DiscoverRequest(BaseModel):
    """Validate a user's initial product-discovery query and optional thread."""

    query: str = Field(default="", max_length=1000, description="Apparel search request; optional for explicit category browsing.")
    category_prefixes: CategoryPrefixes | None = None
    thread_id: str | None = Field(default=None, max_length=128, description="Existing session ID, if continuing a thread.")


class RecommendRequest(BaseModel):
    """Validate selected examples and free-text refinement for a follow-up."""

    thread_id: str = Field(min_length=1, max_length=128, description="Existing recommendation session ID.")
    selected_product_ids: list[str] | None = Field(default=None, max_length=3, description="Displayed cards to use as visual examples.")
    refinement: str = Field(default="", max_length=1000, description="Preference update for the current session.")


class SessionRequest(BaseModel):
    """Create or resume a persistent shopping session."""
    thread_id: str | None = Field(default=None, max_length=128, description="Session to resume, or none to create a new one.")


class CartRequest(BaseModel):
    """Explicit cart actions; selecting an example never mutates the cart."""
    thread_id: str = Field(min_length=1, max_length=128, description="Session whose cart should change.")
    action: Literal["cart_add", "cart_remove", "cart_show"] = Field(description="Explicit cart operation.")
    product_ids: list[str] = Field(default_factory=list, max_length=20, description="Displayed product IDs for add/remove operations.")


def _find_image_path(image_meta: Any) -> Path | None:
    """Resolve a safe local image under the active catalog directory."""
    if not isinstance(image_meta, dict):
        return None
    relative = image_meta.get("local_path")
    if not isinstance(relative, str):
        return None
    from rag.catalog import local_asset
    try:
        return local_asset(ROOT / ".data" / "catalog", relative)
    except ValueError:
        return None



def _product_image_path(product_id: str) -> Path | None:
    """Resolve current apparel image metadata; never serve a stale cached path."""
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
        row = connection.execute("SELECT image FROM products WHERE id = %s AND "
                                 "(raw_source->>'ground_truth_category'=%s OR "
                                 "starts_with(raw_source->>'ground_truth_category',%s))",
                                 (product_id, APPAREL_CATEGORY, APPAREL_CATEGORY + ' > ')).fetchone()
    return _find_image_path(row[0]) if row else None


def _serialize_products(items: list[dict[str, Any]], query: str = "", filters: dict[str, Any] | None = None,
                        product_phrase: str | None = None) -> list[dict[str, Any]]:
    """Serialize current apparel facts and source details in one database batch."""
    products = []
    facts, listings = {}, {}
    if items:
        with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
            facts, listings = load_products(connection, [item['id'] for item in items])
    for item in items:
        if item['id'] not in listings:
            continue
        fresh = facts[item['id']]
        item, _ = refresh_product_facts(item, fresh)
        image_path = _find_image_path(item.get("image_meta"))
        product = {key: value for key, value in item.items() if key != "image_meta"}
        product["title"] = product.get("title") or "Untitled product"
        product["image_url"] = f"http://127.0.0.1:8000/api/products/{item['id']}/image" if image_path else None
        product["listing"] = listings[item["id"]]
        product["match_checks"] = listing_checks(product["title"], product["listing"]["details"], query, filters or {}, product_phrase)
        products.append(product)
    return products


def _response(
    state: dict[str, Any], thread_id: str, heading: str, *, create_surface: bool
) -> dict[str, Any]:
    """Build the JSON response and A2UI operations for one agent turn."""
    intent = state.get("search_intent")
    query = state.get("plan", {}).get("query", state.get("query_context", ""))
    products = _serialize_products(state.get("recommendations", []), query,
                                  state.get("plan", {}).get("filters", {}),
                                  intent.get("product_phrase", "") if intent is not None else None)
    surface_id = f"product-recommendations-{thread_id}"
    if state.get("result_kind") == "fallback":
        heading = "Closest alternatives"
    return {
        "thread_id": thread_id,
        "a2ui_surface_id": surface_id,
        "query_context": query,
        "search_intent": intent,
        "selected_product_ids": state.get("selected_product_ids", []),
        "removed_filters": state.get("removed_filters", []),
        "products": products,
        "status": state.get("status", "ok"),
        "message": state.get("message", ""),
        "categories": state.get("categories", []),
        "cart": state.get("cart", []),
        "preferences": state.get("plan", {}).get("preferences", []),
        "filters": state.get("plan", {}).get("filters", {}),
        "trace": [event.model_dump() for event in state.get("trace", [])],
        "a2ui_operations": product_gallery_operations(
            products, heading=heading, surface_id=surface_id, create_surface=create_surface
        ),
    }


def _search_response(thread_id: str, turn: dict[str, Any], heading: str,
                     *, create_surface: bool) -> dict[str, Any]:
    """Run discovery/refinement with their shared HTTP error contract.

    Args:
        thread_id: Persistent shopping session identifier.
        turn: Initial query/categories or validated refinement/example fields.
        heading: Default gallery heading, overridden for closest alternatives.
        create_surface: Whether A2UI must replace/create the current surface.

    Returns:
        Serialized product facts, session context and A2UI operations.

    Raises:
        HTTPException: 409 for overlap/stale state; 503 for database, model or
            search validation failures. Explicit cart requests keep their
            separate 400 invalid-action contract.
    """
    try:
        state = run_recommendation_turn(thread_id=thread_id, turn=turn)
        return _response(state, thread_id, heading, create_surface=create_surface)
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except psycopg.Error as exc:
        logger.exception("Database operation failed during recommendation")
        raise HTTPException(status_code=503, detail="Catalogue database is unavailable. Start PostgreSQL and retry.") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    """Return a small liveness response for the local API."""
    return {"status": "ok"}


@app.post("/api/discover")
def discover(request: DiscoverRequest) -> dict[str, Any]:
    """Start or continue a recommendation thread from a free-text request."""
    thread_id = request.thread_id or str(uuid4())
    return _search_response(thread_id, {"query": request.query,
        "category_prefixes": request.category_prefixes}, "Choose products you like", create_surface=True)


@app.post("/api/session")
def session(request: SessionRequest) -> dict[str, Any]:
    """Load saved cards/cart, or show categories on a new session without an LLM."""
    thread_id = request.thread_id or str(uuid4())
    try:
        with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
            saved = connection.execute("SELECT state FROM agent_sessions WHERE thread_id=%s", (thread_id,)).fetchone()
            state = sanitize_apparel_products(connection, saved[0]) if saved else None
        if state is not None:
            from agent.tools.retrieval import list_categories
            state = {**state, "categories": list_categories()}
        else:
            state = run_recommendation_turn(thread_id=thread_id, turn={})
        return _response(state, thread_id, "Recommended for you", create_surface=True)
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (psycopg.Error, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Could not load session; check database and migrations.") from exc


@app.post("/api/cart")
def cart(request: CartRequest) -> dict[str, Any]:
    """Persist a cart action and remove cart items from displayed recommendations."""
    try:
        state = run_recommendation_turn(thread_id=request.thread_id,
            turn={"action": request.action, "product_ids": request.product_ids})
        return _response(state, request.thread_id, "Recommended for you", create_surface=True)
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Could not update cart.") from exc


@app.post("/api/recommend")
def recommend(request: RecommendRequest) -> dict[str, Any]:
    """Use selected products and refinement text to update recommendations."""
    if request.selected_product_ids and len(set(request.selected_product_ids)) != len(request.selected_product_ids):
        raise HTTPException(status_code=422, detail="Selected product IDs must be unique.")
    if not request.selected_product_ids and not request.refinement.strip():
        raise HTTPException(status_code=422, detail="Select a product or add a preference.")
    turn: dict[str, Any] = {}
    if request.selected_product_ids is not None:
        turn["selected_product_ids"] = request.selected_product_ids
    if request.refinement.strip():
        turn["refinement"] = request.refinement.strip()
    return _search_response(request.thread_id, turn, "Recommended for you", create_surface=False)


@app.get("/api/products/{product_id}/image")
def product_image(product_id: str) -> FileResponse:
    """Serve a product image only when it resolves inside the local catalog."""
    path = _product_image_path(product_id)
    if not path:
        raise HTTPException(status_code=404, detail="Product image is not available locally.")
    return FileResponse(path)


def main() -> None:
    """Start the loopback API on port 8000; Compose runs the separate web UI."""
    import uvicorn

    # The local browser calls the loopback API; containers do not proxy session data.
    uvicorn.run("agent.api:app", host="127.0.0.1", port=8000, reload=False)
