"""Stateful LangGraph recommendation flow with catalogue retrieval as its RAG step."""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag.retrieval import retrieve_products


class RecommendationState(TypedDict, total=False):
    """Checkpointed state. ``turn`` contains only the latest caller input."""

    turn: dict[str, Any]
    query_history: list[str]
    query_context: str
    selected_product_ids: list[str]
    category_prefixes: list[str]
    limit: int
    image_weight: float
    recommendations: list[dict[str, Any]]
    status: str


def _recommend(state: RecommendationState) -> dict[str, Any]:
    """Merge this turn into session context, then pass that context to retrieval."""
    turn = state.get("turn", {})
    reset = bool(turn.get("reset_context", False))
    history = [] if reset else list(state.get("query_history", []))
    selected = [] if reset else list(state.get("selected_product_ids", []))
    categories = [] if reset else list(state.get("category_prefixes", []))

    new_text = turn.get("query") or turn.get("refinement") or ""
    if new_text.strip():
        history.append(new_text.strip())
    query_context = " ".join(history)

    # A new selection becomes the active visual seed. If omitted, the previous
    # selection stays in context so a short follow-up like “more like this” works.
    if "selected_product_id" in turn:
        selected = [turn["selected_product_id"]]
    elif "selected_product_ids" in turn:
        selected = list(turn["selected_product_ids"])

    if "category_prefixes" in turn:
        categories = list(turn["category_prefixes"])

    limit = int(turn.get("limit", state.get("limit", 5)))
    image_weight = float(turn.get("image_weight", state.get("image_weight", 0.5)))
    recommendations = retrieve_products(
        query_context=query_context,
        selected_product_ids=selected,
        category_prefixes=categories,
        limit=limit,
        image_weight=image_weight,
    )
    return {
        "query_history": history,
        "query_context": query_context,
        "selected_product_ids": selected,
        "category_prefixes": categories,
        "limit": limit,
        "image_weight": image_weight,
        "recommendations": recommendations,
        "status": "ok" if recommendations else "no_candidates",
    }


def build_recommendation_agent(*, checkpointer: Any | None = None):
    """Compile a checkpointed one-node agent; inject a persistent saver if available."""
    graph = StateGraph(RecommendationState)
    graph.add_node("recommend", _recommend)
    graph.add_edge(START, "recommend")
    graph.add_edge("recommend", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_recommendation_turn(
    agent: Any,
    *,
    thread_id: str,
    turn: dict[str, Any],
) -> dict[str, Any]:
    """Invoke one turn in a session. Reuse ``thread_id`` to retain its context."""
    if not thread_id.strip():
        raise ValueError("thread_id must not be empty")
    return agent.invoke(
        {"turn": turn},
        config={"configurable": {"thread_id": thread_id}},
    )
