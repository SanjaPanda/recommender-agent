"""Run a local interactive recommendation session backed by the LangGraph agent."""
from __future__ import annotations

import argparse
import json
import uuid

from .workflow import build_recommendation_agent, run_recommendation_turn


def _show(state: dict) -> None:
    """Print the current query context and ranked product candidates."""
    recommendations = state.get("recommendations", [])
    print(state.get("message", ""))
    trace = state.get("trace", [])
    if trace:
        print("\nAgent trace:")
        for event in trace:
            print(f"- {event.node}: {event.duration_ms} ms"
                  f" | route={event.route or '-'} | status={event.status or '-'}"
                  f" | results={event.result_count if event.result_count is not None else '-'}"
                  f" | text={event.text_count if event.text_count is not None else '-'}"
                  f" | CLIP={event.clip_count if event.clip_count is not None else '-'}"
                  f" | examples={event.example_count if event.example_count is not None else '-'}")
    if state.get("status") == "categories":
        for category in state.get("categories", []):
            print(f"- {category['name']} ({category['count']})")
        return
    print(f"\nContext: {state.get('query_context') or '(selected examples only)'}")
    for index, item in enumerate(recommendations, start=1):
        print(
            f"{index}. {item['title']}"
            f" | text={item['text_similarity']}"
        )
        print(f"   id: {item['id']}")
    if not recommendations:
        print("No catalogue candidates found. Try a broader request or reset the session.")


def main() -> None:
    """Run the interactive terminal client for a LangGraph recommendation thread."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", default=str(uuid.uuid4()))
    parser.add_argument("--category-prefix", action="append", default=[])
    args = parser.parse_args()

    agent = build_recommendation_agent()
    print("Recommender session started. Enter a request, /select N, /refine TEXT, /reset, or /quit.")
    print(f"Session: {args.thread_id}")
    initial = run_recommendation_turn(agent, thread_id=args.thread_id, turn={})
    _show(initial)
    previous_results: list[dict] = initial.get("recommendations", [])
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.casefold() in {"/quit", "/exit"}:
            break
        if line.casefold() == "/reset":
            state = run_recommendation_turn(
                agent, thread_id=args.thread_id,
                turn={"reset_context": True, "category_prefixes": args.category_prefix},
            )
            previous_results = []
            print("Session preferences and selected examples cleared.")
            continue
        if line.startswith("/select "):
            try:
                selected_index = int(line.removeprefix("/select ").strip())
                if selected_index < 1:
                    raise ValueError("Result positions start at 1")
                selected_id = previous_results[selected_index - 1]["id"]
            except (ValueError, IndexError):
                print("Choose a result number from the most recent list, such as /select 2.")
                continue
            turn = {"selected_product_id": selected_id}
        elif line.startswith("/refine "):
            refinement = line.removeprefix("/refine ").strip()
            if not refinement:
                print("Add a preference after /refine, such as /refine green and battery powered.")
                continue
            turn = {"refinement": refinement}
        else:
            turn = {"query": line}
        if args.category_prefix:
            turn["category_prefixes"] = args.category_prefix

        try:
            state = run_recommendation_turn(agent, thread_id=args.thread_id, turn=turn)
        except (RuntimeError, ValueError) as exc:
            print(f"Unable to retrieve recommendations: {exc}")
            continue
        previous_results = state.get("recommendations", [])
        _show(state)


if __name__ == "__main__":
    main()
