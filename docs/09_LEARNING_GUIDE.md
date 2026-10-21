# Learning guide

This guide explains engineering patterns as they appear in this repository.

## 1. Typed model output is an input contract, not trusted state

**Concept:** Structured generation constrains response shape; application code still needs to validate meaning and authorization.

**Where:** `TurnEdits` and `PlanEdit` in `src/agent/models.py`; `complete()` in `src/agent/llm.py`; `apply_edits()` in `src/agent/edits.py`.

**Walkthrough:** Ollama receives JSON Schema for `TurnEdits`, and Pydantic parses the result. The application then verifies each edit's quote against the current message, checks targets and conflicts, and builds a new validated `Plan`.

**Trade-off:** More validation code, but the model cannot simply replace the full session snapshot. Schema validity alone does not prove the edit is grounded.

**Review questions:** Which checks belong in Pydantic versus business logic? What should happen when output is valid JSON but unsupported by the user's words?

## 2. LangGraph as explicit state transition orchestration

**Concept:** A graph names steps and branches while each node returns partial state updates.

**Where:** `RecommendationState`, `_route`, `_retrieve`, `_respond`, and `build_recommendation_agent` in `src/agent/workflow.py`.

**Walkthrough:** `_route` returns `route=retrieve` for recommendation or `route=respond` for clarification/categories/cart. Conditional edges choose the next node. Retrieval proceeds to response; response ends the graph.

**Trade-off:** The graph makes routing inspectable, while durable storage is managed separately by PostgreSQL code. Adding nodes requires aligned edges, state fields, trace behavior, and test coverage.

**Review questions:** Which fields are durable versus transient? Can each graph edge be tied to a concrete node result?

## 3. Hybrid retrieval and reciprocal-rank fusion

**Concept:** Different embedding spaces can produce ranked lists that are fused by rank instead of assuming raw similarity scores are directly comparable.

**Where:** `retrieve_products()` in `src/rag/retrieval.py`.

**Walkthrough:** Nomic embeds the query against product-text vectors. CLIP embeds the query into image feature space. PostgreSQL returns ranked product lists. Reciprocal-rank fusion adds `weight / (60 + rank)` per list; present ranking modes are normalized by available total weight. Selected products may contribute a third list.

**Trade-off:** Rank fusion avoids comparing unlike score scales, but weights and candidate pool size influence ordering. Exact vector search is simple at the documented catalog scale; approximate indexes add recall and operational trade-offs.

**Review questions:** What happens if CLIP is unavailable? Do all lists share hard filters? How should relevance be evaluated beyond a small query sample?

## 4. Deterministic filters around semantic ranking

**Concept:** Use SQL for facts supported by catalog fields and semantic similarity for softer natural-language needs.

**Where:** `filter_clause()`, `is_relevant_product()`, and `sanitize_apparel_products()`.

**Walkthrough:** SQL always includes Apparel & Accessories, with optional category, brand, condition, title color, and product type. Embeddings rank candidates. A post-rank gate checks core terms and visible colors. Fallback relaxes title color/type while preserving category/brand/condition.

**Trade-off:** Hard filters protect scope and explicit supported facts, but incomplete catalog metadata can reduce recall. The response states that preferences are not verified facts.

**Review questions:** Is a user constraint hard or soft? Does fallback preserve every constraint the UI says is active?

## 5. Row locks for per-session consistency

**Concept:** A database row lock serializes turns that update the same session snapshot.

**Where:** `run_recommendation_turn()` in `src/agent/workflow.py`; `agent_sessions` schema migration.

**Walkthrough:** The wrapper inserts a missing row, selects it `FOR UPDATE`, runs the graph against that snapshot, and writes the updated JSONB in the transaction. Turns for one thread contend on the same lock, with a five-second local lock timeout.

**Trade-off:** Simple per-thread consistency; holding the transaction during inference/retrieval can lengthen lock duration. Measure contention before redesigning it.

**Review questions:** What can be computed before locking? What should happen after a lock timeout? Is an additional LangGraph checkpointer useful here?

## 6. Local inference and failure boundaries

**Concept:** Local models make the system self-contained but shift availability, caching, and hardware requirements to the host.

**Where:** `Settings`, `complete()`, `embed_texts()`, and `_clip_encoder()`.

**Walkthrough:** Ollama requests use loopback validation and bounded timeouts. CLIP loads from the local cache only. Router calls have no implicit retries; CLIP query encoding can fall back to Nomic-only ranking, while Nomic/database failures propagate.

**Trade-off:** Local data path without hosted API keys, against cold starts, RAM/VRAM constraints, and the need to install model files first.

**Review questions:** Which failures preserve useful output? Can a request download model files? Are timeout budgets appropriate for cold starts?

## 7. A2UI as a rendering boundary

**Concept:** The backend sends declarative surface operations and data; the frontend catalog maps a known component to React rendering.

**Where:** `product_gallery_operations()` in `src/agent/a2ui.py`, `catalog.tsx`, and `page.tsx`.

**Walkthrough:** The API emits operations to create/update a `ProductGallery` surface. The page passes them to `processMessages`; the catalog renderer reads product data and uses React context callbacks for selection and cart actions.

**Trade-off:** Recommendation logic stays separate from presentation, but operation schema, surface lifecycle, product fields, and renderer must stay aligned.

**Review questions:** Which side owns product identity and order? How do duplicate surface creation and missing image URLs behave?

## Suggested exercises

1. Trace a plain search through API, row lock, graph, embeddings, SQL, relevance gate, response serialization, and UI rendering.
2. Compare a plain search with “Actually, show used Adidas” and identify the router decision.
3. Follow a selected ID from UI state to example-vector lookup and identify where it is excluded from results.
4. Find each Apparel & Accessories enforcement point and explain why ingestion and retrieval both need one.
5. Inspect the persisted session keys and identify which current-turn fields are transient.

These are code-reading exercises. Verify expected runtime outputs in a configured environment before describing them as observed results.
