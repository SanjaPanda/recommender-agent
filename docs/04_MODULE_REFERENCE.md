# Module reference

This reference covers active Python files under `src/agent/`, `src/rag/`, and `scripts/`. Helpers are grouped by module; line numbers can move as code changes. Caller/callee relations are derived from current source, not runtime profiling.

## `src/agent/models.py`

Shared Pydantic contracts use `extra="forbid"` through `Contract`.

- `AgentTraceEvent`: one completed graph node's safe duration, route/status, and count metrics.
- `Filters`: optional catalog-backed title color/product, exact brand, condition, and category-prefix filters.
- `Preference`: user requirement text, hard/soft strength, and visible-attribute flag.
- `Plan`: persistent shopping intent, result limit (1–5), query, filters, preferences, message, and cart IDs.
- `PlanEdit`: base explicit edit contract with field, operation, target, value, evidence, strength, and visual flag. `AddEdit`, `ReplaceEdit`, and `RemoveEdit` specialize value requirements.
- `TurnEdits`: model response containing intent and a list of edits rather than a full plan.
- `Summary`: optional generated message.
- `ProductReference`, `ProductFacts`: bounded product data supplied to model calls.
- `RouterRequest`: prior plan, current message, available categories, displayed products, and cart.
- `ExplanationRequest`: plan and bounded product facts for summary generation.

## `src/agent/config.py`

- `ROOT`: repository root resolved from the module path.
- `Settings`: Pydantic Settings configuration for required secret `DATABASE_URL`, local Ollama URL, model names, reasoning/GPU settings, router timeout, and optional summaries. `require_local_ollama` rejects non-loopback hosts. A module-level `settings` instance loads on import.

## `src/agent/llm.py`

- `ModelError`: provider/timeout/validation failure signal.
- `complete(schema, system, data, *, timeout=None)`: sends one schema-constrained Ollama chat request, parses JSON into the requested Pydantic model, and raises `ModelError` on provider or validation issues. No retries. Called by workflow routing and optional response summaries.

## `src/agent/edits.py`

- `AmbiguousEdit`: unsafe or conflicting proposed edit.
- `normalized(value)`: case-fold and collapse whitespace for equality/evidence comparisons.
- `apply_edits(previous, turn, message)`: deep-copies a plan, validates every quote against the current message, updates category/preference/scalar fields, checks conflicts, and validates the final plan. Returns a new plan; raises `AmbiguousEdit` on ambiguity. Called by `_route`.

## `src/agent/refinements.py`

- `COLORS`, `COUNTS`: fixed supported lexical values.
- `color_refinement(previous, message)`: recognizes complete color/count follow-up forms and “any color” forms. Copies the plan, updates title color/count, and may infer a single title product noun from the existing query. Returns `None` when the text is not an unambiguous supported form. Called before LLM routing.

## `src/agent/relevance.py`

- `is_relevant_product(product, query, *, has_visual_examples=False, required_title_color=None, required_product_type=None, ignore_query_attributes=False)`: applies explicit color and lexical checks plus Nomic/CLIP/example thresholds. Excluded colors reject a candidate. Fallback callers can ignore query color/product terms.
- `_tokens(value)`: tokenize alphanumeric words and omit generic terms.
- `_canonical(token)`: limited plural normalization.
- `_positive_colors(query)`: return recognized non-negated colors.
- `_negated_colors(query)`: detect colors following supported negation terms.

## `src/agent/workflow.py`

- `RecommendationState`: TypedDict for session state and per-turn transient fields.
- `_route(state)`: classify/reset/category/cart/search/refinement; optionally call typed router and atomic edit applier. Validates displayed selection IDs and apparel category paths. Returns partial state update with `route` (`retrieve` or `respond`).
- `_retrieve(state)`: builds query from plan plus preferences, calls retrieval, applies relevance gate and optional relaxed fallback. Returns candidates/result kind.
- `_respond(state)`: turns candidates into recommendations and user message; optional summary LLM call; updates bounded history and status.
- `build_recommendation_agent(*, checkpointer=None)`: registers router/retrieve/respond nodes with safe wrappers and compiles graph.
- `_trace_node(name, node)`: returns a wrapper that times one node, logs safe error metadata, and appends an `AgentTraceEvent`; exceptions are re-raised.
- `sanitize_apparel_products(connection, state)`: checks saved recommendation/cart/selection IDs against apparel source category in SQL and removes invalid references, preserving other state/history.
- `_is_plain_search(text)`: true for nonempty text without router edit cues.
- `_can_pass_plain_search(text, previous_plan)`: additionally prevents shortcut when saved hard filters exist or new colors conflict with stored exclusions.
- `run_recommendation_turn(agent, *, thread_id, turn)`: PostgreSQL row-lock wrapper; sanitize, invoke graph, persist selected durable fields, and return full state. Raises `ValueError` for invalid thread ID and database errors for persistence failures.

## `src/agent/api.py`

- `DiscoverRequest`, `RecommendRequest`, `SessionRequest`, `CartRequest`: validate endpoint JSON fields and bounds.
- `_find_image_path(image_meta)`: resolves a relative image under the two known local sample roots; rejects absolute paths and traversal.
- `_product_image_path(product_id)`: use in-memory path cache, then look up image metadata in PostgreSQL.
- `_serialize_products(items)`: remove internal image metadata, default missing title, and add a local image endpoint URL when a file resolves.
- `_response(state, thread_id, heading, *, create_surface)`: project graph state into response JSON and A2UI operations.
- `health()`: liveness response only.
- `discover(request)`: initial/free-text discovery; maps database errors and runtime/value errors to HTTP 503.
- `session(request)`: load/sanitize saved snapshot or initialize a new categories session; failure maps to HTTP 503.
- `cart(request)`: explicit cart action; invalid action/product maps to 400, service errors to 503.
- `recommend(request)`: validates selection/refinement presence and unique IDs, then continues session; errors map to HTTP 503.
- `product_image(product_id)`: serves only resolved local image; returns 404 otherwise.
- `main()`: runs local Uvicorn on `127.0.0.1:8000`.

## `src/agent/a2ui.py`

- `CATALOG_ID`: catalog identifier shared with the UI renderer.
- `product_gallery_operations(products, *, heading, surface_id, create_surface)`: emits optional delete/create plus component and data-model update operations. Called from API response construction.

## `src/agent/cli.py`

- `_show(state)`: prints message, trace, category list, query context, and recommendations.
- `main()`: interactive terminal loop; handles `/select`, `/refine`, `/reset`, `/quit`, and ordinary queries through the same turn wrapper.

## `src/rag/embed_text.py`

- `EmbeddingRequest`, `EmbeddingResponse`: validate Ollama embedding payloads and finite, consistent, nonzero vectors.
- `post_json(path, payload)`: post JSON to local Ollama with a 300-second timeout.
- `embed_texts(inputs)`: validate input, call `/api/embed`, validate response and vector count.
- `model_digest()`: obtain configured model digest from `/api/tags`; raises `SystemExit` with pull instruction when absent.
- `document_text(title, description)`: creates Nomic document-prefixed product text.
- `main()`: CLI that selects apparel product rows missing current vectors (unless forced), embeds in batches, and upserts versioned vectors.

## `src/rag/clip_runtime.py`

- `_clip_encoder(model_name)`: process-cached CPU CLIP model/processor using `local_files_only=True`; reports missing dependencies/cache as `RuntimeError`.
- `text_embedding(query)`: encodes nonempty text, validates finite nonzero output, and returns unit-normalized CLIP feature vector.

## `src/rag/embed_images.py`

- `resolve_image(local_path, roots)`: resolve local image path using configured roots; current helper accepts an existing absolute path too.
- `main()`: load cached CLIP, select eligible apparel products and local images, batch image encoding, normalize vectors, and upsert image modality rows.

## `src/rag/retrieval.py`

- `retrieve_products(*, query_context="", selected_product_ids=None, category_prefixes=None, limit=5, filters=None)`: validate inputs; obtain embeddings; run exact pgvector searches; fuse ranked lists; return top products and score fields.
- `_search_embedding(connection, *, vector, modality, model, category_clause, category_params, excluded_ids, limit)`: one exact cosine-distance SQL search over products, applying shared filters.
- `filter_clause(filters, category_prefixes)`: validate allowed filter keys and apparel taxonomy; create parameterized SQL predicates.
- `list_categories()`: return apparel category names and counts.

## `src/rag/search.py`

- `main()`: validate query/reference/filter CLI arguments, call `retrieve_products`, print JSON rows without image metadata.

## Scripts

- `scripts/db/apply_migrations.py`: `main()` creates migration ledger if needed, sorts SQL files, skips applied versions, runs each file's statements transactionally, and records the version.
- `scripts/db/ingest_sample.py`: `source_image_meta(row)` extracts image metadata; `is_apparel(row)` enforces root/descendant taxonomy; `read_apparel_rows(paths, limit=None)` parses JSONL, filters, de-duplicates IDs; `main()` upserts accepted products.
- `scripts/data/fetch_sample.py`: helpers normalize source values, infer image extension, derive content identity, load compact identity maps; `main()` fetches/prepares local train/test sample files. See script help for exact flags; not executed here.
- `scripts/check_agent_live.py`: `main()` orchestrates live API/database checks. Nested helpers `check`, `post`, `start` assert, send timed JSON requests, and launch/poll a temporary API process. This diagnostic has external side effects (process, DB session, report/artifacts) and was not run.
- `scripts/check_router_edits.py`: `main()` loads requested cases, makes real router calls, captures output, compares plan changes and writes a report. Nested `shopping` extracts comparable constraints; `observe` records real completion output. It performs inference and writes output; not run.

## Not included

The UI is TypeScript/React, not a Python module. Migration SQL is documented in [05_DATA_FLOW.md](./05_DATA_FLOW.md). Ignored `.pyc` files are not treated as active modules.
