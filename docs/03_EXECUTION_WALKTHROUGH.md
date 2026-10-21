# Execution walkthrough

All paths below are relative to the repository root. Commands are source/documentation-derived; the verification update at the end records which portions were later executed. Local model, data, database, and Docker prerequisites must be met as described in [SETUP.md](./SETUP.md).

## Step 1: Start the API

**What happens:** The console script resolves to `agent.api:main`, which starts Uvicorn on `127.0.0.1:8000`. Importing `agent.api` creates the FastAPI app, CORS policy, and compiled graph.

**Why:** The browser calls a loopback API and shares the host's local database and Ollama endpoint.

**Files/functions:** `pyproject.toml` script entry, `src/agent/api.py:main`, module-level `app`, `build_recommendation_agent`.

**Command:** From repo root, after `uv sync` and `.env` configuration: `uv run recommender-agent-api`. **Unverified; not run.**

**Inputs/outputs/state:** No request yet; API listens on loopback. Graph state is initialized per invocation.

**External calls:** None until endpoints are invoked.

**Expected behavior and failures:** Source specifies port 8000. Missing settings can fail during import; an occupied port prevents Uvicorn startup. Debug with API process logs and the environment settings in [SETUP.md](./SETUP.md).

## Step 2: Browser creates or resumes a session

**What happens:** `page.tsx` posts `/api/session` with the saved localStorage thread ID. The handler reads `agent_sessions`; an existing state is sanitized and current categories are fetched. A new session enters the graph with an empty turn, which yields categories without a router model call.

**Why:** The UI restores cards/cart and displays available categories on a new session.

**Files/functions:** `src/ui/app/page.tsx:submit`, `DiscoverRequest`-family models in `src/agent/api.py`, `/api/session`, `sanitize_apparel_products`, `run_recommendation_turn`, `_route`, `_response`.

**Command:** Start UI with `docker compose up -d --build web` from repo root, after Docker and API/database prerequisites. **Unverified; not run.**

**Inputs:** Optional thread ID. **Processing:** Database row lookup and apparel product ID check. **Output:** Session state, categories, and A2UI surface operations. **State change:** New thread creates a JSONB session snapshot.

**Failures:** Database/runtime/value errors produce HTTP 503 with a session-load message. Verify database readiness and migrations; no live diagnosis was performed.

## Step 3: User submits a discovery query

**What happens:** The UI example or search interaction posts `/api/discover`; API validates query length and generates a thread ID if missing. `run_recommendation_turn` row-locks and invokes the graph. `_route` fetches category options and determines whether the search can pass directly or requires LLM interpretation.

**Plain search path:** `_can_pass_plain_search` allows an ordinary query only if it contains no edit cue and cannot conflict with stored brand/condition/title constraints or excluded colors. It sets the query and chooses retrieval.

**LLM path:** `_route` sends the previous plan, message, available categories, displayed products, and cart to `complete(TurnEdits, RouterRequest)`. `apply_edits` checks exact message evidence and atomic consistency. Invalid proposed edits yield a clarification state without retrieval.

**Files/functions:** `api.discover`, `run_recommendation_turn`, `_route`, `_is_plain_search`, `_can_pass_plain_search`, `complete`, `apply_edits`.

**Inputs/outputs:** Query/refinement and saved plan become a validated plan and graph branch. **External calls:** PostgreSQL; Ollama only on router path.

**Failures:** Model failures propagate as `ModelError`; discovery maps `RuntimeError` and `ValueError` to HTTP 503. `ModelError` is a `RuntimeError` subclass. No retry occurs in `complete`.

## Step 4: Retrieve and gate candidates

**What happens:** `_retrieve` combines plan query and preferences. For browse state it obtains category candidates. Otherwise it asks `retrieve_products` for a wider pool (`max(result_limit * 4, 20)`), then applies `is_relevant_product`. If no strong result remains and title filters were active, it makes one retrieval with those title filters relaxed for alternatives.

**Files/functions:** `_retrieve`, `retrieve_products`, `embed_texts`, `clip_runtime.text_embedding`, `_search_embedding`, `filter_clause`, `is_relevant_product`.

**Data transformations:** Nomic query text gets the `search_query: ` prefix. CLIP generates a normalized query vector when available. Selected product image vectors are averaged and normalized. SQL applies the apparel root and supported filters before vector results are limited. RRF combines present result lists.

**Outputs/state:** Candidate rows carry stable IDs, product data, per-mode similarities, fused score, and source marker. Category/brand/condition remain hard constraints even for fallback.

**Failures:** CLIP query encoder errors trigger a warning and Nomic-only ranking. Missing Ollama text embedding or PostgreSQL errors fail the turn. Selected IDs without available image vectors raise `ValueError`.

## Step 5: Form response and persist

**What happens:** `_respond` creates deterministic result messages and statuses. If summaries are enabled and products exist, it calls `complete(Summary, ExplanationRequest)` with bounded product facts; `ModelError` is ignored. The turn wrapper saves durable plan/messages/recommendations/cart state and removes transient fields.

**Files/functions:** `_respond`, `complete`, `run_recommendation_turn`, `_trace_node`.

**Outputs:** Recommendations, message, status, history, and safe node trace. The trace holds durations, route/status, and counts; it does not hold prompts, user text, product text, or model reasoning.

**Failures:** Graph node exceptions are logged and re-raised; session DB write failure prevents normal response completion. No retry is implemented.

## Step 6: Serialize API response and render gallery

**What happens:** `_response` resolves local image paths through `_serialize_products`, omits internal `image_meta`, adds image URLs only for valid local sample files, and builds A2UI operations. UI feeds operations to `processMessages`; the fixed `ProductGallery` renderer shows cards and selection/cart controls.

**Files/functions:** `api._response`, `_serialize_products`, `_find_image_path`, `product_gallery_operations`, `page.tsx`, `catalog.tsx`.

**Outputs/state:** JSON response includes thread ID, products, plan filters/preferences, cart, categories, trace, and A2UI operations. UI stores thread ID in localStorage and session in React state.

**Failures:** Missing image files omit image URLs. API image endpoint returns 404 if a local image cannot be resolved. Browser network failures are shown in the UI error area.

## Follow-up interactions

- Selecting up to three cards and submitting a refinement posts `/api/recommend`; selection is checked against cards visible in this session. The selected IDs affect CLIP example retrieval and are not cart actions.
- Add/remove cart requests post `/api/cart` with explicit action and product IDs. Adds are allowed only for currently displayed recommendations; removes require current cart membership.
- `/api/health` returns `{"status":"ok"}` and does not verify database/model readiness.

## Verification status

During the later end-to-end run, the root search, direct `/api/discover` text query, app health/UI HTTP checks, full live API lifecycle diagnostic, unit suite, and web Docker build were executed. Results and exact differences from the example commands are recorded in [07_RUNBOOK.md](./07_RUNBOOK.md). Database migration, ingestion, and vector generation commands were not rerun because the database already had the full apparel catalog and required vectors.
