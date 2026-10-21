# Agent guide

## Responsibilities

The application is one LangGraph agent with three steps: route the latest user message, retrieve apparel candidates, and form the response. Plain product searches pass directly to hybrid retrieval; messages that appear to change filters/preferences or request unsupported behavior use the Ollama router. FastAPI validates external requests; PostgreSQL persists session state and the cart; the UI receives a bounded set of product cards and A2UI operations.

The route step calls Ollama with a typed `RouterRequest` and validates `TurnEdits`. Code applies supported edits atomically. Category, brand, condition, title color/product filters, product selection, and cart eligibility are deterministic checks. Preferences persist across turns. Unsupported prices and ambiguous edits produce clarification responses.

## Data contracts

- `Plan`, `Filters`, `Preference`, and `TurnEdits` define shopping state and router output.
- `RouterRequest` and `ExplanationRequest` define structured LLM inputs.
- `Summary` validates optional free-form summary text.
- FastAPI request models validate discovery, session, cart, and refinement inputs.

`complete()` accepts a Pydantic v2 input model and output model class. It sends the output JSON Schema to Ollama, parses the returned JSON with Pydantic, and raises `ModelError` if the provider is unavailable or returns invalid data. Calls have bounded timeouts and no hidden retries.

## Product identity and missing data

The stable source ID is carried unchanged in PostgreSQL, embeddings, saved sessions, API product cards, and A2UI selections. Missing descriptions, brands, or local image files are allowed. Missing images omit the UI image URL and do not prevent text retrieval. Products missing CLIP vectors cannot be used as visual examples; the current catalog has CLIP vectors for all 2,173 apparel products.

Image descriptions are outside the active product pipeline. The small catalog uses product text and CLIP image vectors; no vision-language model is loaded and no caption generation, caption embedding, or caption import commands are installed.

Retrieval always constrains the root category to `Apparel & Accessories`, including before database cleanup. Saved sessions from the old mixed catalog are filtered to currently valid apparel product IDs when loaded or continued; the session row and message history remain in PostgreSQL.

## Local interfaces

- API: `uv run recommender-agent-api` (`http://127.0.0.1:8000`)
- Web/A2UI: `docker compose up -d --build web` (`http://localhost:3000`)
- Terminal harness: `uv run recommender-agent-chat`
- Retrieval inspection: `uv run recommender-search "blue cotton dress"`

The web gallery supports selecting up to three displayed products as visual examples and adding free-text preference refinements. See [setup](SETUP.md) and [architecture](architecture.md).

Each API response includes an `Agent trace` summary for the current turn: router, retrieval, and response node durations, selected route/status, result counts, and how many displayed products have text, CLIP, or selected-example scores. Expand the trace beneath the message in the UI. It omits user text, prompts, product descriptions, and model reasoning, and it is not persisted in session snapshots.

Search results are relevance-gated instead of always filling five cards. Strong matches must pass explicit color/product text checks (or CLIP can support a visible color) and the initial score cutoffs: Nomic text similarity 0.61 or query-to-image CLIP similarity 0.28. Exact title/description term matches may pass below those cutoffs. If no strong result passes, the response apologizes and shows closest alternatives; selected subcategory, brand, and condition filters stay active for both lists. Only title color/product constraints are relaxed for fallback suggestions. The threshold values are conservative starting points based on a small judged query sample, not universal model guarantees.
