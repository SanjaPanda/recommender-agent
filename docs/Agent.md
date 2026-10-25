# Agent guide

## Responsibilities

The application is one LangGraph agent with three steps: route the latest user message, retrieve apparel candidates, and form the response. Plain product searches skip edit interpretation but receive structured local LLM search-intent extraction in the retrieval step. Messages changing filters/preferences or requesting unsupported behavior also use the Ollama edit router. FastAPI validates external requests; PostgreSQL persists session state and the cart; the UI receives a bounded set of product cards and A2UI operations.

The route step calls Ollama with a typed `RouterRequest` and validates `TurnEdits`. Code applies supported edits atomically. Category, brand, condition, title color/product filters, product selection, and cart eligibility are deterministic checks. Preferences persist across turns. Unsupported prices and ambiguous edits produce clarification responses.

## Data contracts

- `Plan`, `Filters`, `Preference`, and `TurnEdits` define shopping state and router output.
- `RouterRequest` and `ExplanationRequest` define structured LLM inputs.
- `SearchIntentRequest` and `SearchIntent` extract an open-vocabulary product phrase, equivalent search terms and attributes with exact request evidence. The interpretation is saved with its source query and reused only for identical query/preference text. Extraction errors stop the turn; there is no silent fallback to product-type guesses.

Product grounding tolerates regular singular/plural and supported spelling variants using the same phrase matcher as retrieval. For example, model output `dress` is supported by request `Dresses`; this is not an unrelated inferred product. Complete phrase boundaries remain required, and attribute values/evidence still require literal request support.
- `Summary` validates optional free-form summary text.
- FastAPI request models validate discovery, session, cart, and refinement inputs.

`complete()` accepts a Pydantic v2 input model and output model class. It sends the output JSON Schema to Ollama, parses the returned JSON with Pydantic, and raises `ModelError` if the provider is unavailable or returns invalid data. Calls have bounded timeouts and no hidden retries.

## Product identity and missing data

The stable source ID is carried unchanged in PostgreSQL, embeddings, saved sessions, API product cards, and A2UI selections. Missing descriptions, brands, or local image files are allowed. Missing images omit the UI image URL and do not prevent text retrieval. Products missing CLIP vectors cannot be used as visual examples; the expanded 2,000-product catalog has both vector modalities; independently judged relevance evaluation remains pending.

Image descriptions are outside the active product pipeline. The small catalog uses product text and CLIP image vectors; no vision-language model is loaded and no caption generation, caption embedding, or caption import commands are installed.

Retrieval always constrains the root category to `Apparel & Accessories`, including before database cleanup. Saved sessions from the old mixed catalog are filtered to currently valid apparel product IDs when loaded or continued; the session row and message history remain in PostgreSQL.

## Local interfaces

Search-intent calls release the local LLM after generation so retrieval encoders can use memory. Other routing/summary calls retain their existing keep-alive behavior. The current local `.env` selects the existing `AGENT_NUM_GPU=0` CPU inference option after an integrated partial-offload request timed out; this is a machine-specific setting, not a measured general speedup.

- API: `uv run recommender-agent-api` (`http://127.0.0.1:8000`)
- Web/A2UI: `docker compose up -d --build web` (`http://localhost:3000`)
- Terminal harness: `uv run recommender-agent-chat`
- Retrieval inspection: `uv run recommender-search "blue cotton dress"`

The web gallery supports selecting up to three displayed products as visual examples and adding free-text preference refinements. See [setup](SETUP.md) and [architecture](architecture.md).

Expand **How your request was understood** to inspect the interpreted product phrase, quoted attributes and proposed search terms. These are request interpretation, not verified product facts. The source product phrase must end a main-title segment before candidate limits; proposed terms expand BM25 ranking only. Vectors retain the original query and RRF weights are unchanged. This avoids a growing subtype ontology but can miss synonym-only titles. New textual searches require one additional local model call; unchanged-query turns reuse the saved interpretation. Browse, cart and image-only operations do not require this extraction call.

Each API response includes an `Agent trace` summary for the current turn: router, retrieval, and response node durations, selected route/status, result counts, and how many displayed products have text, CLIP, or selected-example scores. Expand the trace beneath the message in the UI. It omits user text, prompts, product descriptions, and model reasoning, and it is not persisted in session snapshots.

Search results are relevance-gated instead of always filling five cards. Strong matches must pass explicit color/product text checks (or CLIP can support a visible color) and the initial score cutoffs: Nomic text similarity 0.61 or query-to-image CLIP similarity 0.28. Exact title/description term matches may pass below those cutoffs. If no strong result passes, the response apologizes and shows closest alternatives; selected subcategory, brand, and condition filters stay active for both lists. Only legacy title color/product filters are relaxed for fallback suggestions; an extracted product phrase remains required. The threshold values are legacy defaults, unvalidated on any replacement dataset.

Explicit known query types are now checked against primary product titles in retrieval and the strong-match gate. Image-only searches inherit a type only from unanimously known seed title types; explicit product-type filters override seed inference. Same-type filtering may return fewer than five products. It does not validate material, age/gender or image semantics, and unreviewed extraction candidates remain outside hard filters.

Query retrieval now includes BM25 over source title/description/features in addition to Nomic and CLIP. Weighted RRF combines their ranks (k=60, equal base query weights; nonempty modes renormalized). Product payloads retain raw BM25, cosine scores and each RRF contribution; the card details show these separately. No LLM is required for BM25 or score fusion.

BM25 candidates retain their source features for the relevance gate: feature-only keyword intent can pass lexical checks. Color checks continue to use the prior title/description/CLIP rules; a color mentioned only in a feature variant is not automatically verified.


## State and evidence safeguards

Color changes reconcile the effective query and saved preferences. Session turns reject overlaps with HTTP 409 and run inference outside database transactions. Changed source hashes invalidate historical cards and refresh cart facts. Encoder revisions/dimensions and conservative primary phrase boundaries are enforced. See the [canonical architecture safeguards](architecture.md#audit-hardening-state-provenance-and-phrase-evidence) for the contracts, recovery behavior and recall limitations.
