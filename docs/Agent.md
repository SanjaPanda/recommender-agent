# Agent behavior

## Turn execution

One LangGraph workflow runs three nodes: router, retrieval and response. FastAPI validates user requests, PostgreSQL persists sessions/cart, and the UI receives bounded product cards plus A2UI operations.

Natural-language searches and refinements make one local Ollama call. The router receives the previous plan, whether the session is new, the latest message, available categories and displayed products. It returns a complete `RouterDecision`: updated query, filters, preferences, result count and grounded search interpretation. It also names cleared filters in `removed_filters`, which the UI displays. The model is instructed to preserve untouched constraints and reconcile replacements; Pydantic validates category scope, removal flags and interpretation evidence before state changes. Ambiguous requests return clarification while preserving the search plan and cards.

Explicit UI actions already identify an operation and its inputs. Category choices, selected examples and reset update validated state without a router call. Cart actions use a separate deterministic path; neither cart contents nor cart actions are part of the LLM contract. For example, clicking Add to cart sends the product ID directly to the cart handler.

Explicit condition requests receive an unsupported response before model calls or retrieval because the catalog does not reliably identify new/used products. Price/budget filtering is unsupported. Missing product facts remain unknown.

## Contracts and inference

- `Plan`, `Filters` and `Preference` describe saved shopping state; `RouterDecision` adds the interpretation and filter-removal flags.
- `RouterRequest` supplies the previous plan, session-new flag, latest message, eligible categories and displayed product references.
- `SessionTurn` validates user text and explicit category, example, reset or cart inputs.
- `SearchIntentRequest` and `SearchIntent` describe a product phrase, equivalent lexical terms and quoted request attributes.
- `ExplanationRequest` and `Summary` bound optional grounded response prose.
- API models validate discovery, session, refinement, selections and cart operations.

`agent.llm.complete()` sends Pydantic JSON Schema to local Ollama and validates the returned data. Errors raise `ModelError`; calls have bounded timeouts and no hidden retries. Search-intent generation releases the local LLM before retrieval encoders need memory. Summary generation is disabled by default.

## Retrieval and responses

The effective query combines the product need with preserved preferences. Natural-language turns include its interpretation in the router response, avoiding a second model call. Example-only and category actions reuse the saved interpretation when the effective query is unchanged; saved sessions without a matching interpretation use standalone extraction. Empty-query browsing and image-only searches need no interpretation. The UI interpretation panel shows request interpretation, not verified catalog facts.

Product-phrase grounding accepts regular singular/plural and supported spelling variants, while requiring complete evidence. For example, `dress` can be grounded by `Dresses`; an unrelated or shortened compound cannot. Attributes must be quoted from the request. The interpreted phrase constrains primary titles before retrieval limits; proposed equivalent terms expand BM25 only.

Retrieval combines BM25, Nomic text, query-to-image CLIP and selected-image examples using weighted RRF. Strong matches pass the relevance gate. When none pass, alternatives retain apparel/category/brand/product-phrase constraints; title color/type may relax. The response explains missing matches and may show fewer than five cards. Ranking parameters and evidence rules are specified in [architecture](architecture.md#search-and-ranking).

Product details expose source descriptions, features, structured details and snapshot ratings. Similarity details separate raw BM25, cosine values and RRF contributions. Match checks report explicit listing evidence as match/mismatch/unknown. Color filtering uses titles; image color and other preferences are not guaranteed.

## Sessions, selections and cart

Stable source IDs connect database rows, vectors, sessions and cards. Users can select up to three displayed products as visual examples. Selected products need compatible stored image vectors; selecting an example does not add it to the cart. Cart buttons require eligible product IDs. Adding a product removes it from recommendations and selected examples. Every retrieval mode excludes cart IDs before candidate limits. Removing a cart item makes it eligible for the next search; existing cards stay in place. Cart actions preserve the query, filters and preferences and require no inference.

Missing brands, descriptions or local image files do not prevent text retrieval. Missing images omit the image URL. Resume refreshes facts and checks product eligibility/source hashes. Changed source evidence invalidates saved recommendations/scores and refreshes cart facts. Unsupported saved fields are removed safely before plan validation; message history and session identity remain.

Session turns reject overlaps with HTTP 409, run inference outside transactions and reject stale saves. Existing snapshots with cart action plans are normalized on load, and cart/recommendation overlap is removed without rewriting the stored snapshot during read-only resume. See [state safeguards](architecture.md#state-and-evidence-safeguards) and [database contracts](database.md#sessions-and-caches).

## Interfaces

```powershell
uv run recommender-agent-api
uv run recommender-agent-chat
uv run recommender-search "blue cotton dress"
```

The API runs at <http://127.0.0.1:8000>; the Docker web UI runs at <http://localhost:3000>. Each completed turn includes a safe node trace with durations, route/status and modality/result counts. It excludes input text, prompts and model reasoning and is not stored in the session snapshot. Follow [setup](SETUP.md) before running these interfaces.
