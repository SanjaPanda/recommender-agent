# Data flow and persistence

## Catalog ingestion

`scripts/db/ingest_sample.py` reads train and test JSONL files in order. Each parsed row must have a nonblank string `id` and a `ground_truth_category` equal to `Apparel & Accessories` or beginning with that root plus ` > `. Duplicate IDs across inputs are kept once. The importer maps source fields to product columns and stores the source record as JSONB `raw_source`, then upserts on stable `id`.

The stable ID is used by products, embeddings, sessions, API cards, and UI selection. The code uses content-derived IDs in the source-fetch workflow, but the full ID-generation algorithm is in [scripts/data/fetch_sample.py](../scripts/data/fetch_sample.py).

## Embedding pipeline

- Text product content is formatted as `search_document: <title>\n<description>`; queries use `search_query: ` prefix. Ollama returns vectors validated for finite values, equal dimensions, and nonzero content.
- Image embedding loads product files and cached CLIP model/processor, obtains image features, normalizes vectors, and writes modality `image` rows.
- Embeddings are keyed by product ID, modality, and model name; model digest/version is stored. Upserts update vector/status/version.
- Query retrieval creates Nomic and (when local CLIP is available) CLIP text vectors. Selected products retrieve stored CLIP image vectors and average them before normalization.

## Retrieval record shape

The retrieval functions combine fields into dictionaries resembling:

```python
{
    "id": "<stable product id>",
    "title": "<catalog title>",
    "description": "<optional text>",
    "brand": "<optional brand>",
    "category": "<ground truth category when present>",
    "image_meta": {"local_path": "<relative sample path>"},
    "is_secondhand": False,
    "text_similarity": 0.0,
    "image_similarity": 0.0,
    "example_similarity": None,
    "score": 0.0,
    "visual_source": "hybrid",
}
```

This is an illustrative shape based on retrieval construction; values and optional fields depend on which ranking modes return a row. API serialization removes `image_meta` and supplies `image_url` only when a local image resolves.

## Plan and turn state

`Plan` contains intent, bounded result limit, query, `Filters`, preference list, user message, and explicit product IDs. `TurnEdits` is a model-facing change set, not a full replacement plan. `apply_edits` checks edit evidence against the current user message and modifies a deep copy, preserving untouched state. `RecommendationState` also contains turn fields and transient candidate/trace/category fields.

Each API session is persisted in `agent_sessions.state` as JSONB. The coordinator creates a missing row, locks the row for the complete turn, sanitizes saved product IDs, invokes the graph, then omits transient `turn`, `candidates`, `uncertain`, `categories`, and `trace` keys from the persisted snapshot. The session row and message history remain even when old product IDs are filtered.

## Filter path and taxonomy boundary

Filter values are first validated by Pydantic and application logic. `filter_clause` rejects unknown keys and category prefixes outside the root, then builds parameterized SQL for:

- Apparel root and descendant path, always applied.
- Optional category path(s), brand, `is_secondhand`, title color, and primary title product type.
- Product type parsing uses a title segment before `for`, `with`, or `fits`; color is a title token match.

The retrieval boundary independently enforces Apparel & Accessories, so an LLM response cannot widen SQL scope. `sanitize_apparel_products` also prevents old session references from resurfacing.

## Request/response serialization

FastAPI Pydantic request models bound text, IDs, and list lengths. Retrieval products become response products; `_find_image_path` rejects absolute/traversal paths and restricts local files to known sample roots. The response carries status/message, plan filters/preferences, cart, category counts, trace events, and A2UI operations. The browser stores the thread ID in localStorage and keeps session/cards/selection in React state.

## Database objects

Current schema behavior is defined by migration history under [migrations/](../migrations/). The initial schema defines products, product embeddings, feedback and legacy recommendation sessions; a later migration adds agent session JSONB state. Migration history includes retired image-description and attribute work. The current application retrieval code uses only `text` and `image` modalities. The actually applied migrations and live schema were not checked during this phase; see [database.md](./database.md) for recorded historical inspections, clearly separated from current verification.

## Error/data handling notes

Product descriptions/brands may be absent. Missing image files do not prevent text retrieval. Invalid router edits lead to clarification rather than partial state application. API errors are mapped by endpoint; see the execution walkthrough. No session state or DB records were inspected live while writing this document.
