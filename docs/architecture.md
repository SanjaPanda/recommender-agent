# Architecture and Phase 1 findings

## Project objective

Build a local-first product discovery prototype over the Shopify Product Catalogue. Users describe needs in natural language, browse catalogue-backed product cards, refine results with feedback and supported filters, and compare a shortlist.

The project uses a single stateful LangGraph workflow. The agreed next design adds an LLM router coordinating SQL and RAG components, conditional preference verification, and grounded summaries. These are bounded nodes or subgraphs within the workflow, not independently looping agents. See the finalized design and implementation notes below.

## Finalized agent design (2026-10-18)

Status: implemented as router, retrieval, conditional verifier, and response nodes. Session snapshots and cart state persist in PostgreSQL through migration 002. Local Ollama `gemma3:4b` is configured for routing, summaries, and vision; Nomic and CLIP remain the retrieval encoders. Visual attribute caching and arbitrary image uploads are not implemented.

### Session and routing

- A new session and the exact command `#buynew` show the categories available in the local database. `#buynew` resets the active shopping context and selected examples while preserving the cart.
- An LLM router interprets the latest message and structured session state. It chooses category browsing, clarification, SQL-only retrieval, semantic/visual retrieval, combined retrieval, or explicit cart actions.
- Persist session state in PostgreSQL: conversation context, category scope, structured filters, hard exclusions, soft preferences, selected product IDs, last displayed products, and cart contents.
- Update preferences rather than concatenating every message. "Actually Adidas" replaces the brand; "also lightweight" adds a preference; "blue is fine" removes the blue exclusion. Ask a clarification when the intended change cannot be determined.

### Retrieval contract

- Pass validated structured filters to retrieval, never arbitrary LLM-generated SQL. Parameterized queries enforce category, brand, and `is_secondhand` constraints. Unknown condition does not satisfy an explicit request for brand-new products.
- SQL-only requests return a bounded, deterministically ordered product set. RAG handles descriptive text, selected images, or both. Combined requests apply the SQL constraints inside each vector query before candidate limits and ranking.
- Reuse stored image vectors for selected catalogue products, retaining the existing maximum of three examples and excluding those seeds from recommendations. Uploaded images require a separate ingestion path and are not implied by existing support.
- Start with a configurable pool of 20 unique candidates and display at most five final products. This is an initial operating limit, not a calibrated ranking parameter.
- Nomic text vectors and CLIP image vectors remain separate; combine their rankings rather than mixing vectors from different embedding spaces.

### Conditional preference verification

Run verification only when active session preferences require checks beyond constraints already enforced reliably by SQL. An empty preference list, or a list fully resolved by SQL, skips this node.

1. Check reliable stored product attributes and supporting product text.
2. Inspect candidate images with a vision-capable model only for unresolved visually observable attributes. Distinguish the product from its background and accessories.
3. Return structured evidence per product and preference: `matches`, `violates`, or `uncertain`. Missing information stays unknown; vision cannot establish nonvisual properties such as actual material composition or condition history reliably.
4. Exclude hard-constraint violations. Keep uncertain products separate from verified recommendations. Soft preferences affect ranking rather than eligibility.

For "no blue clothes", exclude confirmed visible blue on the garment, including multicolored garments. Do not treat a blue background as a violation. Cached extracted attributes should retain evidence, provenance, and model version; model confidence alone does not make an attribute verified.

Verification makes one bounded pass over the candidate pool. **There is no automatic refill or retrieval retry loop.** Return fewer than five verified products, or none, with a clear explanation when necessary. A later user message can initiate a new retrieval. Any future refill policy must have an explicit attempt cap and is outside this design.

### Ranking, response, and cart

- Rank eligible candidates using relevance and soft preferences. Ranking and summarization cannot reintroduce excluded products or silently relax hard constraints.
- Generate summaries from retrieved product facts and verification evidence. Do not invent prices, availability, or unsupported attributes. Price filtering remains unsupported by this catalogue.
- Render validated category choices, product cards, and cart actions through A2UI when needed; clarification and explanatory replies can be text-only.
- Persist cart additions/removals only for explicit user actions and validated product IDs. Selecting a visual example is not an add-to-cart action. The cart is a tracker, not checkout or inventory reservation.

### Flow

```text
User message or UI action
  -> Router + persistent session state
     -> New session / #buynew: category choices
     -> Clarification: text response
     -> Cart action: update tracker and acknowledge
     -> SQL-only / RAG / SQL-filtered RAG: bounded candidates
        -> Conditional preference verifier (one pass)
        -> Rank eligible products
        -> Grounded summary + A2UI when needed
```

### Acceptance cases

- New session and `#buynew` show local categories; reset preserves the cart.
- Brand-only and condition-only requests use SQL; descriptive requests use RAG; combined requests preserve all SQL constraints across both modalities.
- Preferences survive turns and API restarts; replacements and removals do not leave stale constraints active.
- No unresolved preference means no vision call. "No blue" excludes confirmed blue garments and separates uncertainty; "prefer black" influences order.
- Fewer than five qualifying candidates produces fewer cards with no automatic retrieval loop. Verifier failure does not label unchecked products as verified.
- Selected examples do not enter the cart without an explicit action. Summaries and cards reference only returned catalogue records.

Implementation entry points: [workflow](../src/agent/workflow.py), [retrieval](../src/rag/retrieval.py), [API](../src/agent/api.py), and [A2UI operations](../src/agent/a2ui.py).

## Phase 1 workspace and environment

- This is a new project. The workspace contains unrelated directories, including ShopSphere; none is the repository for this project and none was modified.
- The project directory is `recommender-agent/`.
- Python 3.12.15 is managed by `uv` 0.13.1. Project dependencies are locked in `uv.lock` and installed in `.venv/`.
- Docker Desktop with its WSL 2 backend is installed and running. `compose.yaml` defines the local pgvector service with a named data volume; the initial migration is `migrations/001_initial_schema.sql`.
- Ollama 0.40.5 is installed and its local API responds. `nomic-embed-text` has been downloaded; a two-query API probe returned 768-dimensional vectors.
- PostgreSQL/pgvector is running; the initial migration is applied and the 250-product sample is loaded.

## Dataset findings

Source: [Shopify Product Catalogue](https://huggingface.co/datasets/Shopify/product-catalogue)

- Hugging Face lists 48,289 records: 38,631 train and 9,658 test.
- The repository is Parquet and the published files total about 9.5 GB, split over five test shards and fifteen train shards.
- Documented fields: `product_title`, `product_description`, `product_image`, `potential_product_categories`, `ground_truth_brand`, `ground_truth_is_secondhand`, and `ground_truth_category`.
- The dataset card reports descriptions on 92.9% of records and brands on 98.2%.
- There is no documented price field or stable product identifier. Budget limits therefore cannot be enforced from this dataset alone, and robust identifier-based deduplication is not available unless additional fields are found during inspection.
- The card declares Apache-2.0. Before redistributing image data, verify applicable image-specific terms and any source constraints.
- A bounded streaming sample was fetched from the train split at dataset revision `002ca15536bb015ebc47b99cc2d716d7040bb09e`: 250 records and 250 image files, about 39.8 MiB total, under `.data/shopify-catalogue-sample/`. The manifest records missing values and duplicate-title candidates. In this sample, 12 descriptions and 11 brands are missing; no duplicate-title candidates were found. The sample image values are embedded image bytes saved as local files. The fetch script can recreate the sample and keeps source fields without filling missing values.

## Architecture decisions

1. Use Python managed with `uv`, FastAPI, one LangGraph workflow, PostgreSQL with `pgvector`, Ollama for language and text embeddings, and a CLIP-compatible image encoder.
2. Start with a configurable small data sample and measure missing values, duplicate candidates, image usability, and data quality before selecting ingestion details. Never fill missing product fields with invented values.
3. Implement text retrieval and deterministic SQL constraints first. Add image embeddings and hybrid ranking only after the text baseline and evaluation are reproducible.
4. Enforce hard constraints in application logic and SQL. The catalogue's published schema has no price field, so do not present budget filtering as supported without a separate verified price source.
5. Use A2UI with the React renderer for catalogue-backed product cards. The current web client connects directly to this project's FastAPI workflow; interaction state remains in React while recommendation context remains in the LangGraph thread.
6. Keep the database, model names, embedding dimensions, and frontend configuration externalized. Keep model-based ingestion separate from API request handling.

## Planned phases

See [phases.md](phases.md) for the phase-by-phase scope, deliverables, dependencies, and completion criteria. Image embedding is Phase 3, after the Phase 2 text retrieval baseline can be measured.

## Current checkpoint (2026-10-17)

- Docker Desktop's Linux engine is healthy; Compose runs PostgreSQL/pgvector and the Next.js frontend.
- The database contains 1,221 products with ready 768-dimensional Nomic text vectors and 512-dimensional CLIP image vectors. Migration and idempotent ingestion are complete.
- Ollama is installed and responding locally; `nomic-embed-text` is used for request-time text embeddings.
- The FastAPI/LangGraph agent and CopilotKit/A2UI interface are available. Source code is grouped under `src/agent`, `src/rag`, and `src/ui`; data and database preparation scripts are under `scripts/`.
- The initial Phase 2 baseline details remain in [phase-2-evaluation.md](phase-2-evaluation.md); the wider prototype limitations are summarized in [the code walkthrough](code-walkthrough.md).

## Open questions and blockers

- Add structured attribute extraction and enforcement for attributes present in the catalogue, then evaluate its effect on ranking quality.
- Decide whether the in-memory LangGraph checkpointer should be replaced with persistent storage.
- Confirm image redistribution terms and whether an external price source is permitted or desired.

