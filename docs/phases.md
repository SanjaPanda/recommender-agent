# Project phases

This guide describes the planned build sequence for the local-first product discovery prototype. Each phase has a concrete outcome and completion criteria. Later work builds on the preceding phase's working foundation.

## Phase 1 — Data and database

**Goal:** establish a reproducible, inspectable catalogue sample and a usable local data store.

**Work:**

- Fetch and pin a bounded sample of the Shopify Product Catalogue; record its source revision, missing values, image paths, and hashes.
- Preserve source fields and represent missing values as missing.
- Provide repeatable sample inspection and ingestion commands.
- Start PostgreSQL with `pgvector`, apply ordered schema migrations, and import products idempotently.
- Keep image files on disk; store their relative paths and hashes in the database.

**Complete when:** a developer can recreate or inspect the sample, start the database, apply migrations, ingest the sample repeatedly without duplicate products, and verify stored product and image metadata.

**Current status:** complete. The original 250-row sample plus category-based expansions are imported. PostgreSQL/pgvector is healthy, the initial migration is applied, and ingestion is idempotent. The database has 1,221 distinct products as of 2026-10-17.

## Phase 2 — Text retrieval baseline

**Goal:** retrieve relevant products from natural-language text queries and establish a measurable baseline before adding image similarity.

**Work:**

- Use `nomic-embed-text` through Ollama as the initial baseline (768 dimensions; model digest recorded with each embedding). It was selected for the laptop's 8 GB RAM and longer context than the smaller `all-minilm` option.
- Generate embeddings for product titles and available descriptions without fabricating missing descriptions.
- Persist vectors in `pgvector` and implement nearest-neighbor text search.
- Add deterministic filters for supported catalogue fields, such as category, brand, or secondhand status where appropriate.
- Build a small query set and record retrieval quality, latency, and limitations.

**Current status:** complete for the current 1,221-product prototype catalogue. `nomic-embed-text` is installed; all 1,221 products have ready 768-dimensional text vectors. Semantic search supports brand, category, secondhand, and ground-truth category-prefix filters, applies a configurable 0.65 minimum similarity, and declines budget requests because prices are unavailable. Initial and expanded evaluation results are documented in [the evaluation notes](phase-2-evaluation.md); query sets are hand-labeled and exploratory, not representative benchmarks.

**Complete when:** embeddings can be generated and refreshed reproducibly, text search and supported filters return stable results, and a documented evaluation provides a baseline for Phase 3. Budget filtering remains unsupported without a verified price source.

## Phase 3 — Image embeddings and multimodal retrieval

**Goal:** use product imagery alongside text and measure the change against the Phase 2 text-only baseline.

**Work:**

- Select a CLIP-compatible image encoder and configure it independently from the text model.
- Generate embeddings from local sample images; record encoder/version and vector dimensions, and handle unreadable images explicitly.
- Store image vectors and connect each vector to its product record.
- Implement configurable hybrid ranking that combines text and image signals.
- Evaluate text-only and hybrid results on the same query set; document gains, regressions, cost, and latency.

**Current status:** implemented and evaluated as a prototype. The encoder is `openai/clip-vit-base-patch32` (CPU, 512-dimensional shared image/text space); all 1,221 current products have ready image vectors. Text-image hybrid retrieval and search from one to three selected catalogue examples are implemented. Evaluation shows little aggregate change at low image weight, while a high weight helps a color-specific yellow-sweeper query but lowers overall ranking quality. More labeled cases are needed before treating results as final. See [the Phase 3 prototype notes](phase-3-prototype.md) and [the code walkthrough](code-walkthrough.md).

**Complete when:** image vectors can be rebuilt reproducibly, hybrid retrieval returns catalogue-backed products, and evaluation fairly compares it with the Phase 2 baseline. **Image embedding happens in Phase 3**, after the text baseline is working and measurable.

## Phase 4 — Agent workflow

**Goal:** coordinate retrieval and user actions through one explicit, stateful LangGraph workflow.

**Current status:** the one-node LangGraph workflow retains query history, selected product IDs, and category scope per thread in an in-memory checkpointer, then passes accumulated query text and selected image examples to shared RAG retrieval. The terminal client supports search, selection, refinement, and reset. Structured attribute extraction/hard constraints and persistent checkpoints remain open. See [the code walkthrough](code-walkthrough.md).

**Work:**

- Define typed workflow state for the query, retrieved candidates, supported constraints, feedback, shortlist, and response.
- Add deterministic search, filter, compare, and feedback tools that call the retrieval/data layer.
- Track session preferences from explicit feedback and apply them transparently to later searches.
- Keep hard constraints in deterministic code and SQL.

**Complete when:** a documented workflow can search, refine, compare, and record feedback with predictable state transitions and reproducible behavior.

## Phase 5 — API and user interface

**Goal:** make the workflow usable through an API and a product discovery interface.

**Current status:** a FastAPI service and Dockerized Next.js/A2UI interface are implemented. The UI presents five candidates, supports selecting up to three examples and adding free-text preferences, and updates recommendations in the same LangGraph thread. The frontend is published on localhost by Compose; the API and Ollama run on the host. Persistent checkpoints, feedback capture, comparison tools, and structured attribute enforcement are not implemented yet. See the [UI setup guide](copilotkit-ui.md) and [code walkthrough](code-walkthrough.md).

**Work:**

- Expose search, refinement, comparison, and feedback through FastAPI endpoints.
- Build a React interface for natural-language search, product cards, supported filters, feedback, and comparison.
- Evaluate the A2UI React renderer with a small integration spike; use it if it fits product-card and feedback interactions, otherwise use a straightforward React UI.
- Present missing catalogue data honestly and do not imply unsupported price filtering.

**Complete when:** a user can submit a query, inspect catalogue-backed results, refine them, compare a shortlist, and give feedback through the running application.

## Phase 6 — Reproducibility and completion

**Goal:** make the complete prototype straightforward to set up, evaluate, and maintain.

**Work:**

- Document environment setup, model downloads/configuration, database lifecycle, sample recreation, and application startup.
- Run the documented checks and retrieval evaluations; record outcomes and known limitations.
- Verify repeatable ingestion, embedding generation, and application startup using documented commands.
- Review data/image licensing and document redistribution constraints.

**Complete when:** a fresh setup can follow the documentation to run the prototype and reproduce its evaluation, with results, limitations, and data handling constraints recorded.

## Cross-phase decisions

- Use one LangGraph workflow, not a multi-agent architecture.
- Keep the catalogue sample bounded and configurable while iterating.
- Keep model names, vector dimensions, and retrieval weights configurable and recorded.
- Do not claim price or budget filtering without a verified price source.
- Preserve source data and represent missing values as missing.

## Current project checkpoint (2026-10-17)

- The working dataset is intentionally held at 1,000 added Outdoor & Garden records (1,221 distinct products total after deduplication with earlier samples). A broader 10,000-row fetch timed out at Hugging Face and was stopped; the existing sample was retained.
- Database verification found 1,221 ready Nomic text embeddings and 1,221 ready CLIP image embeddings. All 1,221 products have a title or description.
- The next work should stay on the 1,000-item scope: evaluate the new stateful search ? select ? refine flow, add structured physical-attribute handling, and improve the small evaluation set. Benchmark IVF/HNSW against exact search later if useful; do not broaden the data fetch unless the scope changes.
- The available executable surfaces are the CLI and the API-backed web UI. LangGraph context retention and the shared RAG step are implemented; structured attributes and persistent session storage remain future work.
- For a module-by-module account of current behavior, see [the code walkthrough](code-walkthrough.md).

## Recommendation behavior requirements

These are the primary user journeys to use when choosing the next implementation slice:

1. **Recommend from selected products.** After seeing five candidates, the user can select up to three products and request five similar products. Selected products become visual seeds and are excluded from their own recommendations. This flow is available in the CLI and the A2UI web prototype; arbitrary image uploads are not supported.
2. **Search and refine using physical attributes.** A natural-language request such as “blue dress” should return dresses matching the requested color. The user can then choose a style such as halter or refine the shade of blue, and subsequent recommendations should preserve the dress category and selected attributes while changing the results accordingly.
3. **Combine evidence from products and stated preferences.** Product selections provide examples of what the user likes; words such as color, style, size, material, or power source provide explicit intent. Retrieval should balance these signals and avoid relaxing an explicit attribute without telling the user.

For the next prototype step, add labeled examples covering product selection, attribute refinement, and combined selection-plus-attribute requests. Then extend retrieval with explicit structured attribute extraction and enforcement. Test exact text, image, and hybrid rankings against those cases before adding approximate indexes. The current catalogue is Outdoor & Garden, so the blue-dress journey is a target behavior for a later apparel slice or a suitable apparel fixture; do not infer that the current garden data can evaluate dress retrieval.
