# Architecture and Phase 1 findings

## Project objective

Build a local-first product discovery prototype over the Shopify Product Catalogue. Users describe needs in natural language, browse catalogue-backed product cards, refine results with feedback and supported filters, and compare a shortlist.

The project uses a single stateful LangGraph workflow with deterministic retrieval and filtering tools. It will not use Google ADK or a multi-agent design.

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

