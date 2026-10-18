# Code walkthrough

This guide describes the current codebase. It is a **local-first retrieval prototype**: preparation scripts fetch and ingest a bounded catalogue sample, the RAG package embeds and searches products in PostgreSQL/pgvector, the agent package maintains recommendation context, and a Dockerized Next.js/A2UI client provides the UI.

## Repository layout

```text
scripts/
  data/       sample download and image extraction
  db/         migrations and catalogue ingestion
src/
  rag/        embeddings, retrieval, search, evaluation, product normalization
  agent/      LangGraph workflow, CLI, FastAPI, A2UI response operations
  ui/         Next.js app, package files, Dockerfile
```

Database and sample preparation stay outside the agent. `src/agent` calls the shared retrieval function in `src/rag`; the UI runs independently in Compose. The `pyproject.toml` console commands point to these organized packages, while preparation scripts run directly with `uv run python scripts/...`.

## What happens end to end

```text
Shopify dataset (pinned train split)
  | scripts/data/fetch_sample.py
  v
products.jsonl + local image files
  | scripts/db/ingest_sample.py
  v
PostgreSQL: products
  +-- recommender-embed-products -- Ollama Nomic --> text vectors
  +-- recommender-embed-images ---- CPU CLIP ------> image vectors
                                                   |
User text -- Nomic query vector -- pgvector text search --+
          +-- CLIP query vector ---- pgvector image search +--> hybrid rank

LangGraph session -- accumulated query + selected item --> RAG retrieval above
Labeled JSONL queries -------- evaluation scripts --------> ranking metrics
```

The database stores normalized product fields and the original source JSON. It stores vectors as separate rows by product, modality (`text` or `image`), and model. Image files remain on disk; product metadata records their paths and hashes.

## Source modules

### Data shaping and ingestion

- [`catalogue.py`](../src/rag/catalogue.py) defines `ProductRecord`, the normalized product representation. `clean_text` collapses whitespace and preserves missing values as `None`; `clean_categories` turns category lists into readable strings. `ProductRecord.from_row` maps source field names to app fields, creates a deterministic SHA-256 fallback ID if needed, and retains every original field in `raw_source`.
- [`fetch_sample.py`](../scripts/data/fetch_sample.py) streams a pinned revision of the Shopify train split, optionally filters by category prefix, writes JSONL and local images, and emits a manifest with row counts, missing values, and duplicate-title candidates. `identity_key` and `existing_sample_identities` let category expansions reuse IDs for records already in the original sample. `image_extension` picks a file extension from image bytes.
- [`ingest_sample.py`](../scripts/db/ingest_sample.py) reads JSONL rows and upserts product fields into PostgreSQL. `UPSERT_PRODUCT` makes reruns update existing product rows instead of duplicating them. `source_image_meta` extracts local image metadata; the complete row goes into `raw_source` for later category filtering and auditing.
- [`apply_migrations.py`](../scripts/db/apply_migrations.py) creates/checks the migration ledger, applies sorted SQL files once, and records each successful migration inside a transaction.

### Embedding generation

- [`embed_text.py`](../src/rag/embed_text.py) implements `recommender-embed-products`. `document_text` constructs the Nomic `search_document:` input from title and description. `model_digest` checks which installed Ollama model revision is active; `post_json` calls the local Ollama API. `main` selects products without a ready vector for the current model revision, batches calls, validates vector count and dimensions, and upserts vectors. `--force` recomputes all text vectors.
- [`embed_images.py`](../src/rag/embed_images.py) implements `recommender-embed-images`. `resolve_image` locates a saved image under one of the supplied roots. `main` loads CLIP on CPU, selects products missing image vectors, reads and converts images to RGB, embeds each batch, normalizes vectors for cosine similarity, and saves results. Missing paths and unreadable files are counted; `--force` recomputes vectors. Image libraries are an optional dependency group.

### Retrieval

- [`search.py`](../src/rag/search.py) implements `recommender-search`. `main` validates options, rejects budget/price queries because the data has no structured price field, embeds the query with Nomic, applies optional SQL filters, and orders candidates by pgvector distance. Text mode uses Nomic only. Hybrid mode also embeds query text with CLIP, retrieves image candidates, and combines the two rank lists using weighted reciprocal-rank fusion (RRF). RRF is used because raw similarity values from different models should not be treated as directly calibrated against one another. `--image-weight` adjusts image contribution; the default is deliberately conservative.
- [`search_by_examples.py`](../src/rag/search_by_examples.py) implements `recommender-search-by-examples`. It accepts one to three existing product IDs and optional natural-language refinement text, then calls the same retrieval function as the agent. It is a CLI inspection path, not a separate ranking implementation.
- [`retrieval.py`](../src/rag/retrieval.py) contains `retrieve_products`, the shared RAG retrieval path. It averages selected products' normalized CLIP vectors, embeds accumulated text with Nomic when present, applies optional category-prefix filtering, excludes selected examples, and fuses image and text rankings with weighted reciprocal-rank fusion. With no seed it performs text retrieval; with a seed it can use image similarity alone or blend it with refinement text.
- [`workflow.py`](../src/agent/workflow.py) defines `RecommendationState`, `_recommend`, `build_recommendation_agent`, and `run_recommendation_turn`. Each LangGraph thread retains query history, active selected products, and category scope. `_recommend` merges the new turn into that state and passes the resulting query context and seed IDs into `retrieve_products`. The default `MemorySaver` retains sessions only while that process is alive; a persistent database checkpointer is a later improvement.
- [`cli.py`](../src/agent/cli.py) implements `recommender-agent-chat`, an interactive harness around the graph. `/select N` chooses a result from the last list, `/refine TEXT` adds preference text, `/reset` clears that thread's context, and `/quit` exits. Pass `--category-prefix` to keep candidates in a cohesive category.

### Evaluation

- [`evaluate_text.py`](../src/rag/evaluate_text.py) implements `recommender-evaluate`. `read_queries` loads JSONL labels. `main` embeds each query, times embedding and database retrieval separately, checks the rank of labeled relevant IDs, and reports Hit Rate@k, MRR, top results, out-of-catalogue behavior, and an exploratory similarity-threshold sweep.
- [`evaluate_multimodal.py`](../src/rag/evaluate_multimodal.py) implements `recommender-evaluate-multimodal`. `read_queries` loads the category-slice labels. `normalized_text_features` obtains normalized CLIP text features. `metrics` computes Hit Rate@k and MRR. `main` compares text-only Nomic ranking with the RRF text+image ranking at a configurable image weight.
- [`phase2_queries.jsonl`](../data/phase2_queries.jsonl) and [`outdoor_garden_queries.jsonl`](../data/outdoor_garden_queries.jsonl) are hand-labeled evaluation inputs. Their relevance labels are small exploratory sets, not statistically representative ground truth.

## Database and runtime configuration

- [`001_initial_schema.sql`](../migrations/001_initial_schema.sql) enables pgvector and creates `products`, `product_embeddings`, `recommendation_sessions`, and `product_feedback`. Product attributes live in `products`; vectors are keyed by product, modality, and model in `product_embeddings`. Session and feedback tables lay groundwork for a later interactive workflow.
- [`compose.yaml`](../compose.yaml) runs PostgreSQL with pgvector and the production web frontend in Docker. The Python API and Ollama run on Windows.
- [`pyproject.toml`](../pyproject.toml) defines Python dependencies, the optional image dependency group, and maps command names to module `main` functions. LangGraph runs the recommendation workflow, and FastAPI serves it to the web client.
- [`src/agent/`](../src/agent/) and [`src/rag/`](../src/rag/) are the installable Python packages. The React app and its container files live in [`src/ui/`](../src/ui/).

## Core functions to focus on

For understanding the current recommendation behavior, read these in order:

1. [`ProductRecord.from_row`](../src/rag/catalogue.py) — how raw records become product fields and stable IDs.
2. [`document_text`](../src/rag/embed_text.py) and [`embed_text.main`](../src/rag/embed_text.py) — what text the model sees and how text vectors enter the database.
3. [`retrieve_products`](../src/rag/retrieval.py) - how query text and selected product examples become candidates, category filters apply, and modality ranks are fused.
4. [`_recommend`](../src/agent/workflow.py) - how follow-up turns update selected items and accumulated query context before calling RAG retrieval.
5. [`build_recommendation_agent`](../src/agent/workflow.py) and [`run_recommendation_turn`](../src/agent/workflow.py) - how LangGraph and per-thread checkpoints are used.
6. [`evaluate_retrieval.main`](../src/rag/evaluate_text.py) and [`evaluate_multimodal.main`](../src/rag/evaluate_multimodal.py) - how ranking quality is checked before changing retrieval behavior.

For data reliability, also study `fetch_sample.main`, `ingest_sample.main`, and the schema. Structured extraction and enforcement of product attributes remain future work; the current UI passes natural-language refinements into semantic retrieval.

## Important behavior and limitations

- Search uses exact pgvector ordering at this sample size. IVF and HNSW indexes have not been added; compare approximate indexes against exact rankings once there is a larger representative corpus.
- Text and image vectors are model-specific and stored separately. Regenerating with another model/revision creates or updates vectors according to the model key and digest logic.
- Category filters inspect `ground_truth_category` from retained source JSON, while regular `--category` filters the normalized category string.
- A similarity cutoff is a prototype control, not a calibrated confidence score. Evaluation data is hand-labeled and small.
- Budget filtering is explicitly unsupported because the source catalogue has no documented price field.
- The agent retains accumulated query text and the active selected product between turns within a running process, then passes them to shared RAG retrieval. It does not yet extract colors, sizes, materials, or power sources into structured hard constraints; those attributes currently influence semantic ranking only.
- The agent uses LangGraph's in-memory saver by default. Restarting the process loses its conversation checkpoints. The FastAPI and UI layers are implemented, but structured hard constraints and persistent checkpoints remain open.
- Current local database count: 1,221 products, with ready text and image vectors for all 1,221, as verified on 2026-10-17.

## Where to go next

Use [`README.md`](../README.md) for environment setup and runnable commands, [`phase-2-evaluation.md`](./phase-2-evaluation.md) and [`phase-3-prototype.md`](./phase-3-prototype.md) for measured retrieval results, and [`phases.md`](./phases.md) for the broader build sequence.
