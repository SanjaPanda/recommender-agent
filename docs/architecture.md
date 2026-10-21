# Architecture

## Scope

One local apparel recommender over the 2,173 unique Apparel & Accessories records in the pinned train/test samples. Data IDs are content-derived and stay consistent across source JSONL, PostgreSQL, retrieval vectors, LangGraph state, API responses, and UI cards.

## Request flow

```text
Train/test JSONL
  -> apparel-root validation and ID de-duplication
  -> idempotent PostgreSQL upsert
  -> cached Nomic text vectors + optional Nomic caption vectors + CLIP image vectors
  -> FastAPI request validation
  -> one LangGraph router node
  -> SQL apparel/brand/condition/title filters + hybrid RRF retrieval
  -> response node and validated product list
  -> A2UI gallery; selection/refinement starts the next turn
```

All retrieval SQL applies the Apparel & Accessories root constraint independently of LLM output. More specific apparel category paths, brand, new/used status, title product type, and title color use parameterized SQL. Unknown attributes remain unknown. Preference statements also enter semantic retrieval, and responses say clearly that unverified preferences are not guaranteed.

## Ranking and models

- **Nomic `nomic-embed-text`:** product title/description and query embeddings. A single query vector searches product text and cached image descriptions.
- **CLIP `openai/clip-vit-base-patch32`:** offline image vectors are queried with a cached local CLIP text encoder; selected product cards use the same image-vector space. The three ranked lists are fused with reciprocal-rank fusion (Nomic product text 0.50, captions 0.25, CLIP image 0.25; available modes are renormalized). Selected examples add their own CLIP image ranking.
- **Ollama language model:** one LangGraph turn router emits a Pydantic v2 `TurnEdits` contract, which application code validates and applies atomically. Optional summaries use the typed `Summary` response and are disabled by default. Category filters, cart actions, sorting, and selection checks remain deterministic.
- **Ollama image caption model:** offline only. It emits a Pydantic `VisualAttributes` JSON object; application code converts the validated fields to a caption vector. No vision inference runs per recommendation request.

CLIP loads lazily once per API process from the local model cache. If it is missing, the app logs a warning and returns Nomic text/caption results. Selected examples require their stored CLIP image vectors. No vector index is created at this catalog size: pgvector performs exact ranking over 2,173 rows.

## Persistence and services

PostgreSQL/pgvector stores product facts, embeddings, and complete LangGraph turn snapshots. Product embeddings have a foreign key with `ON DELETE CASCADE`. The application/API and Ollama run on the host; Compose runs PostgreSQL and the Next.js A2UI UI. Compose caps PostgreSQL at 1 GiB/1.5 CPUs and the UI at 512 MiB/1 CPU. The database volume is persistent.

Pydantic Settings loads the project `.env` and process environment at startup. `DATABASE_URL` is secret-wrapped. Ollama calls use typed request models, provider JSON Schema, Pydantic v2 validation, bounded request timeouts, and no implicit retry loop.

## Deliberate limits

- The catalog has no dependable price field, so price/budget filters are not supported.
- There is no arbitrary image upload path; CLIP queries use text or selected catalog products.
- Color hard filtering uses product titles. Image and semantic preferences are ranked, not verified.
- PostgreSQL sessions store conversation and cart snapshots. When old snapshots contain other-category product references, the API filters those product references while preserving the session row/history.
- Approximate vector indexes, a reranker, multiple agents, and hosted inference are out of scope for 2,173 products.
