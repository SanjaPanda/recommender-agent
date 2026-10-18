# Recommender Agent

A local-first product discovery prototype over the Shopify Product Catalogue, built around one LangGraph workflow, deterministic retrieval tools, FastAPI, PostgreSQL/pgvector, and Ollama.

## Catalogue sample

A bounded training sample is stored under `.data/shopify-catalogue-sample/` (ignored by Git): 250 source records and 250 local image files, about 40 MB total. `manifest.json` pins the dataset revision and records missing values and duplicate-title candidates. Missing fields remain missing. The source has no documented price field.

Re-create the sample with:

```powershell
uv run python scripts/data/fetch_sample.py --rows 250
```

The command streams only the requested prefix of the train split. First use requires Hugging Face network access. Product images are saved separately; each JSONL row records its relative path and SHA-256.

## Code layout

The implementation is organized by responsibility: [`src/rag/`](src/rag/) contains retrieval and embedding code, [`src/agent/`](src/agent/) contains the LangGraph workflow and API/CLI, and [`src/ui/`](src/ui/) contains the Next.js app. One-off catalogue and database preparation scripts live under [`scripts/`](scripts/). See the [code walkthrough](docs/code-walkthrough.md) for the module map and core functions.

## Start the local database

Docker Desktop uses its WSL 2 backend. Confirm Docker Desktop has finished starting and `docker info` can reach the Linux engine, then start only the PostgreSQL/pgvector service:

```powershell
docker compose up -d db
docker compose ps
uv run python scripts/db/apply_migrations.py
uv run python scripts/db/ingest_sample.py
```

The ignored `.env` file contains local database settings; `.env.example` documents the variables. Compose binds PostgreSQL to localhost and stores database files in the named `recommender_pgdata` volume. Ollama and the Python application stay on Windows. If the Docker engine reports an error, resolve Docker Desktop startup before running migrations or ingestion.

The migration enables `vector` and creates product, embedding, session, and feedback tables. The importer upserts by deterministic source-row ID, so rerunning it updates the same records. Image bytes remain in local files; the database stores relative paths and hashes, not binary images.

## Current setup (2026-10-18)

- PostgreSQL/pgvector is running in Docker Desktop. The database contains 8,671 distinct products, including all 7,483 Apparel & Accessories, Health & Beauty, Baby & Toddler, and Animals & Pet Supplies records from both source splits.
- All 8,671 products have validated 768-dimensional Nomic text vectors and 512-dimensional CLIP image vectors, including all 7,483 products in the four selected categories.
- The 1,000-record Outdoor & Garden expansion saved 1,000 images (~182 MiB); end-to-end embedding time and missing-value counts are in the [scale-up checkpoint](docs/scale-up-checkpoint.md).
- The catalogue has no documented price field, so price and budget filtering are unsupported.

**Next:** keep development on the existing 1,000-item expansion. Evaluate the query → select examples → refine LangGraph flow and add structured physical-attribute handling. The interactive harness keeps context in memory while the process runs. The CopilotKit/A2UI web prototype now adds a five-card visual selection and free-text refinement flow; its LangGraph checkpoint is still in memory while the API process runs. Add labeled cases and compare rankings before considering IVF/HNSW at this scale. A broader 10,000-row fetch was stopped after a Hugging Face timeout. See the [recommendation requirements](docs/phases.md#recommendation-behavior-requirements), [scale-up checkpoint](docs/scale-up-checkpoint.md), [code walkthrough](docs/code-walkthrough.md), and [web agent setup](docs/copilotkit-ui.md).

## Text retrieval baseline (Phase 2)

The initial local model is `nomic-embed-text` (Ollama; 768 dimensions). It uses a `search_document:` prefix for product text and `search_query:` for search queries. Download it with `ollama pull nomic-embed-text`, then generate product vectors and search with:

```powershell
uv run recommender-embed-products
uv run recommender-search "lightweight rain jacket for hiking" --limit 5
```

The embedding command skips rows already stored for the same Ollama model digest. Pass `--force` to refresh them or `--limit 10` to work on a smaller subset. Search supports `--brand`, `--category`, and `--secondhand true|false`, and hides matches below a 0.65 similarity cutoff by default. Adjust it with `--min-similarity N`. Budget requests return an explicit warning because the catalogue has no structured price field.

Run the expanded smoke evaluation with `uv run recommender-evaluate --k 5`. See the [evaluation notes](docs/phase-2-evaluation.md) for results, out-of-catalogue behavior, and limitations.

## Image retrieval prototype (Phase 3)

The first image encoder is OpenAI CLIP ViT-B/32 through Transformers, running on CPU. It maps images and text into the same 512-dimensional space. Install the optional image dependencies and embed a small batch first:

```powershell
uv sync --extra image
uv run recommender-embed-images --limit 8 --batch-size 1
```

The command reads the saved image paths from product metadata and checks the original and Outdoor & Garden sample directories. Use `--image-root PATH` for another sample directory, repeating the option when needed. `--limit` caps the number of candidate products; existing vectors for the same model revision are skipped. The first run downloads model weights (about 605 MB); inference runs on CPU and can be slow on older hardware. Image vectors are stored separately from Nomic text vectors because they use different dimensions and encoders.

Hybrid search combines Nomic text ranking with CLIP query-to-image ranking using weighted reciprocal-rank fusion:

```powershell
uv run recommender-search "compact garden tools for pruning" --mode hybrid --limit 5
```

Use `--image-weight 0.1` to adjust the image contribution (0 through 1). This is only a conservative prototype default: on the first 10-query category slice, it produced effectively the same metrics as text-only. Compare the same labeled queries in text and hybrid modes before treating this weight or the similarity cutoff as calibrated.

The first eight-item CPU run and its limitations are recorded in the [Phase 3 prototype notes](docs/phase-3-prototype.md).

Users can select one to three existing product cards as visual examples and retrieve similar products by their averaged CLIP image vectors:

```powershell
uv run recommender-search-by-examples `
  --reference-product-id 76d733a1f1fb392e9dc52d4562bd37175d700613168edd2249df8d27214d9c81 `
  --reference-product-id fe7c4f1533ef66bc64e79daaa1d7866a143b5729f425b67610ef463336079645 `
  --ground-truth-category-prefix "Home & Garden > Lawn & Garden" --limit 5
```

This prototype accepts catalogue product IDs whose image vectors already exist. The agent CLI can pass one selected result or a group of up to three examples.

## Stateful recommendation agent

Run the LangGraph-backed interactive harness to keep a request and selected product in context across turns:

```powershell
uv run recommender-agent-chat --category-prefix "Home & Garden > Lawn & Garden"
```

Enter a request, then use `/select 2` to get five similar products from the second result, `/refine lightweight and battery powered` to add a preference, `/reset` to clear the session, or `/quit` to exit. The workflow blends the selected product image vector with accumulated text context through the shared RAG pipeline. The default checkpoint is in memory and lasts only while the process runs. Physical attributes currently affect semantic ranking but are not yet extracted into hard constraints.


## LLM recommendation workflow (2026-10-18)

The [agent workflow](src/agent/workflow.py) now routes through SQL-only or filtered text/image retrieval, conditionally verifies preferences, and generates a grounded summary. There is no automatic retrieval refill loop. At most 20 candidates enter verification and at most five eligible products are displayed. Unknown hard requirements are excluded from verified results.

Run `ollama pull gemma3:4b` once. Configure the local model settings shown in [.env.example](.env.example), apply migrations with `uv run python scripts/db/apply_migrations.py`, then start `uv run recommender-agent-api` and `docker compose up -d --build web`. No hosted inference or external API key is needed for the configured Ollama path. The model download needs internet access; inference uses localhost. The 4B vision model may be slow on this 8 GB machine.

The browser opens or resumes a PostgreSQL-backed session. `#buynew` clears shopping preferences and examples while preserving the cart. Selecting examples and adding to the cart are separate actions. Session IDs are stored in browser local storage; this remains a localhost prototype without user authentication or multi-user access control.

Verification has a 40-call ceiling and stops starting calls after a 120-second budget; each in-flight model call has its own 120-second timeout. Remaining unresolved hard requirements stay uncertain. Model calls are not retried. A summary failure falls back to factual result counts, while a router failure returns an explicit error and preserves the previous saved session. Visual attributes are checked on demand and are not cached yet. Session persistence saves complete turns, not intermediate execution checkpoints.

Run behavioral checks with `.venv/Scripts/python.exe -m unittest discover -s tests -v`. See the [finalized design](docs/architecture.md#finalized-agent-design-2026-10-18).
