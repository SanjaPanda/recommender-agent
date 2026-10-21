# Apparel Recommender Agent

A local apparel discovery app for **2,173** Apparel & Accessories products. It combines deterministic PostgreSQL filters, Nomic product-text retrieval, CLIP image retrieval, one LangGraph agent, Ollama structured outputs, and an A2UI product gallery.

## Run locally

Requirements: Python 3.12, `uv`, Docker Desktop, and Ollama. Copy `.env.example` to `.env`, set a local PostgreSQL password, then install and start:

```powershell
uv sync --extra image
docker compose up -d db
uv run python scripts/db/apply_migrations.py
uv run python scripts/db/ingest_sample.py
uv run python -m rag.embed_text
uv run python -m rag.embed_images
```

Pull the configured Ollama model and text embedder once:

```powershell
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

The CLIP model is loaded from the local Hugging Face cache at request time. Download/cache `openai/clip-vit-base-patch32` before running the app; request handling never downloads model files. If CLIP files are unavailable, retrieval falls back to product text.

Start the API and UI in separate terminals:

```powershell
uv run recommender-agent-api
docker compose up -d --build web
```

Open <http://localhost:3000>. The UI displays relevance-gated product cards, accepts visual selections, lets the user refine preferences, and exposes a per-turn agent trace. If no strong products match, it explains that and shows the closest alternatives in the selected category. `#buynew` starts a fresh apparel search while keeping the cart.

## Dataset and database

The importer reads the pinned train and test samples under ignored `.data/` and rejects every record outside the `Apparel & Accessories` taxonomy. The current files contain 2,173 unique apparel records: 1,705 train and 468 test. Their deterministic IDs match the retained database apparel IDs. The importer upserts by product ID and is safe to rerun.

If the samples are absent, fetch apparel-only splits. The versioned content-identity map in `data/apparel_product_ids.json` preserves the existing 2,173 source IDs, including IDs inherited from the original sample:

```powershell
uv run python scripts/data/fetch_sample.py --rows 50000 --split train --output-dir .data/shopify-selected-train
uv run python scripts/data/fetch_sample.py --rows 50000 --split test --output-dir .data/shopify-selected-test
```

Database schema, backup/cleanup procedure, row counts, and retained objects are in [database notes](docs/database.md). The product source does not provide a reliable price field; price and budget filters are unsupported.

## Search and validation

Inspect the same retrieval path used by the agent:

```powershell
uv run recommender-search "blue cotton dress" --limit 5
uv run recommender-search "lightweight jacket" --reference-product-id PRODUCT_ID --limit 5
```

Run offline tests:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

See [setup and troubleshooting](docs/SETUP.md), [agent behavior](docs/Agent.md), and [architecture](docs/architecture.md).
