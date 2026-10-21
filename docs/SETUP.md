# Setup

## Requirements

- Windows with Python 3.12 and `uv`
- Docker Desktop for PostgreSQL and the web UI
- Ollama running locally
- Hugging Face CLIP model cached locally for the hybrid ranker

## Configure and start

1. Copy `.env.example` to `.env`, set a local PostgreSQL password, and keep `.env` private. Pydantic Settings validates the application configuration at startup.
2. Install Python and local image support:

   ```powershell
   uv sync --extra image
   ```

3. Start PostgreSQL. For an existing database, first make and validate a fresh backup, inspect its counts, and only then apply pending migrations. Migration 009 removes stored image descriptions and their vectors:

   ```powershell
   docker compose up -d db
   uv run python scripts/db/apply_migrations.py
   ```

4. Download the local Ollama chat and text-embedding models:

   ```powershell
   ollama pull gemma3:4b
   ollama pull nomic-embed-text
   ```

5. Cache the CLIP model once while network access is available:

   ```powershell
   uv run python -c "from transformers import CLIPModel, CLIPProcessor; name='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(name); CLIPProcessor.from_pretrained(name)"
   ```

   Runtime loading uses `local_files_only=True` and does not download weights while serving requests.

6. Ensure pinned train/test JSONL files exist under ignored `.data/shopify-selected-train/` and `.data/shopify-selected-test/`. If needed, fetch apparel-only samples with the commands in the root README.
7. Ingest records and build persistent retrieval vectors:

   ```powershell
   uv run python scripts/db/ingest_sample.py
   uv run python -m rag.embed_text
   uv run python -m rag.embed_images
   ```

   Ingestion rejects non-apparel records and upserts stable product IDs. Embedding scripts skip rows with the current model digest.

8. Start the API and web UI:

   ```powershell
   uv run recommender-agent-api
   docker compose up -d --build web
   ```

   Open <http://localhost:3000>.

## Tests and diagnostics

Run unit tests with:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Run a query through the same retrieval code as the agent:

```powershell
uv run recommender-search "blue cotton dress" --limit 5
```

The first query in a new API process loads CLIP; later queries reuse its process-local cache. If CLIP files are unavailable, the agent warns and continues with product-text retrieval. Troubleshoot local services with `docker compose ps`, `ollama ps`, and `ollama list`.
