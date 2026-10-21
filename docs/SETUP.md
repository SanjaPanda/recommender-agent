# Setup

## Requirements

- Windows with Python 3.12 and `uv`
- Docker Desktop with at least 1.5 GiB available for the PostgreSQL and web containers
- Ollama running locally
- Hugging Face CLIP model cached locally for the full hybrid ranker

## Configure and start

1. Copy `.env.example` to `.env`, choose a local PostgreSQL password, and keep `.env` private. Pydantic Settings validates the application configuration at startup.
2. Install Python and local image support:

   ```powershell
   uv sync --extra image
   ```

3. Start PostgreSQL and initialize the schema:

   ```powershell
   docker compose up -d db
   uv run python scripts/db/apply_migrations.py
   ```

4. Download the local Ollama chat and embedding models:

   ```powershell
   ollama pull gemma3:4b
   ollama pull nomic-embed-text
   ```

   For cached image captions, also run `ollama pull moondream`. Caption generation is offline and optional; recommendation requests never call the vision model.

Run a small resumable caption batch with `uv run recommender-describe-images --limit 8`. The catalog currently has 124 cached apparel descriptions and 2,049 pending; `--all` can take a long time on a laptop.

5. Cache the CLIP model once while network access is available:

   ```powershell
   uv run python -c "from transformers import CLIPModel, CLIPProcessor; name='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(name); CLIPProcessor.from_pretrained(name)"
   ```

   The runtime uses `local_files_only=True` and will not download model weights while serving requests.

6. Ensure the pinned train/test JSONL files exist under ignored `.data/shopify-selected-train/` and `.data/shopify-selected-test/`. If needed, fetch apparel-only samples with the commands in the root README.
7. Import records and build persistent retrieval vectors:

   ```powershell
   uv run python scripts/db/ingest_sample.py
   uv run python -m rag.embed_text
   uv run python -m rag.embed_images
   ```

   The importer ignores non-apparel rows and upserts by stable ID. Embedding scripts skip rows with the current model digest.

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

Run a shared-pipeline query with:

```powershell
uv run recommender-search "blue cotton dress" --limit 5
```

The first query in a new API process loads the CPU CLIP model. Later queries reuse that process-local model cache. If CLIP is unavailable, the agent warns and continues with product text and cached captions. Troubleshoot local services with `docker compose ps`, `ollama ps`, and `ollama list`.
