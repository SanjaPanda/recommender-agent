# Runbook

Commands here are from current project documentation/source. PowerShell examples assume the repository root. Keep `.env` private and do not copy its values into logs.

## Prerequisites

- Windows, Python 3.12, `uv`, Docker Desktop, and Ollama.
- CLIP weights in the local Hugging Face cache.
- Pinned train/test product JSONL and image files under ignored `.data/` directories.
- A private PostgreSQL password and matching `DATABASE_URL` in `.env`.

## Environment and dependencies

From the repository root:

```powershell
Copy-Item .env.example .env
# Edit .env locally; do not paste secrets into the chat or commit them.
uv sync --extra image
```

`uv sync` is documented, not run here. `--extra image` installs Pillow, Torch, and Transformers.

## Fetch source samples if absent

From the repository root, with network access and dependencies installed:

```powershell
uv run python scripts/data/fetch_sample.py --rows 50000 --split train --output-dir .data/shopify-selected-train
uv run python scripts/data/fetch_sample.py --rows 50000 --split test --output-dir .data/shopify-selected-test
```

The script writes JSONL, image files, and a manifest for apparel rows. Commands are documentation-derived and not executed here.

## Models and local services

```powershell
ollama pull gemma3:4b
ollama pull nomic-embed-text
docker compose up -d db
```

Use model names matching local config. Compose starts PostgreSQL/pgvector on loopback. These commands are unverified here.

Cache CLIP weights while network access is available:

```powershell
uv run python -c "from transformers import CLIPModel, CLIPProcessor; name='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(name); CLIPProcessor.from_pretrained(name)"
```

Runtime loading uses `local_files_only=True`. This command was not run here.

## Database and vector preparation

For an existing database, follow the project backup/count inspection procedures before applying migrations. Migration 009 drops retired image-description columns and vectors. From repo root:

```powershell
uv run python scripts/db/apply_migrations.py
uv run python scripts/db/ingest_sample.py
uv run python -m rag.embed_text
uv run python -m rag.embed_images
```

These are state-changing commands and are unverified in this task. The importer filters to apparel and upserts by stable ID. Embedding jobs call local Ollama/CLIP and write PostgreSQL vectors.

## Start the application

Use separate terminals from repository root:

```powershell
uv run recommender-agent-api
```

```powershell
docker compose up -d --build web
```

Open `http://localhost:3000`; API is on `http://127.0.0.1:8000`. Both commands are unverified here.

## Search and tests

From repository root:

```powershell
uv run recommender-search "blue cotton dress" --limit 5
uv run recommender-search "lightweight jacket" --reference-product-id PRODUCT_ID --limit 5
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Search uses the same retrieval function as the agent. The test command is repository-mandated. None were run for this documentation pass.

## Diagnostics and Docker scope

```powershell
docker compose ps
ollama ps
ollama list
```

Check database container health, active Ollama models, and installed model tags. `/api/health` checks API liveness only, not database or model readiness.

`scripts/check_agent_live.py` starts a temporary API, creates a test session, and writes logs/reports; its default mode may invoke models. `scripts/check_router_edits.py` calls the configured router and writes a report. Review their source before running. Neither was executed here.

Compose defines `db` and `web`; API and Ollama are host processes. The persistent DB volume is `recommender_pgdata`.

## End-to-end verification update (2026-10-21)

From the repository root, the following were executed successfully:

- `.venv/Scripts/python.exe -m unittest discover -s tests -v` — 62 tests passed.
- `.venv/Scripts/python.exe -m rag.search "blue cotton dress" --limit 1` — hybrid retrieval returned a product with both text and CLIP similarity scores. This direct module invocation was used because `uv run` could not acquire the global uv cache lock in the sandbox.
- `scripts/check_agent_live.py` — all 16 checks passed, including apparel/vector integrity, selected-example retrieval, category/session initialization, brand-filtered cards, cart add/remove, API restart durability, brand replacement, and reset. The script cleaned up its own session.
- A direct `/api/discover` request for `blue cotton dress` returned five strong matches with router/retrieval/response trace and A2UI operations. Its temporary session was removed afterward. In that API process, retrieval took 76,345 ms, including cold model initialization; this is one observation, not a benchmark.
- `docker compose up -d --build --no-deps web` — the Next.js production build completed, and the UI served HTTP 200. API health also returned HTTP 200.

The read-only database inspection showed 2,173 apparel products, 2,173 ready text vectors, 2,173 ready image vectors, and the eight checked-in migrations recorded as applied. Ingestion, migration, and embedding jobs were therefore not rerun; they write database state and were unnecessary for this verification. The `uv run` search command remains unverified in this sandbox because of the uv cache-lock access error. The currently running API and Compose web service were left available after verification.
