# Multimodal Apparel Discovery

A local recommendation application for an Amazon Reviews 2023 Apparel & Accessories catalog, bounded to 2,000 products. Search with text, refine preferences, select up to three products as visual examples, inspect listing details and explanations, and maintain a persistent cart.

BM25, Nomic text embeddings and CLIP image embeddings contribute separate ranked lists combined with weighted reciprocal-rank fusion. A local Ollama router returns updated shopping state and search intent in one call; cart buttons update state directly. PostgreSQL/pgvector stores catalog data, vectors and sessions; Next.js renders the web/A2UI interface.

## Start

Requires Python 3.12, uv, Docker Desktop and Ollama. Configure `.env` from `.env.example` first.

```powershell
uv sync --extra image
docker compose up -d db
uv run python scripts/db/apply_migrations.py
uv run python scripts/db/inspect_database.py
ollama pull gemma3:4b
ollama pull nomic-embed-text
uv run recommender-agent-api
```

In another terminal:

```powershell
docker compose up -d --build web
```

Open <http://localhost:3000>. The API and Ollama run on the host; PostgreSQL and the web UI run in Docker. A fresh checkout needs private catalog assets and compatible embeddings before populated searches work. Follow [setup](docs/SETUP.md) for catalog preparation and the CLIP model cache.

## Repository

| Path | Responsibility |
| --- | --- |
| `src/agent/` | Contracts, local-model calls, LangGraph workflow, sessions and API |
| `src/rag/` | Acquisition, normalization, import, embeddings and retrieval |
| `src/ui/` | Next.js/A2UI web interface |
| `migrations/` | Current schema and ordered schema changes |
| `scripts/` | Database maintenance and application checks |
| `tests/` | Application and database regression tests |
| `docs/` | [Architecture](docs/architecture.md), [agent behavior](docs/Agent.md), [database](docs/database.md) and [setup](docs/SETUP.md) |
| `observations/` | Engineering decisions and verified checkpoints |
| `archive/` | Reference material outside normal development and setup |

## Validate

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

See [database verification](docs/database.md#verification) for opt-in integration tests and [frontend checks](docs/SETUP.md#validation).

For reproducible retrieval comparisons with text queries, selected images and
descriptions, see the [experiment notebooks](notebooks/README.md).

The catalog has no dependable price or condition facts, so budget and new/used filtering are unsupported. Material and visual preferences guide ranking without guaranteeing product facts. The sequential catalog sample and relevance thresholds have not been evaluated against independently judged relevance labels. Similarity percentages are not match probabilities.
