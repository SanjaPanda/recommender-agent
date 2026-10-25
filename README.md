# Multimodal Product Discovery

Local apparel discovery with a 2,000-product private Amazon catalog. BM25 searches listing title, description and all feature bullets; weighted RRF combines keyword, Nomic text and CLIP image rankings. Cards retain similarity percentages, explicit listing checks, source details and a per-signal RRF breakdown. Retrieval quality has not been measured on a held-out judged benchmark.

The existing application remains: FastAPI, one LangGraph workflow, PostgreSQL/pgvector, Ollama, CLIP and a Next.js/A2UI interface. Docker Compose runs the database and web UI; Python API and Ollama run on the host.

See [setup](docs/SETUP.md), [current architecture](docs/architecture.md), [agent behavior](docs/Agent.md), and [database state](docs/database.md).

## Development

```powershell
uv sync --extra image
docker compose up -d db
uv run python scripts/db/apply_migrations.py
uv run recommender-agent-api
docker compose up -d --build web
```

Configure `.env` using `.env.example` first. Open <http://localhost:3000>. The database contains 2,000 products with validated local images, 2,000 ready Nomic text vectors and 2,000 ready CLIP image vectors. Use the database inspection command in setup to verify the local state. Search results expose BM25 and RRF scores in **Similarity details**. Product-type and material relevance limitations remain.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The new local Amazon-shaped pilot importer is documented in [setup](docs/SETUP.md#curated-local-pilot-importer). A [bounded downloader](docs/SETUP.md#bounded-downloader-and-pilot-report) is implemented; the private catalog has expanded from 50 to 2,000 records under the same recorded personal-use decision. This is a bounded sequential sample; representativeness and independently judged ranking quality remain unmeasured. The existing Shopify fetch/import utilities are retained for compatibility, but are not the new dataset pipeline. Do not run them as part of normal setup. User article drafts under `.medium/` are independent work and were preserved. Superseded engineering records are under `observations/archive/`, excluded from current project guidance.


Audit fixes reconcile color follow-ups, reject incomplete compound interpretations,
constrain phrase evidence to primary title segments, enforce vector revisions and
dimensions, and prevent overlapping session turns from holding long transactions.
Changed catalog snapshots invalidate historical scores; expansion interruptions can
be recovered using the journal described in setup. These safeguards are tested
contracts, not evidence of judged relevance accuracy or a latency speedup.
