# Database contract

PostgreSQL 17 and pgvector run in Compose service `db`, bound to loopback with a persistent named volume. FastAPI connects using the secret-wrapped `DATABASE_URL`; Compose uses `POSTGRES_*`. API and Ollama run on the host; Next.js runs in Compose.

## Setup and inspection

From the repository root, with `.env` configured:

```powershell
docker compose up -d db
uv run python scripts/db/apply_migrations.py
uv run python scripts/db/inspect_database.py
```

Migration filenames are recorded in `schema_migrations`. Apply pending migrations before starting the API; never rewrite applied migrations. Inspection is read-only and reports versions, migration names and counts without printing credentials. Migration 012 is required for catalog revision tracking. No new migration is required for the audit fixes.

## Tables and provenance

- `dataset_sources`: pinned source revision, source URL, recorded use decision and frozen normalization/quality configuration. Operational scan budgets may increase without changing source identity.
- `products`: stable source identity, normalized apparel taxonomy, raw source JSON, title, description, feature bullets, local image metadata, nullable historical price, source-record/image hashes and the versioned attributes document.
- `product_embeddings`: separate text/image vectors, model name, immutable model version, generated dimensions and ready/error status. Product deletion cascades to embeddings. One vector per product/modality/model name is retained; rebuilding overwrites that model's prior revision in batches.
- `agent_sessions`: durable graph snapshots and carts keyed by thread ID. Trace/candidate/request fields are transient.
- `catalog_revision`: singleton counter maintained transactionally by a statement trigger on product INSERT/UPDATE/DELETE/TRUNCATE.
- `product_feedback` and `recommendation_sessions`: retained legacy tables, outside the active request flow.

The active importer is `rag.catalog`, with bounded acquisition in `rag.download_catalog` and orchestration in `scripts/expand_catalog.py`. The Shopify utilities are compatibility interfaces, not the active pipeline. Source-linked products require source identity, a positive original line number and record hash; the database rejects taxonomy outside Apparel & Accessories. Importer checks also validate source descriptions, image inventory, safe local paths, decoded images and source-use records. Absent records are not deleted during import; changed source/image hashes invalidate their vectors.

## Retrieval and model compatibility

Every retrieval path independently constrains the authoritative apparel root. Exact vector SQL also requires the query encoder's immutable model revision and vector dimensions. Nomic tag resolution selects the exact installed tag (`:latest` when omitted), checking for an alias change across query encoding. CLIP query retrieval uses the loaded cached encoder revision. Image-only examples must share one nonempty revision and dimension; candidates must match that revision. Mixed or stale revisions are excluded, so interrupted rebuilds can reduce vector coverage until resumed. BM25 remains independently available inside the same eligibility rules; text-query embedding service failure still fails the request.

Importers and embedding CLIs share an advisory maintenance lock. Expansion also holds it through publication and keeps `.data/catalog/.promotion.json` until the metadata and ingestion report are published. Database commit and filesystem replacement are not atomic; rerunning expansion verifies the staged metadata hashes, idempotently imports it and completes pending file replacement. Keep the stage and catalog backups until recovery is complete. See [setup](SETUP.md#catalog-expansion-and-recovery).

## Sessions and caches

Session turns take a nonblocking, connection-scoped advisory lock for their thread. Short repeatable-read transactions load and save snapshots; model inference runs outside a transaction. A competing turn receives HTTP 409. Compare-and-save prevents overwriting an externally changed session, and a catalog revision change during inference rejects the stale turn. Closing the connection releases its advisory lock even on failure.

Cards store source-record/image hashes. Session resume refreshes current facts, removes invalid/changed recommendations and visual selections, and refreshes cart entries while clearing obsolete ranking scores. Older snapshots without matching provenance are invalidated. API serialization refreshes current apparel listing facts and image references independently. Historical rankings are not asserted against a changed listing.

Category counts use an eight-entry process cache keyed by database identity and committed revision. Revision/count reads share one read-only repeatable-read snapshot. BM25 caches immutable document tokens and statistics for the exact eligible document sequence. Restart API workers after database restore to discard process-local caches.

## Verification and operational limits

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
$env:RUN_DATABASE_TESTS='1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
Remove-Item Env:RUN_DATABASE_TESTS
```

The default suite skips opt-in PostgreSQL integration tests. Most use rollback fixtures; cross-connection cache/session tests commit temporary authored rows and clean them up. They can advance the catalog revision, so run against a development database, not a read-only audit target. No test downloads a catalog or rebuilds production embeddings.

The schema does not enforce a model/dimension registry. Runtime revision/dimension predicates protect retrieval, but administrative writers must still supply valid finite vectors. Optional input/preprocessing hash columns are not populated by the existing encoder CLIs. Historical price is not an API price filter; candidate attributes are unreviewed and do not enable hard filters. No ANN index or judged relevance benchmark is claimed.

The last read-only audit observed 2,000 products, 4,000 embedding rows and 11 applied migrations on 2026-10-24. These are observations, not fresh-install guarantees. Earlier reset/pilot milestones and backup references are preserved in [the historical database record](../observations/archive/2026-10-24-before-audit-fixes/database.md); do not follow historical next steps as current setup.
