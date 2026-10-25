# Database contract

PostgreSQL 17 with pgvector runs in Compose service `db`, bound to loopback with a persistent volume. FastAPI and local maintenance commands use secret-wrapped `DATABASE_URL`; Compose uses matching `POSTGRES_*` credentials. API and Ollama run on the host, and Next.js runs in Docker.

## Initialize and inspect

From the repository root with `.env` configured:

```powershell
docker compose up -d db
uv run python scripts/db/apply_migrations.py
uv run python scripts/db/inspect_database.py
```

`migrations/schema.sql` is the complete current schema for a new installation. Descriptively named changes in `migrations/` are registered in application order in `scripts/db/apply_migrations.py`. `require_source_name.sql` makes explicit source identity mandatory without changing existing rows.

The runner holds a transaction-scoped PostgreSQL advisory lock and commits the complete plan atomically. Success messages appear only after commit. `schema_migrations` records stable filenames; applied entries are not renamed or rewritten. Existing databases complete any required numbered history using unchanged, Git-tracked SQL under `archive/migrations/`, then apply current changes. A fresh database initializes directly from `schema.sql` without replaying the archive.

Initialization refuses to overwrite products without recorded migration history. Before taxonomy-enforcing upgrades, a locked preflight rejects missing/out-of-scope taxonomy with an actionable count and preserves the records. Inspect and resolve those records explicitly. Do not run archived SQL directly or edit applied SQL. Add future changes as new descriptive files and append them to `CHANGES` in the runner.

Inspection is read-only and reports versions, migration entries and row counts without exposing credentials.

## Tables and provenance

| Table | Contract |
| --- | --- |
| `dataset_sources` | Pinned source/revision, source URL, recorded use evidence and normalization/quality configuration |
| `products` | Stable identity, authoritative apparel taxonomy, source text/features/details, local image metadata and source/image hashes |
| `product_embeddings` | Separate text/image vectors, immutable model revision, generated dimension and ready/error status |
| `agent_sessions` | Durable shopping snapshots and cart state keyed by thread ID |
| `catalog_revision` | Singleton counter incremented transactionally by a product-write statement trigger |
| `product_feedback`, `recommendation_sessions` | Persistence structures retained for database compatibility; unused by active requests |

Source-linked products require a source ID, upstream product ID, positive original line number and source-record hash. The database rejects missing/non-apparel taxonomy. Historical price is nullable source evidence, not a supported price filter. The nullable condition column is retained for database compatibility and is not used by the active application. Unknown condition never means new.

Import validates source facts, use evidence, image provenance/decoding and safe local paths. Source/curation policy is immutable for an existing source identity; operational scan budgets can increase. Missing source records are not deleted. Changed source/image hashes invalidate vectors in the import transaction.

## Vector compatibility and maintenance

Text/image embeddings have a cascading product foreign key and one row per product/modality/model name. Rebuilds update that model's stored revision in batches. Runtime SQL requires the query encoder's immutable revision and dimension. Selected image examples must agree on both; partial rebuilds can reduce coverage until resumed.

Catalog import, publication and embedding commands share a maintenance advisory lock. Publication uses a hash-checked journal to recover database/filesystem interruption through idempotent replay. Keep stages, images, journals and backups until recovery completes. See [maintenance and recovery](SETUP.md#catalog-maintenance-and-recovery).

## Sessions and caches

Turns take a nonblocking connection-scoped advisory lock per session. Short repeatable-read transactions load/save state; inference runs outside transactions. Compare-and-save and catalog-revision checks reject stale writes with HTTP 409. Closing a failed connection releases its session lock.

Cards store source/image hashes. Resume removes ineligible or changed recommendations and visual selections, refreshes cart facts, and clears obsolete scores. Unsupported saved condition filters are dropped before plan validation; non-null condition constraints clear recommendations/selections and explain unsupported condition. Cart identity, session rows and history remain. Transient route/query_context fields are no longer persisted; query_context is derived from the saved plan on resume. Existing snapshots need no bulk migration. Read-only resume does not rewrite stored JSON; a successful turn saves normalized state.

Cart contents and recommendations are mutually exclusive. Resume removes cart IDs from saved cards and selections and converts older cart-action plans to search plans while retaining their constraints. Cart updates run through the same session lock and compare-and-save protection as searches, without model inference or schema/data migration.

Category counts are cached by database identity and committed revision. Reads of revision/counts share one read-only repeatable-read snapshot. BM25 caches immutable tokens/statistics for the exact eligible document sequence. Restart API workers after restoring a database.

## Verification

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
$env:RUN_DATABASE_TESTS='1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
Remove-Item Env:RUN_DATABASE_TESTS
```

Default tests skip opt-in database checks. Schema initialization/upgrade tests create and remove a uniquely named disposable database; the test role needs `CREATEDB` and permission to install pgvector there. Other integration tests roll back fixtures or clean temporary authored records/sessions and can advance the catalog revision. Run them against a development installation. They do not acquire a catalog or rebuild application vectors.

Runtime revision/dimension checks are not a schema-level model registry. Administrative vector writers must supply finite compatible vectors. Optional preprocessing/input hash columns are not populated by the encoder commands. No approximate index or judged relevance benchmark is assumed.
