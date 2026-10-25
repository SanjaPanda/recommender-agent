# Setup

Requires Python 3.12, uv, Docker Desktop and local Ollama. Run commands from the repository root. Copy `.env.example` to `.env`, set matching database credentials and keep `.env` private.

```powershell
uv sync --extra image
docker compose up -d db
uv run python scripts/db/apply_migrations.py
uv run python scripts/db/inspect_database.py
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

Compose exposes PostgreSQL 5432 and the web UI 3000 only on loopback. Its persistent database volume is retained during normal shutdown; avoid `docker compose down -v`. The private catalog is not committed: a fresh checkout requires local acquisition/import and embedding before populated searches work. Existing installations do not need to reacquire or rebuild unchanged compatible data.

CLIP runtime uses locally cached `openai/clip-vit-base-patch32`. Populate the cache once if needed:

```powershell
uv run python -c "from transformers import CLIPModel, CLIPProcessor; name='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(name); CLIPProcessor.from_pretrained(name)"
```

Runtime does not download CLIP weights. Missing query CLIP permits Nomic/BM25 ranking; visual examples require compatible stored image vectors. `EMBEDDING_MODEL` supports explicit Ollama tags; an omitted tag means `:latest`. Updating model weights requires rebuilding the corresponding vectors. See [revision enforcement](database.md#retrieval-and-model-compatibility).

## Start and test the UI

Start the host API in one terminal:

```powershell
uv run recommender-agent-api
```

In another terminal:

```powershell
docker compose up -d --build web
docker compose ps
```

Open <http://localhost:3000>; read-only API liveness is <http://127.0.0.1:8000/api/health>. Try a product search, a color follow-up, selecting up to three visual examples and explicit cart actions. Expand product details, similarity details and the interpretation/trace panels. Cosine percentages are not match probabilities. Concurrent turns on one session return a retryable HTTP 409; wait for its current request to finish.

New textual requests can be slow with local CPU inference. Saved interpretations are reused only when the effective query/preferences are unchanged. Color follow-ups remove superseded color clauses from that query and preferences. A catalog change invalidates historical cards/scores; search again to refresh. These checks do not establish material or image truth.

## Validation

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

For opt-in database checks, use the development database commands in [database verification](database.md#verification-and-operational-limits). Some commit temporary authored fixtures and clean them up; they are not read-only checks. For frontend checks, install locked dependencies from `src/ui` with `npm.cmd ci`, then run `npm.cmd exec tsc -- --noEmit --incremental false` and `npm.cmd run build`. Docker's build installs its own locked dependencies.

Live diagnostics such as `scripts/check_agent_live.py` and `scripts/profile_search_latency.py` require the populated database, compatible encoders and Ollama. They create isolated sessions; the profiler writes `.codex-work/search-latency-profile.json` by default. They are not required for an offline unit run. Past timing observations are under `observations/experiments/`; no new speedup or relevance accuracy follows from passing tests.

## Curated local pilot importer

`rag.catalog` imports prepared Amazon metadata and one selected validated image per product under `.data/catalog/`. It does not acquire data. The manifest records immutable source identity, use decision, original source line mappings and explicit images. This illustrative template requires real values; do not overwrite an existing manifest with placeholders:

```json
{
  "source_id": "amazon-clothing-pilot",
  "source_name": "McAuley-Lab/Amazon-Reviews-2023",
  "revision": "REPLACE_WITH_PINNED_COMMIT",
  "source_url": "REPLACE_WITH_EXACT_METADATA_URL",
  "rights_status": "permitted",
  "rights_evidence": "REPLACE_WITH_REVIEWED_TERMS_AND_USE_DECISION",
  "records": "records.jsonl",
  "images": {
    "REPLACE_WITH_PARENT_ASIN": {
      "local_path": "images/selected-image.jpg",
      "source_url": "REPLACE_WITH_HTTPS_URL_FROM_SOURCE_RECORD"
    }
  },
  "max_products": 50,
  "max_lines": 100000,
  "max_bytes": 250000000,
  "min_description_words": 50
}
```

The recorded active source-use decision is personal use; it is not an upstream license grant. Use `personal_use` with the actual recorded evidence when applicable. Normalization accepts only supported apparel branches, requires at least 50 description words and nonempty string-valued details, and preserves every feature bullet. Decoded JPEG/PNG/WebP images must be at least 224 pixels per side, at most 20 million pixels and at most 10 MB; selected URLs must belong to the source row. Traversal and symlink escapes are rejected. Bounds are at most 2,000 products, 100,000 lines and 250 MB metadata, with a 1 MB prepared-record cap.

```powershell
uv run python -m rag.catalog .data/catalog/manifest.json
uv run python -m rag.embed_text
uv run python -m rag.embed_images
uv run python scripts/db/inspect_database.py
```

These commands write products/vectors; do not run them merely to inspect an existing installation. Importing absent records does not delete existing products. Changed source/image hashes invalidate vectors; embedding jobs skip unchanged ready vectors for the same revision and commit in batches. Optional `--limit` and `--batch-size` bound embedding work; `--force` rebuilds. Run each module with `--help` for its current interface. BM25 needs no separate embedding or persistent index build.

## Bounded downloader and pilot report

For a new acquisition, prepare a separate manifest-shaped plan with `images: {}`, real source-use evidence and the pinned metadata URL/revision. Then run:

```powershell
uv run python -m rag.download_catalog .data/pilot-plan.json
```

The downloader refuses an occupied output and never connects to PostgreSQL. It validates HTTPS hosts, redirects, public DNS, response/image byte budgets, image decoding and original source line references. `pilot-report.json` records rejection reasons, budgets, stop reason and target deficit. Use the expansion wrapper below for an existing catalog rather than replacing its assets. The optional `evidence-v2` quality policy is not active for the current catalog; unreviewed attribute candidates do not enable hard filters.

## Catalog expansion and recovery

```powershell
uv run python scripts/expand_catalog.py
```

Expansion preserves existing source IDs, records and unchanged vectors, stages acquisition under `.data/expansions/<run>/`, and backs up active metadata under `.data/catalog-backups/`. Optional `--staged .data/expansions/<run>` promotes an already acquired run. The target remains 2,000 products; a complete active manifest skips further acquisition unless recovery is pending.

Before importing, promotion prepares replacement files and writes `.data/catalog/.promotion.json` with the stage identity and metadata hashes. It holds the maintenance lock through import and publication. Database commit and multiple file replacements are not atomic. If interrupted, keep the stage, images, journal and backups, and rerun the same command: recovery verifies hashes, repeats the idempotent import and finishes publication before removing the journal. Changed staged metadata is rejected. After recovery/import, run the text/image embedding commands above to complete any missing compatible vectors.

The sample is sequential, not representative. Source metadata/images/reports remain private under ignored `.data/`; customer review bodies are not part of the catalog pipeline. Existing Shopify utilities are retained compatibility interfaces and must not be run as normal setup.

## Troubleshooting

- Database errors: inspect `docker compose ps`, confirm `.env` credentials and apply pending migrations.
- Missing vector results after a model update: resume the corresponding embedding command. Runtime excludes incompatible revisions and dimensions.
- HTTP 409: wait for the session's in-flight turn or reload after a concurrent catalog/session change.
- Interpretation rejected: the model must preserve compound nouns and account for explicit modifiers with source quotes. Rephrase an ambiguous request; there is no automatic broadening retry.
- Fewer results: conservative phrase/title boundaries can exclude ambiguous listings. Missing metadata remains unknown; material similarity does not make a material an exact filter.

Earlier pilot counts, scope resets and operational measurements are preserved in [historical setup notes](../observations/archive/2026-10-24-before-audit-fixes/SETUP.md). They are evidence of past runs, not current instructions. The authoritative design is [architecture](architecture.md); request behavior is in [the agent guide](Agent.md).
