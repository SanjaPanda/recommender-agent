# Setup

Requires Python 3.12, uv, Docker Desktop and local Ollama. Run commands from the repository root unless stated otherwise. Copy `.env.example` to `.env`, set matching database credentials and keep it private.

## Dependencies and database

```powershell
uv sync --extra image
docker compose up -d db
uv run python scripts/db/apply_migrations.py
uv run python scripts/db/inspect_database.py
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

The migration runner initializes a fresh database from `migrations/schema.sql` and safely upgrades existing installations. See [database contracts](database.md#initialize-and-inspect). PostgreSQL 5432 and the UI 3000 are bound to loopback. The database volume persists during normal shutdown.

CLIP uses cached `openai/clip-vit-base-patch32` weights. Populate the local cache once if needed:

```powershell
uv run python -c "from transformers import CLIPModel, CLIPProcessor; name='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(name); CLIPProcessor.from_pretrained(name)"
```

Runtime does not download CLIP weights. Missing query CLIP allows available Nomic/BM25 modes; visual examples require compatible stored image vectors. `EMBEDDING_MODEL` supports explicit Ollama tags, with `:latest` implied when omitted. Changed encoder weights require rebuilding the corresponding vectors. Keep the installed tag stable while encoding.

## Catalog preparation

A fresh checkout needs private metadata, validated images and embeddings before populated searches work. Existing compatible assets need no reacquisition or rebuilding. The catalog is a bounded sequential sample of Amazon Reviews 2023 Apparel & Accessories, with a maximum of 2,000 products.

Prepare a separate source plan such as `.data/catalog-plan.json`. This illustrative template requires actual source identity, an immutable 40-character revision and recorded use evidence; do not replace existing source files with placeholders:

```json
{
  "source_id": "amazon-apparel",
  "source_name": "McAuley-Lab/Amazon-Reviews-2023",
  "revision": "REPLACE_WITH_PINNED_COMMIT",
  "source_url": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/REPLACE_WITH_PINNED_COMMIT/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl",
  "rights_status": "personal_use",
  "rights_evidence": "REPLACE_WITH_ACTUAL_SOURCE_USE_DECISION",
  "records": "records.jsonl",
  "images": {},
  "max_products": 2000,
  "max_lines": 100000,
  "max_bytes": 250000000,
  "min_description_words": 50
}
```

A recorded personal-use decision is not an upstream license grant. Use the decision and evidence appropriate to your source. Keep datasets, images and reports under ignored `.data/`.

```powershell
uv run python -m rag.download_catalog .data/catalog-plan.json
uv run python -m rag.catalog .data/catalog/manifest.json
uv run python -m rag.embed_text
uv run python -m rag.embed_images
uv run python scripts/db/inspect_database.py
```

Acquisition refuses to overwrite an occupied output directory, validates HTTPS hosts/redirects/public DNS and enforces line, byte, image-attempt and time limits. The acquisition report is stored as `pilot-report.json`; `rejections.jsonl` records individual rejected source lines. Original upstream line numbers and selected image provenance are stored in the generated manifest. These names are persisted asset contracts, not separate workflows.

The importer consumes local files only. It accepts supported apparel branches, requires at least 50 description words and nonempty string-valued details, and preserves feature bullets. Decoded JPEG/PNG/WebP images need at least 224 pixels per side, at most 20 million pixels and at most 10 MB. Image URLs must belong to the source record. Safe paths must stay under `.data/catalog/`. The optional `evidence-v2` manifest policy requires stricter description/detail evidence; unreviewed attribute extraction does not enable hard filters.

Import/embedding commands write products/vectors. Unchanged ready vectors at the same revision are skipped; changed source/image hashes invalidate vectors in the import transaction. Missing source records are not deleted. Embedding `--limit` and `--batch-size` bound work; `--force` rebuilds. BM25 needs no separate persistent index or embedding job. Each module supports `--help`.

## Start the application

Start the host API:

```powershell
uv run recommender-agent-api
```

In another terminal:

```powershell
docker compose up -d --build web
docker compose ps
```

Open <http://localhost:3000>. Read-only API liveness is <http://127.0.0.1:8000/api/health>. Try a text search, a color follow-up, selecting up to three visual examples, explicit cart add/remove actions and session resume. Product details, similarity details and interpretation/trace panels explain listing evidence and ranking signals.

New textual requests require local inference; identical effective requests reuse saved interpretation. Concurrent turns on one session return HTTP 409. Condition and budget filtering are unsupported; unknown source facts remain unknown. Material/visual preferences guide ranking without guaranteeing facts. Cosine percentages are not probabilities.

Optional terminal and retrieval interfaces:

```powershell
uv run recommender-agent-chat
uv run recommender-search "blue cotton dress"
```

## Catalog maintenance and recovery

```powershell
uv run python scripts/expand_catalog.py
```

This command maintains the catalog toward its bounded target using the manifest's existing source identity and policy. It preserves source IDs, source lines and unchanged vectors, stages acquisition under `.data/expansions/<run>/` and backs up metadata under `.data/catalog-backups/`. A complete manifest skips further acquisition unless recovery is pending. Optional `--staged .data/expansions/<run>` publishes an already acquired stage.

Before import, publication prepares replacement files and writes `.data/catalog/.promotion.json` with stage identity and metadata hashes. A maintenance lock spans import/publication. Database commit and file replacement are not atomic. On interruption, preserve the stage, images, journal and backups, then rerun the same command. Recovery verifies hashes, repeats the idempotent import and completes publication before removing the journal. Changed staged metadata is rejected. Resume text/image embedding commands afterward to complete missing compatible vectors.

## Validation

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

For opt-in PostgreSQL checks, see [database verification](database.md#verification). Some tests create and clean authored fixtures; schema tests use a disposable database. They are not read-only checks and do not download data or rebuild application vectors.

From `src/ui`:

```powershell
npm.cmd ci
npm.cmd exec tsc -- --noEmit --incremental false
npm.cmd run build
```

Docker builds install their own locked frontend dependencies.

Available diagnostics:

- `scripts/check_agent_live.py --infrastructure-only`: API/session/cart checks using an isolated test session without model inference. Omitting the option includes the local router.
- `scripts/check_router_edits.py --list`: list authored edit cases; `--case NAME` evaluates one against the local model.
- `scripts/check_search_intent.py --case compound`: evaluate authored interpretation evidence without product/vector/session writes. Omit `--case` to evaluate all cases in `data/search_intent_cases.jsonl`.
- `scripts/profile_search_latency.py`: optional instrumented search diagnostics, requiring local inference and a populated catalog; creates isolated sessions and writes `.codex-work/search-latency-profile.json`.

Authored checks verify specific interpretation/contracts, not general relevance accuracy. No diagnostic runs automatically during setup or unit discovery.

## Troubleshooting

- Database errors: inspect `docker compose ps`, confirm `.env` credentials and apply pending schema changes.
- Missing vector results after an encoder update: resume the corresponding embedding command. Runtime excludes incompatible revisions/dimensions.
- HTTP 409: wait for the current session turn or reload after a concurrent catalog/session change.
- Interpretation rejected: preserve compound product names and explicit modifiers; rephrase ambiguity. There is no automatic broadening retry.
- Fewer results: title/phrase evidence can exclude ambiguous listings. Missing metadata stays unknown.
- Condition requests: search without a new/used/pre-owned requirement. Unsupported saved constraints are sanitized on resume while preserving cart/history.
