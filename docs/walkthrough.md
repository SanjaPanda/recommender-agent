# Execution walkthrough

Run commands from the repository root with `.env`, PostgreSQL, local Ollama and
the catalog prepared as described in [setup](SETUP.md). Application data remains
private under `.data/`. The API and Ollama run on the host; the web UI and database
run in Docker. There is no graph runtime, model retry loop or hosted inference.

## Run and test

Start the API and, in a second terminal, the web UI:

```powershell
uv run recommender-agent-api
docker compose up -d --build web
```

Open <http://localhost:3000>. Optional terminal interfaces use the same configured
catalog and encoders:

```powershell
uv run recommender-agent-chat
uv run recommender-search "blue cotton dress"
```

The chat CLI persists shopping state. The search CLI is a raw ranker diagnostic:
it skips conversational interpretation and the agent's response relevance gate.
Neither is a replacement for a browser end-to-end check.

Run the prescribed default tests:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Enable database integration tests only on a development installation:

```powershell
$env:RUN_DATABASE_TESTS='1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
Remove-Item Env:RUN_DATABASE_TESTS
```

These tests create/clean authored fixtures and use a disposable migration database.
They do not acquire data or rebuild application vectors. For frontend validation,
run `npm.cmd exec tsc -- --noEmit --incremental false` and `npm.cmd run build`
from `src/ui`. The implementation checkpoint records actual verification results;
commands here describe supported operations, not an assertion that every command
was executed during every change.

## Startup

1. `agent.config.Settings` reads the project environment and validates required
   database configuration, local Ollama host, hardware overrides and timeouts.
   SecretStr keeps the database URL out of ordinary settings representations.
2. `agent.api` creates FastAPI and loopback CORS rules. It does not compile an
   agent or load a language/image model during import. CLIP loads lazily from its
   local cache when a text-to-image query needs it.
3. The browser's `RecommendationApp` in `src/ui/app/page.tsx` posts the saved
   localStorage thread ID to `/api/session`. A new session returns categories;
   an existing session refreshes its saved catalog facts without rewriting JSON.
4. `src/ui/app/a2ui.ts` contains renderer imports, the ProductSchema, inferred
   Product type and response types. `catalog.tsx` renders the ProductGallery and
   connects visual selection/cart buttons through ProductActions.

## A textual shopping turn

For the illustrative input `blue cotton dress`:

1. **Validate the request.** `/api/discover` accepts initial text and optional
   explicit categories; `/api/recommend` accepts refinement and up to three
   unique selected examples. FastAPI/Pydantic reject malformed request fields.
   The API chooses or retains the thread ID.
2. **Load under a session lock.** `agent.sessions.run_recommendation_turn` obtains
   a nonblocking advisory lock. A short repeatable-read transaction reads catalog
   revision and session JSON. `sanitize_apparel_products` normalizes retired
   fields, refreshes facts and removes ineligible/stale cards. A busy thread fails
   immediately; it does not enter a retry loop.
3. **Route directly.** `agent.workflow.run_turn` calls `_route`. Explicit actions
   are deterministic. Text calls `agent.llm.complete` once with RouterDecision's
   JSON Schema, the previous plan and bounded visible-product/category context.
   The result contains the full updated Plan, removal flags and SearchIntent.
   Pydantic and `validate_search_intent` check scope, consistency, quoted evidence
   and compound preservation before changes are accepted.
4. **Resolve interpretation.** `_retrieve` uses `Plan.effective_query`, which
   combines product need and preference requirements. It reuses the router's
   interpretation for that exact effective query. Example/category turns can
   reuse saved interpretation; older sessions without a matching interpretation
   use one standalone extraction call. Blank browsing/image-only queries do not.
5. **Prepare vectors once.** `agent.tools.retrieval.prepare_query` creates
   request-local QueryVectors. Nomic's installed digest is checked around
   encoding. CLIP uses a compatible cached local snapshot; unavailable query
   CLIP degrades to text/keywords. QueryVectors is never persisted or reused
   across sessions. Selected examples are read and checked against stored image
   revisions/dimensions by retrieval.
6. **Retrieve and rank.** `retrieve_products` applies parameterized apparel,
   category, brand, title/phrase and cart-exclusion predicates before candidate
   limits. BM25 and compatible vector searches produce separate ranked lists;
   `_fuse_rankings` combines reciprocal ranks with unchanged weights and k=60.
   `is_relevant_product` determines strong matches.
7. **Handle alternatives.** If no strong match passes, `_retrieve` can issue one
   relaxed-title search using the same QueryVectors. Only title color/type
   filters relax. Apparel/category/brand, interpreted phrase and cart exclusions
   remain. If no explicit title filter existed, the initial pool is reused.
8. **Respond.** `_respond` chooses deterministic count/status prose, preserves
   uncertainty notices and limits cards to the requested maximum of five.
   Optional generated summaries remain disabled by default; when enabled, their
   15-second failure cannot discard otherwise valid cards.
9. **Save safely.** A second short transaction rejects a changed catalog revision
   or externally edited snapshot, refreshes/stamps eligible product identities,
   and compare-and-saves durable shopping fields. Inference runs outside these
   transactions. Closing/unlocking the session connection releases ownership.
10. **Serialize current evidence.** `api._response` derives query_context from
    Plan.query. `_serialize_products` uses the shared `load_products` batch read
    for current facts/listings, clears stale scores and resolves safe image URLs.
    `listing_checks` reports match/mismatch/unknown from listing evidence.
    A2UI operations update the gallery and the browser stores the thread ID.

The typed interpretation describes the request; it does not prove catalog facts.
Required matches can return fewer than five cards. Source descriptions, image
variants and similarity scores do not establish material/color suitability.

## Other supported paths

| Trigger | Execution |
| --- | --- |
| New session or `#buynew` | Clear search context/cards/selections, retain cart, return categories without model/retrieval calls. |
| Resume | Read saved state, refresh evidence and categories, create/replace the A2UI surface; do not save during read-only resume. |
| Category browse | Validate the explicit category and browse eligible rows; a blank effective query needs no encoders. |
| Visual examples | Validate displayed IDs, obtain compatible stored image vectors and normalized mean direction, then rank without adding to cart. |
| Cart add | Validate the entire ID batch, add only displayed eligible products, remove them from cards and selected examples. |
| Cart remove | Remove eligible cart IDs and retain existing cards; removed IDs become available on the next retrieval. |
| Clarification | Preserve shopping constraints and cards; skip retrieval. |
| Explicit new/used condition | Return the unsupported-condition explanation before model/encoder calls. |
| Invalid model output | Fail explicitly with ModelError; no guessing or broadening retry. |

Successful stages emit the existing `router`, `retrieve`, `respond` trace labels
with duration, status and counts. Failures log stage name/type/duration and
propagate. Prompts, request text and model reasoning are excluded from trace
events. A direct-response turn still includes a respond event for compatibility.

## Durable and transient state

Plan is the authoritative shopping context. Durable fields include the plan,
bounded message history, selected IDs, recommendations, cart, response status/text,
result kind, removal flags and saved interpretation/query identity. Requests,
candidate pools, category counts, trace events, route and derived query_context
are not saved. Existing snapshots remain readable without bulk database migration.

`agent.product_evidence.load_products` shares the apparel-scoped fact/listing read
between session normalization and API serialization. Reads at separate load/save/
response boundaries remain intentional freshness checks.

## Offline processing

`rag.download_catalog` acquires bounded pinned-source metadata/images;
`rag.catalog` normalizes and imports it transactionally. Optional `rag.attributes`
and `rag.quality` preserve unreviewed evidence/curation semantics. `rag.embed_text`
and `rag.embed_images` write revisioned vectors, using shared runtime encoder
functions where appropriate. `scripts/expand_catalog.py` retains its maintenance
lock, backups and hash-checked publication journal. No request imports a batch
embedding command to encode its query.

The notebooks remain diagnostic experiments, outside runtime execution. Their
helper imports the shared ranking/encoding implementation; updated import cells
are marked unexecuted until the experiment runner is deliberately rerun.

## Before and after

| Concern | Previous implementation | Simplified implementation |
| --- | --- | --- |
| Execution | Compile graph, configure/invoke, merge three nodes | Direct run_turn with one optional retrieval branch |
| Persistence | Transaction/sanitization code inside workflow.py | Dedicated sessions.py with the same concurrency safeguards |
| Fallback encoding | Re-enter encoders for relaxed search | Reuse request-local QueryVectors |
| Runtime dependencies | Retrieval imports offline embed_text command | agent/tools contains retrieval, BM25 and shared encoders |
| Stored state | Graph bookkeeping and duplicate query_context | Explicit durable-field allowlist; derive query_context |
| Product refresh | Overlapping SQL in session and API modules | Shared apparel-scoped batch reader, retained freshness boundaries |
| UI contracts | Product type and Zod schema maintained separately | One ProductSchema and inferred TypeScript type in A2UI adapter |
| Dependencies | LangGraph and its dependency chain | Existing FastAPI/Pydantic/PostgreSQL/local encoders |

The previous source and documents are preserved under
`archive/before-end-to-end-simplification-2026-10-25/` with a SHA-256 manifest;
Git checkpoint `52679a3` preserves the tracked baseline and historical migration
SQL. Do not restore a snapshot over unrelated user work. Migration history under
`archive/migrations/` remains an active upgrade dependency.

## Debugging and extension

Start with the safe trace: router failures implicate request/model contracts;
retrieve failures implicate interpretation, encoders, examples or catalog access;
respond failures implicate result formatting/optional summaries. HTTP 409 means
the thread or its inputs changed; reload or retry after the competing operation.
API health is liveness, not proof that models/catalog vectors are available.

Change routing policy in workflow.py, contracts in models.py, ranking/encoding in
agent/tools, persistence in sessions.py, and offline processing in rag/. Preserve
SQL eligibility before limits and add focused observable regression tests for
changed behavior. There is no need for a new framework or tool registry to add a
small deterministic operation.
