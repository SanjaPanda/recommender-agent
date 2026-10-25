# Architecture

## Scope

One local apparel recommender. The old catalog was retired on 2026-10-23; the private Amazon catalog expanded from 50 to 2,000 products on 2026-10-24. This bounded sequential sample has no independently judged relevance benchmark. IDs combine the source namespace and upstream product ID, and stay consistent across PostgreSQL, retrieval vectors, LangGraph state, API responses and UI cards. Source-record and image hashes track content changes separately.

## Request flow

```text
Pinned source manifest + private Amazon metadata JSONL + validated local images
  -> apparel-root validation and ID de-duplication
  -> idempotent PostgreSQL upsert
  -> cached Nomic text vectors + CLIP image vectors
  -> FastAPI request validation
  -> deterministic plain-query pass-through or one LangGraph router node for edits
  -> typed local LLM search-intent extraction for textual requests
  -> SQL apparel/brand/condition/product-phrase/title-color filters + hybrid RRF retrieval
  -> response node and validated product list
  -> A2UI gallery + per-turn safe node trace; selection/refinement starts the next turn
```

All retrieval SQL applies the Apparel & Accessories root constraint independently of LLM output. More specific apparel category paths, brand, new/used status, title product type, and title color use parameterized SQL. Unknown attributes remain unknown. Preference statements also enter semantic retrieval, and responses say clearly that unverified preferences are not guaranteed.

## Ranking and models

Textual application searches now use local LLM extraction of `SearchIntent`: a request-supported core product phrase (regular inflections accepted), proposed equivalent search terms and explicitly quoted attributes. The product phrase can be any supported phrase rather than an entry in a subtype registry. Code validates request evidence and applies escaped, parameterized main-title phrase matching before limits in all retrieval modalities. Generic plural and joined/hyphenated spelling variants are accepted; description-only and compatibility-clause mentions cannot establish the product. Proposed terms expand BM25 only; original-query vectors and RRF weights are preserved. Extracted attributes are not new verified hard filters. The interpretation is persisted with its exact source query and reused for unchanged query/preference text. Invalid extraction raises a model error rather than silently guessing. Legacy direct retrieval/image-only paths retain their prior conservative family rules.

This literal phrase approach may miss equivalent titles using different words; a title phrase is not proof of visual type, material or color. Unrecognized or ambiguous positive product intent can remain empty and use unconstrained semantic/lexical retrieval within the apparel scope. Local inference adds latency for new textual searches. The proposed subtype hierarchy was withdrawn in favor of this open-vocabulary interpretation.

- **BM25:** exact lexical ranking over source title, description and every feature bullet in the eligible SQL-filtered corpus. Okapi BM25 uses k1=1.2 and b=0.75, Unicode word tokens, no learned synonyms and no extra search service. Immutable document token counts are cached by text (4,096 entries), and document frequencies/length statistics are cached for up to eight exact eligible document sequences. SQL still fetches current facts and applies every filter before scoring. Text edits, additions/deletions, ordering and exclusion changes select new cache keys, preserving filtered-corpus scores and tie ordering. These are bounded process-local caches, not persistent indexes.
- **Nomic `nomic-embed-text`:** product title, description, all feature bullets and query embeddings. It contributes an independent semantic ranked list.
- **CLIP `openai/clip-vit-base-patch32`:** offline image vectors are queried with a cached local CLIP text encoder; selected product cards use the same image-vector space. BM25, product-text and query-to-image ranked lists are fused with weighted reciprocal-rank fusion, k=60 and equal base weights of 0.50 each. Selected examples add a 0.20 base weight when a text query exists, or 1.0 for image-only requests. Weights are renormalized over nonempty lists. Each contribution is weight/(60+rank); raw BM25 and cosine scores are never added together. Existing product-type title priority and relevance gating remain in place.
- **Ollama language model:** one LangGraph turn router emits a Pydantic v2 `TurnEdits` contract, which application code validates and applies atomically. Optional summaries use the typed `Summary` response and are disabled by default. Category filters, cart actions, sorting, and selection checks remain deterministic.
- **Relevance gate:** Retrieval ranks a wider candidate pool, then keeps products whose text contains requested core terms and explicit colors, or whose CLIP score supports the visible color. Initial Nomic/CLIP score cutoffs are 0.61/0.28. Exact text matches can pass below a score cutoff. If none pass, the agent apologizes and presents closest alternatives under the same apparel subcategory, brand, and condition constraints; legacy title color/product constraints are relaxed only for those alternatives; the extracted product phrase remains required.

CLIP loads lazily once per API process from the local model cache. If it is missing, the app logs a warning and returns Nomic product-text results. Selected examples require their stored CLIP image vectors. No vector index is created at this catalog size: pgvector performs exact ranking; the active catalog has 2,000 products; the existing offline encoders build independent text/image vectors.

## Persistence and services

Category counts use an eight-entry process-local cache keyed by database identity and the committed `catalog_revision`. Migration `012_catalog_revision.sql` increments the revision transactionally on every product INSERT/UPDATE/DELETE/TRUNCATE statement, including writes outside the importer. A repeatable-read, read-only transaction reads the revision and any cache-miss counts from one snapshot. Callers receive independent dictionaries. There is no TTL or manual invalidation requirement for ordinary committed product changes; restoring a database requires restarting API workers to discard process-local state. Category grouping and apparel-root enforcement still use authoritative source taxonomy.

PostgreSQL/pgvector stores product facts, embeddings, and complete LangGraph turn snapshots. Product embeddings have a foreign key with `ON DELETE CASCADE`. The application/API and Ollama run on the host; Compose runs PostgreSQL and the Next.js A2UI UI. Compose caps PostgreSQL at 1 GiB/1.5 CPUs and the UI at 512 MiB/1 CPU. The database volume is persistent.

Pydantic Settings loads the project `.env` and process environment at startup. `DATABASE_URL` is secret-wrapped. Ollama calls use typed request models, provider JSON Schema, Pydantic v2 validation, bounded request timeouts, and no implicit retry loop.

For local debugging, each completed LangGraph node adds a per-turn trace event with node name, elapsed milliseconds, route/status, result/category counts, and the number of displayed retrieval results with text, CLIP, and selected-example scores. The UI and terminal client show these events. Prompts, user messages, product text, and model reasoning are excluded; trace events are omitted from durable session state.

## Deliberate limits

- The catalog has no dependable price field, so price/budget filters are not supported.
- There is no arbitrary image upload path; CLIP queries use text or selected catalog products.
- Color hard filtering uses product titles. Image and semantic preferences are ranked, not verified.
- PostgreSQL sessions store conversation and cart snapshots. When old snapshots contain other-category product references, the API filters those product references while preserving the session row/history.
- Approximate vector indexes, a reranker, multiple agents, and hosted inference are not part of the current implementation.

## Scope transition: decisions versus implementation

The user-supplied portfolio notes dated 2026-10-22 are planning inputs, not evidence that their proposed system has been built. This cleanup preserves the repository layout and running application boundaries.

Retain PostgreSQL/pgvector in Docker, the Docker web UI, local inference, one LangGraph workflow, independent text/image vector spaces, deterministic taxonomy eligibility and exact retrieval. Offline processing stays in `src/rag/`; no standalone experiment code belongs in `src/agent/`.

The bounded target of 2,000 clothing, shoes, jewelry and accessory records is imported from the pinned Amazon revision, with one validated image and provenance per product. The user accepted title/description/features retrieval despite sparse structured attributes. The optional stricter curation policy is not active. The earlier MiniLM proposal was not adopted: Nomic remains the active text encoder. The private-use decision is recorded separately from the unspecified upstream license. Do not fabricate descriptions or mix vectors from different models.

Current gaps: the sequential 2,000-product sample is not representative or independently judged; arbitrary administrative vector writes still need a schema-level model registry; arbitrary image uploads and reproducible held-out relevance evaluation are not implemented. Migration 010 supplies dimensions and optional input/preprocessing hashes, but existing encoder CLIs do not fill those hashes. Existing relevance cutoffs require evaluation before quality claims. Earlier proposed API contracts and experiment acceptance gates are not evidence of implemented behavior.

Database foundation: migration 010 adds a source registry, normalized product provenance and database-level apparel taxonomy checks. See [the database contract](database.md#tables-and-provenance). The full Amazon corpus is outside the current bounded scope; no encoder migration is required for this expansion.

## Local pilot ingestion

`src/rag/catalog.py` handles a pinned local source manifest, bounded metadata scan, source-text normalization, decoded local image validation, provenance and transactional upserts. It augments raw JSON with the required normalized apparel taxonomy while retaining upstream fields. The CLI requires `.data/catalog/` so the API and offline image encoder agree on the asset root. Source changes invalidate stored product vectors. Source features now enter Nomic document text.

This importer consumes local assets; acquisition is a separate bounded step. The initial pilot and later 2,000-product expansion use the same source identity and normalization policy. Stratified sampling and broader quality review are unimplemented; no MiniLM migration is active. See the [pilot setup](SETUP.md#curated-local-pilot-importer) and expansion walkthrough.

`src/rag/download_catalog.py` prepares bounded local assets with HTTPS host/redirect checks, byte/request limits, decoded-image checks and explicit rejection reporting. It shares metadata normalization with the importer and preserves original upstream line numbers in the manifest. It does not connect to the database. `scripts/expand_catalog.py` orchestrates acquisition and transactional import while preserving existing identities. The user-authorized private catalog has expanded from 50 to 2,000 products; upstream licensing remains unspecified. The optional evidence-v2 policy remains outside the active expansion contract. See the [downloader walkthrough](SETUP.md#bounded-downloader-and-pilot-report).

## Curation review policy

`src/rag/quality.py` assesses local pilot evidence without changing products. Downloader and importer optionally enforce evidence-v2 via the source manifest. Legacy manifests keep their original curation contract. The stricter gate checks description words excluding recognized boilerplate, at least two product-specification signal families, and one useful structured details field. Source feature clues are reported with source indexes and matching text; they are not promoted to verified attributes or used as hard filters. The 50-product review found 48 with feature clues but only one with useful structured details, so the stricter optional policy was not adopted for the later user-approved feature/text retrieval expansion.

## Attribute candidate evidence

`src/rag/attributes.py` produces deterministic unreviewed candidates from structured details, features and descriptions. Each candidate carries exact source references/offsets and context. Import creates the candidate document; the enrichment CLI can update already-imported rows only after checking their source-record hash. Evidence is stored under products.attributes.extraction while the original details remain under attributes.attributes. This fits the existing schema-version-1 document contract; no new migration is needed. Runtime hard filters do not consume candidates. Material/component mentions, explicit source claims and verified facts remain distinct; no claims are verified by this extractor.

The current candidate extractor is rules-v2. Explicit local component/variant/measurement scopes are stored with source evidence; unknown and size-table warnings are preserved. Compound cup/sleeve labels remain distinct. Conflict comparisons respect scope. Coverage gaps remain for coordinated/plural component text; no runtime hard filters consume extraction evidence.

The earlier 50-product pilot had 50 Nomic/50 CLIP vectors; the expanded catalog now has 2,000 of each. Ad-hoc retrieval checks completed, including selected-image seed exclusion and taxonomy restriction, but revealed product-type/material relevance gaps. They do not validate ranking quality or current thresholds as a benchmark.

## Explicit product-type eligibility

`src/rag/product_types.py` recognizes a small set of explicit title/query product families, including simple aliases and compounds. Explicit title_product filters take precedence. For an unambiguous positive query type, retrieval adds a parameterized broad title-word SQL condition before candidate limits, then checks the primary title type before returning results. For image-only requests, a type is inherited only when every selected seed has the same known title type. Mixed/unknown seeds remain unresolved. Browsing without a query/example is unchanged.

The relevance gate independently checks explicit type against the title so boilerplate descriptions cannot make a sweater qualify as a shirt. Existing SQL apparel/category/brand/condition constraints remain active. These legacy direct/image-only family rules do not apply to the open-vocabulary phrase path. The rule does not use unreviewed attribute candidates, alter embeddings/similarity thresholds, infer gender/age or guarantee material. Ambiguous titles may be excluded and recall can fall; unknown/mixed queries retain prior ranking behavior. Explicit types return fewer results or none rather than silently filling with other types. This is title evidence, not a curated type ontology.

## Result transparency

API serialization batches source listing details for displayed products and adds deterministic color/type listing checks through agent.product_evidence. Source description/features/details and snapshot ratings are exposed alongside existing similarity scores. The A2UI schema retains these fields; cards offer labelled similarity percentages, match/mismatch/unknown checks and expandable product details. Checks are explanations of explicit listing text, not image verification or additional filtering. Existing extracted candidates remain unused by these checks and by hard filters. No new extraction work is required for this UI flow.

## Catalog expansion and score transparency (2026-10-24)

`scripts/expand_catalog.py` extends the existing private catalog toward 2,000 records using the same pinned source and current normalization policy. It seeds a separate `.data/expansions/<UTC timestamp>/` directory with byte-identical original records and validated images, preserving original upstream line numbers. It then scans the pinned metadata stream for new eligible IDs while rejecting duplicate titles/images. Assets are content-addressed; imports preserve unchanged vectors and source IDs. Existing session/cart references keep working.

Operational scan limits may increase on an existing source registry row; revision, source decision, normalization and quality policy cannot change under that ID. Acquisition remains bounded to 100,000 metadata lines, 250 MB metadata, 500 MB new image bodies, 6,000 image attempts and 30 minutes per run at the 2,000 target. Partial successful acquisitions can be promoted, with the deficit recorded; rerunning extends from the active manifest. The scan is sequential, not stratified or representative. Original metadata reports are backed up under `.data/catalog-backups/`; raw data stays private and ignored by Git.

No migration or BM25 embedding/index build is required. New records need the existing Nomic and CLIP batch commands for complete multimodal coverage. Every ranked product exposes `bm25_score`, `rrf_score`, `rrf_k` and `rrf_contributions` (signal, rank, normalized weight, contribution); `score` remains the compatibility alias for RRF. The UI keeps cosine percentages and listing checks, and adds raw lexical/fusion scores in Similarity details. No percentage match probability is inferred from RRF, BM25 or missing metadata. Browsing has no query score. This is an implementation/integration change; held-out relevance quality is unmeasured.

BM25 candidates retain their source features for the relevance gate: feature-only keyword intent can pass lexical checks. Color checks continue to use the prior title/description/CLIP rules; a color mentioned only in a feature variant is not automatically verified.

Oversized upstream JSONL records are drained in bounded chunks and rejected, with their bytes/time charged to the scan budget. A large record no longer prevents acquisition of later eligible records; prepared importer records still obey the 1 MB line cap.

Catalog imports take a transaction-scoped PostgreSQL advisory maintenance lock; text/image embedding CLIs hold the corresponding session lock through their batches. This serializes maintenance writers and avoids source-order upserts deadlocking with ID-order embedding foreign-key locks. Retrieval does not take the lock. Optional EMBEDDING_NUM_GPU controls the local Ollama embedding hardware for both queries/documents, independently of the agent language-model setting. No schema migration is required.


## Audit hardening: state, provenance and phrase evidence

Color follow-ups remove superseded recognized color clauses from the effective query and saved preferences, while retaining the product need and explicit new title-color filter. A changed effective query invalidates interpretation reuse. Other preferences remain unverified.

Interpretation validation accounts for meaningful request words outside the product phrase using quoted attributes, known colors and conversational stop words. A shortened compound such as bow tie -> tie leaves an unexplained modifier and is rejected; negated products cannot become the positive product. This improves grounding without a subtype registry, but cannot prove that every model-provided attribute is semantically classified correctly. Main-title phrase eligibility now requires a segment boundary after the phrase, so dress does not establish Dress Shoes. This conservative boundary may exclude legitimate titles with unpunctuated suffix descriptors; it is text evidence, not visual/product truth. Card evidence states this limitation.

Vector searches require the immutable query encoder revision and dimensions. Nomic uses the exact installed tag and checks for alias changes across encoding; CLIP uses the loaded local snapshot revision. Image-only seeds must share a compatible revision and dimensions. Interrupted rebuilds can reduce vector coverage until resumed; incompatible vectors are never mixed into their ranked list. Long CLIP text is explicitly truncated to the encoder's context limit.

Session turns use a nonblocking connection-scoped advisory lock, returning HTTP 409 for a competing request. Inference runs outside database transactions. Short repeatable-read transactions load/save snapshots; compare-and-save prevents external state overwrites, and a changed catalog revision rejects stale inference. Current facts and source hashes refresh saved cards/cart references. Changed or unidentifiable historical recommendation snapshots are invalidated, while cart facts remain current and obsolete scores are cleared. Image serving reads authoritative current apparel metadata instead of keeping a stale ID-to-path cache.

Catalog expansion prepares replacement metadata and writes a hash-checked recovery journal before import. The maintenance lock spans import/publication. Rerunning recovers an interrupted promotion by repeating the idempotent import and publishing all prepared files. Database and filesystem operations are recoverable rather than atomically committed together; retain the stage/journal/backups until completion. See [setup recovery](SETUP.md#catalog-expansion-and-recovery) and [database contracts](database.md).
