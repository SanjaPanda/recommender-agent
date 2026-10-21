# Database notes

## Schema and baseline inspection

The connected database inspected on 2026-10-21 is `recommender_agent` (local Compose PostgreSQL 17 with pgvector). It is the project-local named volume `recommender_pgdata`; no shared schema or unrelated database was inspected or changed.

| Table | Rows before cleanup | Purpose |
|---|---:|---|
| `products` | 8,671 | Catalog facts, images, captions, and optional attributes |
| `product_embeddings` | 17,863 | Text, CLIP image, and caption vectors; product FK cascades on deletion |
| `agent_sessions` | 2 | Durable LangGraph turn snapshots and cart state |
| `product_feedback` | 0 | Reserved feedback table; no records |
| `recommendation_sessions` | 0 | Older empty session table |
| `schema_migrations` | 5 | Applied migration versions |

There are no duplicate product IDs or orphan embeddings. All 8,671 product rows have image metadata. The 2,173 apparel rows all have text and CLIP embeddings. Caption vectors cover 124 apparel products. One apparel row has structured attributes.

| Catalog group | Products | Text vectors | CLIP vectors | Caption vectors |
|---|---:|---:|---:|---:|
| Apparel & Accessories | 2,173 | 2,173 | 2,173 | 124 |
| Other categories | 6,498 | 6,498 | 6,498 | 397 |

The 6,498 other-category records include Health & Beauty, Animals & Pet Supplies, Baby & Toddler, Sporting Goods, Home & Garden, and smaller categories. There are 398 non-empty captions among these products, 397 with vectors. Four apparel products have non-empty captions without vectors. No duplicate-title candidates are treated as duplicate products because the source has no stable identifier beyond the content-derived ID.

The local database size at inspection was 110,196,403 bytes. After cleanup and re-ingestion it was 110,270,131 bytes. The post-cleanup physical compaction details are recorded below.

## Cleanup executed

The user approved backup-first cleanup of all product data outside Apparel & Accessories. Exact pre-delete scope:

- Keep the 2,173 products whose `raw_source.ground_truth_category` is exactly `Apparel & Accessories` or a descendant path.
- Delete the remaining 6,498 product rows. `ON DELETE CASCADE` removes 13,393 embeddings: 6,498 text, 6,498 CLIP image, and 397 caption vectors.
- Product deletion also removes non-apparel product JSON/image metadata, the 398 non-empty captions, and two non-apparel structured-attribute records.
- Retain the one apparel attribute document, all tables and columns, indexes, extension, all five applied migrations, the database volume, and both `agent_sessions` rows. No feedback records or recommendation session records exist.
- One existing session contains 14 product-ID tokens, including nine IDs still in the database; none are apparel. The app now filters stale/non-apparel cards, selected IDs, and cart entries while preserving the stored row and conversation history.

The user approved cleanup and removal of the obsolete garden/mixed-category artifacts. Before SQL ran, a 61 MiB custom-format backup was written to `.codex-work/backups/recommender_agent_pre_apparel.dump`, copied back into the ignored workspace, and validated with `pg_restore --list`. The backup includes all six tables, indexes, constraints, and the pgvector extension.

Migration [`006_apparel_catalog_scope.sql`](../migrations/006_apparel_catalog_scope.sql) ran and is recorded in `schema_migrations`. It deleted 6,498 non-apparel products; the product foreign key cascaded deletion of 13,393 embeddings. The database volume and schema were retained. Immediately after catalog cleanup and a repeated ingestion (before migration 007), verification reported:

| Table | Rows after cleanup |
|---|---:|
| `products` | 2,173 |
| `product_embeddings` | 4,470 |
| `agent_sessions` | 3 |
| `product_feedback` | 0 |
| `recommendation_sessions` | 0 |
| `schema_migrations` | 6 |

The only product root is Apparel & Accessories. Embedding counts are 2,173 text, 2,173 CLIP image, and 124 image-description vectors. All 2,173 IDs match the filtered train/test files, all IDs are unique, every product has text and CLIP vectors, there are no orphan embeddings, and all 2,173 local source images exist. One apparel attribute record remains in the legacy JSONB `attributes` column. The experimental extraction/import code was removed, but the column and migrations 004/005 remain to preserve that record; dropping them would require a separate reviewed migration. Re-running `scripts/db/ingest_sample.py` twice upserted 2,173 rows each time without changing counts.

The 2 pre-existing session rows remain. Runtime session loading filters stale non-apparel IDs from cards, selected examples, and carts while preserving each row and its conversation history. The live UI verification created 1 additional session containing only apparel references; that session remains open in the shared browser.

## Source cache cleanup

The ignored `.data` cache was filtered to the two active splits. Train went from 5,908 mixed rows to 1,705 apparel rows, retaining 537,759,395 bytes of referenced images and removing 769,794,454 bytes of non-apparel images. Test went from 1,575 rows to 468 apparel rows, retaining 159,383,367 bytes and removing 187,089,485 bytes. The source IDs still match the database exactly and all retained image paths resolve.

The obsolete `.data/shopify-catalogue-sample` and three `shopify-outdoor-garden*` directories, cross-category attribute pilot outputs, and generated category/model logs were removed at the user's request. This cleanup freed 1,478,531,159 bytes total, including those image removals and directories. Apparel caption and router evaluation artifacts remain.

The ignored backup is the rollback path if restoration is needed.

## Database objects retained

Existing indexes are product primary key, product brand/category, embedding compound primary key, agent session primary key, feedback primary key/session-time, and recommendation session primary key. At 2,173 products, exact pgvector ranking is retained without approximate indexes. Empty feedback/session tables and schema history remain because dropping schema objects was not part of the approved data cleanup.

## Post-cleanup physical compaction

On 2026-10-21, after the approved catalog cleanup, ran `VACUUM (FULL, ANALYZE)` on `products` and `product_embeddings`. Before compaction, PostgreSQL reported a 105 MB database, with table totals of 31 MB and 66 MB. After compaction, it reported a 30 MB database, with table totals of 5,968 kB and 16 MB. Counts remained 2,173 products and 4,470 embeddings; `n_dead_tup` was zero for both tables, there were zero orphan embeddings, and the only product root remained Apparel & Accessories. `VACUUM FULL` briefly locks the target table while rewriting it. These sizes are PostgreSQL-reported, rounded display values; no application performance improvement was measured.

## Retiring image descriptions

The active recommender no longer generates, stores, or retrieves image descriptions. PostgreSQL still has the old `image_description` and provenance columns, nullable structured image columns added by migration 007, and 124 caption vectors. The last live check before Docker became unavailable found every structured visual column NULL and 124 non-empty apparel descriptions with matching caption vectors. The description importer has not run since that check.

Migration `009_remove_image_descriptions.sql` is staged. It deletes rows with modality `image_description`, narrows the embedding modality constraint to `text` and `image`, and drops the retired image-description columns. It preserves product rows and original images, text vectors, CLIP image vectors, sessions, and product attributes. Migration 008 was never applied and has been removed; dropping `image_components` is folded into migration 009. The migration is not yet applied: Docker PostgreSQL was offline at the time of this change. Before applying it, create and validate a fresh custom-format backup, inspect current table counts, caption-vector count, and non-null retired columns, then apply the migration and verify referential integrity and that product/text/CLIP counts are unchanged. Restore the backup to roll back if post-migration validation fails. Final after-counts and backup path must be recorded after execution.
