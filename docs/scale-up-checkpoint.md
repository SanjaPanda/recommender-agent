# Catalogue scale-up checkpoint

## Four-category expansion (2026-10-17, import verified)

Both complete splits were scanned: 38,631 train rows and 9,658 test rows. The fetch saved 5,908 matching train records and 1,575 matching test records, all successfully imported.

| Category | Verified database products |
| --- | ---: |
| Apparel & Accessories | 2,173 |
| Health & Beauty | 2,162 |
| Baby & Toddler | 1,226 |
| Animals & Pet Supplies | 1,922 |
| Total requested-category products | 7,483 |

There were 33 products already present, so the import added 7,450 new products for a database total of **8,671**. All 1,221 original IDs and all 7,483 imported IDs were verified present. All 7,483 saved image files exist and match their recorded byte lengths (1,654,026,701 bytes total). Verification details are in the local ignored `.data/selected-category-import-report.json`.

The fetcher now accepts `--split train|test`. Both splits use pinned revision `002ca15536bb015ebc47b99cc2d716d7040bb09e` and a 50,000 matching-row limit, larger than either entire split, so each scan runs to exhaustion. Outputs are `.data/shopify-selected-train` and `.data/shopify-selected-test`. The existing importer upserts deterministic IDs and retains existing products. The image embedder's default roots include both new directories and the 1,000-record Outdoor & Garden directory; the API already discovers local sample directories.

```powershell
foreach ($split in @('train', 'test')) {
  uv run python scripts/data/fetch_sample.py --split $split --rows 50000 `
    --category-prefix "Apparel & Accessories" `
    --category-prefix "Health & Beauty" `
    --category-prefix "Baby & Toddler" `
    --category-prefix "Animals & Pet Supplies" `
    --output-dir ".data/shopify-selected-$split"
  if ($LASTEXITCODE -ne 0) { throw "Fetch failed for $split" }
  uv run python scripts/db/ingest_sample.py --input ".data/shopify-selected-$split/products.jsonl"
  if ($LASTEXITCODE -ne 0) { throw "Import failed for $split" }
}
```

Search embeddings for the 7,450 new records have not been generated. The database still has 1,221 text and 1,221 image vectors; new records are stored but not yet available through embedding-based retrieval. Run `uv run recommender-embed-products` and `uv run recommender-embed-images` to index them. The old checkpoint below describes the pre-expansion database.

## Outdoor & Garden checkpoint (2026-10-17)

The category sample includes `Home & Garden > Lawn & Garden` and `Sporting Goods > Outdoor Recreation`, using the pinned Shopify train split revision already documented in [architecture notes](architecture.md).

| Measure | Result |
| --- | ---: |
| Sample records | 1,000 |
| Source rows scanned | 9,240 |
| Image files | 1,000 |
| Image bytes | 190,816,953 (~182 MiB) |
| Missing descriptions | 67 |
| Missing brands | 12 |
| Duplicate-title candidates | 4 |
| Database products after upsert/deduplication | 1,221 |
| Text embeddings (`nomic-embed-text`, 768d) | 1,221 |
| Image embeddings (CLIP ViT-B/32, 512d) | 1,221 |
| Image files missing or unreadable during embedding | 0 |

Embedding the 784 remaining text products in the large batch took 509.8 seconds. The 1,158 missing image vectors took 432.4 seconds. Both measurements include this laptop's current local conditions and are only useful as rough planning rates. At those rates, adding roughly 8,800 products to reach 10,000 would take about 2.5 hours of sequential embedding, plus sample acquisition and database work. The image files would require roughly 1.7 GB more disk if the next sample has a similar average size.

## Reproduce the current sample

```powershell
uv run python scripts/data/fetch_sample.py --rows 1000 `
  --category-prefix "Home & Garden > Lawn & Garden" `
  --category-prefix "Sporting Goods > Outdoor Recreation" `
  --output-dir .data/shopify-outdoor-garden-1000
uv run python scripts/db/ingest_sample.py --input .data/shopify-outdoor-garden-1000/products.jsonl
uv run recommender-embed-products --batch-size 8
uv run recommender-embed-images `
  --image-root .data/shopify-catalogue-sample `
  --image-root .data/shopify-outdoor-garden `
  --image-root .data/shopify-outdoor-garden-1000 `
  --batch-size 1
```

## Path to 10,000

The 1,000 matching records required scanning 9,240 source rows. That suggests the selected narrow prefixes may not contain 10,000 products in the pinned train split. Reaching the target likely requires broadening to adjacent catalogue categories or sampling the whole catalogue. Keep category labels and filters so later evaluations can still compare cohesive slices. Exact search remains the baseline; evaluate IVF and HNSW after the larger sample is loaded and measure recall against exact results.
