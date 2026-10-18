# Phase 2 text retrieval evaluation

## Expanded smoke evaluation (2026-10-17)

| Setting | Result |
| --- | ---: |
| Model | `nomic-embed-text` (Ollama) |
| Embedding dimensions | 768 |
| Indexed products | 250 |
| Labeled queries | 44 (34 in-catalogue, 10 expected empty) |
| In-catalogue hit rate at 5 | 31/34 (0.912) |
| In-catalogue mean reciprocal rank | 0.912 |
| Unfiltered return rate for empty queries | 10/10 (1.00) |
| Mean query embedding time | 107 ms |
| Mean PostgreSQL vector-search time | 13 ms |
| Full product embedding run | 156 seconds |

The query vectors were created after a warm-up call to avoid including initial model loading in per-query latency. Query embedding time includes the local Ollama HTTP request; database time measures the exact cosine-distance search on the 250-row catalogue. Latency varies between runs, especially while other local processes are using the laptop.

## Query set

The labeled queries are in [`../data/phase2_queries.jsonl`](../data/phase2_queries.jsonl). The set includes 34 hand-labeled in-catalogue requests and 10 requests expected to have no matching product, including one budget request the catalogue cannot reliably enforce. Each in-catalogue query has one known relevant product ID. Three requests missed the labeled product in the top five: the baby teether, shoe stretcher, and child's bed frame requests.

The search command currently always returns nearest neighbors, including for out-of-catalogue queries. A diagnostic top-1 similarity threshold sweep shows the tradeoff on this small query set:

| Minimum top-1 similarity | Correct top-1 rate for in-catalogue queries | Acceptance rate for expected-empty queries |
| ---: | ---: | ---: |
| 0.55 | 31/34 (0.912) | 9/10 (0.900) |
| 0.60 | 31/34 (0.912) | 6/10 (0.600) |
| 0.65 | 30/34 (0.882) | 1/10 (0.100) |
| 0.70 | 23/34 (0.676) | 0/10 (0.000) |

The CLI uses 0.65 as an initial cutoff and allows `--min-similarity` to change it. The one expected-empty query above 0.65 is the budget request; the CLI now exits with an explicit warning for that unsupported constraint. None of the other nine out-of-catalogue queries cleared the cutoff in this set. This is encouraging for the sample, but the cutoff is still based on a small, hand-written query set.

This remains a small smoke evaluation, not a representative benchmark. Queries are hand-written around known sample products, each in-catalogue query has only one labeled relevant item, and the 250-row sample is narrow. The threshold is an initial operating point, not a production-calibrated policy. Next, broaden the data and labels, include multiple relevance judgments per query, and compare model or text-preparation choices on the same labels.

## Reproduce

With Docker PostgreSQL running and `nomic-embed-text` available in Ollama:

```powershell
uv run recommender-evaluate --k 5
```

Use `--queries PATH` to evaluate another JSONL file and `--k N` to change the retrieval cutoff.
