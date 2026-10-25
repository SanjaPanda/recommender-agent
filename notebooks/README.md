# Retrieval experiments

Two small, executed Python notebooks compare the existing apparel retrieval signals.
They read the catalog and stored vectors, call local embedding encoders, and save
private experiment caches and observations. They never update products, application
vectors, sessions, ranking weights or thresholds. No generative LLM is used.

| Notebook | Question |
| --- | --- |
| [01_text_queries.ipynb](01_text_queries.ipynb) | When do BM25, Nomic text, CLIP text-to-image and the existing RRF reference differ? |
| [02_image_examples.ipynb](02_image_examples.ipynb) | What changes with 1/2/3 images, conflicting examples, descriptions and short visual captions? |

## Reproduce

From the repository root, with the existing `.env`, populated PostgreSQL catalog,
compatible stored text/image vectors, local Ollama Nomic and cached CLIP model:

```powershell
uv sync --extra image
.venv/Scripts/python.exe scripts/run_retrieval_experiments.py
```

Run just one notebook by appending `01_text_queries.ipynb` or
`02_image_examples.ipynb`. The first run encodes ten queries, six selected source
descriptions, six analyst-written visual captions, a six-seed source-vector audit,
and titles/descriptions/full listings from a 68-product diagnostic pool. Cached
reruns reuse exact inputs and model revisions.
No full-catalog embedding rebuild or model download is performed.

The runner executes ordinary Python cells sequentially and writes actual stdout,
execution counts and errors into the notebooks. It needs no Jupyter server or new
application dependency. It is **not a Jupyter kernel validation**. Package downloads
were unavailable during development, so an interactive kernel was not tested here.
For optional VS Code/Jupyter interaction, install a kernel and select this `.venv`:

```powershell
uv pip install --python .venv/Scripts/python.exe -r notebooks/requirements.txt
```

Run cells in order from the project root or `notebooks/`. They contain no magics,
hidden notebook state or IPython-only display calls. The figures are Markdown links
to locally generated contact sheets; all numerical outputs are ordinary Python.

## Design and interpretation

Cases and seed photos are fixed in [cases.json](cases.json), after checking catalog
availability and seed appearance, before viewing rankings. The small helper
[retrieval_lab.py](retrieval_lab.py) shares read-only loading, normalized cosine math,
cache validation, evidence checks and galleries. Cells show the experiment loops.

- **Shared candidates:** component searches use the same complete paired-vector apparel
  universe. Missing vectors or mixed revisions stop the run explicitly.
- **Components:** no router, product-phrase/title filters, relevance thresholds or
  fallback. These reveal signal behavior and are not production end-to-end scores.
- **RRF reference:** reuse application fusion, k=60, three equal 0.50 base weights,
  top-50 component pools. No tuning or change to application defaults.
- **Image selections:** normalized mean vectors. Every example-count run excludes
  all three fixed group seeds and identical-photo siblings. Mixed examples have
  their own fixed exclusion set. More seeds are not assumed to improve retrieval.
- **Text proxy access:** Nomic cannot directly read selected pixels. Metadata and
  source-description baselines use text attached to those catalog products.
  Analyst captions were written from the six seed photos and assert visible
  content only. Caption queries still search the full-listing index.
- **Observed input confound:** three of six fixed stored text vectors do not reproduce
  current full-listing inputs with the same model digest. Cause and full-catalog
  extent remain unknown. Full-catalog comparisons describe the current index snapshot;
  they cannot establish isolated encoder quality. The notebooks expose this warning.
- **Fresh control:** titles, descriptions, freshly encoded full listings and stored
  vectors search the same 68-product pool. Two three-image groups also compare CLIP
  with fresh text on that pool. Selection uses 12 IDs/branch, fixed seeds and the first
  two title-family/literal-evidence matches per case. It is deliberately enriched,
  not a random benchmark. Do not compare its metrics to the 2,000-product runs.
- **Measurement:** `family@5` is a declared title-regex proxy; `listing@5` additionally
  checks literal claims. These are not independently judged Precision@5, Recall@k,
  visual accuracy or probabilities. Missing metadata is not proof of an exclusion.
  Negation success is left unscored. Raw BM25 and cosine scales are not compared.
- **Visual labels:** [visual_judgments.json](visual_judgments.json) records one post-run,
  unblinded review of all 16 unique photos in the white-sneaker grid. The rubric requires
  mostly white/off-white and a low-top shape. New/ambiguous results remain unknown, with
  score bounds. This illustrative case is not a held-out evaluation.
- **Review:** inspect the five-result image grids independently of listing proxies.
  Catalog photo variants, packaging shots and seller copy can disagree. A source
  description is not automatically an explanation of the selected photo.

The first run exposed two evaluation issues: an overbroad title rule rejected valid
T-shirt dresses, and the small pool contained no bow ties. These controls were corrected
and rerun with queries/seeds unchanged. That iteration is documented in cases.json and
the notebooks. No model or ranking parameters were tuned to improve the results.

## Outputs and evidence

The notebooks retain execution outputs. JSON results, figures, the findings memo
and a checkpoint live under `observations/experiments/retrieval/`; the repository
already treats `observations/` as local engineering evidence. Private embedding
caches are under `.data/experiments/retrieval/`. Keep both when reproducing this run.
Manifest hashes identify candidate texts/vectors, query cases and exact encoder
revisions. Installed package versions are recorded. No dataset/model assets or
credentials are copied into the notebooks.

Read [the run findings](../observations/experiments/retrieval/findings.md) before
drawing conclusions. These are exploratory examples for future article research,
not a general winner, statistical study, timing comparison or article draft.

Primary background sources: [OpenAI CLIP encoders and cosine example](https://github.com/openai/CLIP),
[Nomic query/document task prefixes](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#task-instruction-prefixes),
and [Ollama's Nomic model family](https://ollama.com/library/nomic-embed-text).
The result manifest identifies the actual local models by digest.
