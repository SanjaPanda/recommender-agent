# Phase 3 image retrieval prototype

## First implementation

- Image encoder: `openai/clip-vit-base-patch32`, run on CPU through Transformers and PyTorch.
- Image and CLIP text vectors share a 512-dimensional space. Product image vectors are stored as `modality='image'`; they do not overwrite 768-dimensional Nomic text vectors.
- `recommender-embed-images` reads saved local image paths, skips vectors for the same model revision, and reports missing or unreadable images.
- `recommender-search --mode hybrid` retrieves text and image candidates separately and combines ranks with weighted reciprocal-rank fusion. The conservative prototype image rank weight defaults to 0.1. The output retains each modality's cosine score and reports the fused score separately.
- `recommender-search-by-examples` accepts one to three embedded product IDs, averages their normalized CLIP image vectors, and returns nearest catalogue image vectors. It is the CLI inspection path for product-card selection.
- Image dependencies are optional so text-only setup remains smaller. Image inference runs on CPU.

## First local run (2026-10-17)

Installed the optional image dependencies and embedded an eight-product batch from the current catalogue.

| Measure | Result |
| --- | ---: |
| Products embedded | 8 |
| Dimensions | 512 |
| Missing image files | 0 |
| Unreadable images | 0 |
| Elapsed time | 83.9 s |
| Model revision | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |

The focused Lawn & Garden batch subsequently embedded 55 products with zero missing or unreadable images in 23.5 seconds. The category evaluation uses ten hand-labeled requests and the same 55 products for text-only and hybrid retrieval:

| Ranker | Hit rate@5 | MRR |
| --- | ---: | ---: |
| Text only | 0.8000 | 0.8038 |
| Hybrid, image weight 0.1 | 0.8000 | 0.8042 |
| Hybrid, image weight 0.35 | 0.8000 | 0.7064 |

The light image weight changed MRR by only 0.0004, which is not meaningful evidence of an improvement on ten queries. The higher weight reduced MRR. Keep text-only as the quality baseline; retain hybrid as an experimental mode while broadening relevance labels and adding queries where product appearance matters. The eight-item time above includes first-run model startup and is not a throughput estimate; the 55-item run is a better indicator for this cached local setup, but CPU conditions still affect it.

## Physical-attribute and multiple-example checks

Two additional labeled queries target color that is visible in the product image but absent from the title: “green lawn aerator” (the green scarifier/aerator) and “yellow lawn sweeper” (the yellow Karcher sweeper). On the 12-query set, text-only and hybrid at 0.1 image weight both achieved 0.8333 hit@5, with MRR 0.8402 and 0.8416 respectively. For the yellow sweeper specifically, text-only ranked it 16th; hybrid at image weight 0.75 ranked it 2nd. That stronger visual weighting reduced overall MRR to 0.6271 and hit@5 to 0.75, so treat it as an attribute-specific experiment rather than a global setting.

The two-example retrieval command was run using the green Greenworks tool and green lawn aerator as references. It returned other lawn power equipment among the highest image-similarity matches. This verifies the query-by-example path, but visual relevance still needs human judgment and more labeled examples.

## Reproduce

```powershell
uv sync --extra image
uv run recommender-embed-images --ground-truth-category-prefix "Home & Garden > Lawn & Garden" --batch-size 1
uv run recommender-evaluate-multimodal --k 5 --image-weight 0.1
uv run recommender-search-by-examples --reference-product-id PRODUCT_ID_1 --reference-product-id PRODUCT_ID_2 --limit 5
```

The first command downloads optional PyTorch/Transformers packages. The first image run downloads the CLIP model weights; Hugging Face's published PyTorch weights are about 605 MB. On Windows without symlink support the local cache can consume additional disk space.

## Use-case checks on the 1,221-product catalogue (2026-10-17)

These are exploratory behavior checks tied to the two user journeys: pick a result and get similar products; then refine by explicit physical attributes. They are manual spot checks, not a replacement for a larger labeled evaluation set.

### Select one product and recommend five similar products

Starting query: `battery powered outdoor trimmer`. The text search returned a DeWalt 20V folding string trimmer first, the Greenworks 80V blower/trimmer combo second, and a Husqvarna battery combi powerhead/edger third. Selecting the Greenworks item from those candidates as the seed and retrieving five nearest CLIP image vectors, while restricting to `Home & Garden > Lawn & Garden`, returned:

1. Toro 60V brushless stick edger
2. Husqvarna battery combi powerhead with edger
3. DeWalt 20V folding string trimmer
4. Giantz pole chainsaw/hedge trimmer
5. John Gardener scarificator/aerator

The first three are close outdoor power-tool alternatives; the last two are adjacent lawn equipment. Without the category scope, two of five neighbors came from unrelated outdoor categories (a hunting kit and a fishing reel). This suggests similar-product retrieval needs a category or product-type boundary in addition to image similarity. The manual one-seed query excluded the selected Greenworks product and returned five items. The CLI and LangGraph agent now accept one seed. A graphical selection UI is still open.

The LangGraph CLI was then run as a three-turn session: search `battery powered outdoor trimmer`, select result 2 (the Greenworks combo), and refine with `lightweight`. The graph retained the original query and selection, passed `battery powered outdoor trimmer lightweight` to the shared RAG function, and returned five candidates after both the selection and refinement turns. The Greenworks seed was excluded from its own recommendations. This confirms context forwarding and seed exclusion; it does not prove that `lightweight` is enforced as a hard requirement.

### Search and refine by physical attributes

| Request | Observed result | What it tells us |
| --- | --- | --- |
| `green lawn aerator` | The correct John Gardener scarificator/aerator was first (text similarity 0.4676). | Attribute/category retrieval can work, but the 0.65 default cutoff hides this correct result. |
| `battery powered outdoor trimmer` | DeWalt 20V trimmer first, Greenworks combo second, and Husqvarna battery tool third. | Power-source intent is useful when the catalogue title/description explicitly says battery/voltage. |
| `lightweight cordless leaf blower` | Greenworks blower/trimmer combo first; a lightweight cordless pressure washer followed. | The broad semantic ranking can drift across product types; `lightweight` and `cordless` are not currently enforced constraints. |
| `yellow lawn sweeper` (hybrid, image weight 0.75) | Karcher S4 Twin sweeper first; yellow electric dethatcher also appeared among the top results. | Color evidence can help, but high image weight mixes products with visually similar colors and neighboring tool types. |

Current retrieval embeds the whole query and ranks semantically. It does **not** extract color, size, material, power source, or style into explicit state or hard filters, and it does not yet preserve a user's refinement across turns. To support interactions like “show lawn tools” → select an item → “green, battery powered” robustly, the next implementation should keep the selected seed and explicit attribute refinements separately, enforce known product-type/attribute constraints where catalogue data supports them, and use vector similarity to rank within that candidate set. Missing attributes must stay unknown rather than being guessed.

Reproduce the text spot checks (set cutoff to zero to inspect all nearest candidates):

```powershell
uv run recommender-search "green lawn aerator" --limit 5 --min-similarity 0
uv run recommender-search "battery powered outdoor trimmer" --limit 5 --min-similarity 0
uv run recommender-search "lightweight cordless leaf blower" --limit 5 --min-similarity 0
uv run recommender-search "yellow lawn sweeper" --mode hybrid --image-weight 0.75 --limit 5 --min-similarity 0
```
