# Resume the next work session

The user expanded the scope on 2026-10-17 to include all Apparel & Accessories, Health & Beauty, Baby & Toddler, and Animals & Pet Supplies records across the train and test splits. Acquisition uses `.data/shopify-selected-train` and `.data/shopify-selected-test`. See the latest import status in [the scale-up checkpoint](scale-up-checkpoint.md). Preserve the original sample and Outdoor & Garden expansion (1,221 distinct products before this import). A previous 10,000-row Outdoor & Garden attempt was stopped after Hugging Face timed out; it did not replace existing data.

## Verified checkpoint (2026-10-17)

- PostgreSQL/pgvector is running with 1,221 distinct products.
- All 1,221 products have ready 768-dimensional `nomic-embed-text` text vectors and 512-dimensional CLIP image vectors.
- Every product has at least a title or description; the catalogue has no documented price field.
- Image search supports hybrid text/image ranking. The new shared RAG function accepts one to three selected products and optional refinement text.
- Physical-attribute and visual recommendation behavior should be exercised on this bounded sample before deciding whether to change ranking logic.
- A first manual use-case check is recorded in [the Phase 3 notes](phase-3-prototype.md): battery-powered tools rank well when those words occur in the source text; image-based similar-item retrieval benefits from a Lawn & Garden scope; `green lawn aerator` is ranked correctly but suppressed by the default 0.65 cutoff.
- The agent carries refinements across turns and passes them to RAG retrieval, but does not yet extract physical attributes into hard constraints. In-memory session state is lost when the process exits.
- A one-node LangGraph workflow retains accumulated query text, selected product IDs, and category scope per thread with an in-memory checkpoint. Run `uv run recommender-agent-chat --category-prefix "Home & Garden > Lawn & Garden"`. FastAPI and the Dockerized A2UI interface are implemented; start the API with `uv run recommender-agent-api` and the UI with `docker compose up -d --build web`.

## Suggested next actions

1. Turn the manual observations into labeled cases for the two core journeys: select one to three of five results and get five similar products; describe a product with physical attributes and refine those attributes over follow-up turns.
2. Evaluate single-item selection end to end, including seed exclusion and category scoping.
3. Add structured extraction and enforcement for physical attributes supported by product text (such as color, material, power source, or dimensions); unknown attributes must remain unknown.
4. Compare text-only and hybrid rankings on those labels; treat the small hand-labeled set as exploratory.
5. Add persistent graph checkpoint storage if session context must survive process restarts. The API and web UI are available through the commands in the [UI setup guide](copilotkit-ui.md).
6. Once useful results are established, benchmark IVF and HNSW against exact retrieval on the current dataset. At 1,221 items, exact search may remain preferable; measure before adopting an index.

Start with [the code walkthrough](code-walkthrough.md) for module roles and core functions, then check [the phase status](phases.md), [Phase 2 evaluation](phase-2-evaluation.md), [Phase 3 prototype notes](phase-3-prototype.md), and [README setup instructions](../README.md).

## Local commands

From the project directory, check Docker and start the database if needed:

```powershell
docker info
docker compose up -d db
docker compose ps
```

Then search and evaluate with the current local models:

```powershell
uv run recommender-search "green mower" --limit 5
uv run recommender-search "green mower" --mode hybrid --limit 5
uv run recommender-evaluate --k 5
uv run recommender-evaluate-multimodal --k 5
```
