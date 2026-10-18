# CopilotKit and A2UI product recommender

The local web prototype supports the two recommendation behaviors currently being evaluated:

1. Describe a product need and get five starting candidates with images.
2. Select up to three candidates, add a natural-language detail, and finalize to get five recommendations informed by both image examples and the running text context.

The front end is a small Next.js app in [`src/ui/`](../src/ui/). It uses CopilotKit's A2UI React renderer and a fixed `ProductGallery` catalog. The backend emits A2UI v0.9 surface messages (`createSurface`, `updateComponents`, and `updateDataModel`) with a stable catalog and a thread-specific surface ID. It creates the surface on discovery, then updates that same surface for follow-up recommendations. The renderer owns the product-card UI; interaction state (selected examples and the text box) stays in the React app.

The FastAPI endpoints in [`api.py`](../src/agent/api.py) call the same checkpointed LangGraph recommendation flow used by the CLI. `/api/discover` starts the session with the initial user request. `/api/recommend` sends the selected IDs and refinement text into that thread. The agent accumulates text turns and replaces the active image examples when a new selection is submitted. Image URLs resolve local sample files only and reject absolute/traversal paths.

## Run locally with Docker UI

Start Docker Desktop and PostgreSQL/pgvector as described earlier in the README. Keep Ollama on the host. Start the API in one PowerShell terminal from the repository:

```powershell
uv run recommender-agent-api
```

Then build and start the Dockerized UI from another terminal:

```powershell
docker compose up -d --build web
```

This builds the UI image and starts the Compose service.

Open `http://localhost:3000`. Compose publishes the UI only on localhost. The Next.js rewrite forwards `/api/*` requests from the web container to FastAPI at `host.docker.internal:8000`; FastAPI binds to the host interfaces so Docker Desktop can reach it. The first request still needs the text embedding model reachable through Ollama. The browser will show an explicit error if the database or embedding service is unavailable.

To stop the UI container, run `docker compose stop web`. To develop locally, run `cd src/ui; npm.cmd install; npm.cmd run dev`; local development keeps the default API target at `127.0.0.1:8000`.

## Current limits

- The first text request seeds the initial five cards; there is no random catalogue browsing endpoint.
- Selecting up to three cards replaces the prior active example set. The retrieval layer averages their CLIP vectors and blends that ranking with the accumulated text ranking.
- The service's `MemorySaver` is process-local. Restarting FastAPI clears session context; use a persistent LangGraph checkpointer before deployment or multi-worker use.
- Structured attribute extraction and hard constraints (for example, interpreting “blue” as a color filter) remain future work. Current text details influence semantic retrieval.
- The service serves only local saved images; source-hosted remote images are not fetched by the browser.

The component catalog and operations are defined in [`catalog.tsx`](../src/ui/app/catalog.tsx), [`a2ui.py`](../src/agent/a2ui.py), and the UI orchestration in [`page.tsx`](../src/ui/app/page.tsx).
