"use client";

import { useMemo, useState, type ComponentType, type ReactNode } from "react";
import { A2UIProvider } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/core/A2UIProvider.mjs";
import * as A2UIRendererExports from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/core/A2UIRenderer.mjs";
import { useA2UI } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/hooks/useA2UI.mjs";
import { catalog, ProductActions } from "./catalog";
import type { A2UIMessage, Product } from "./a2ui";

const A2UIRenderer = (A2UIRendererExports as unknown as {
  default: ComponentType<{ surfaceId: string; fallback?: ReactNode }>;
}).default;

type AgentResponse = {
  thread_id: string;
  a2ui_surface_id: string;
  query_context: string;
  selected_product_ids: string[];
  products: Product[];
  status: string;
  a2ui_operations: A2UIMessage[];
};

function RecommendationApp() {
  const [query, setQuery] = useState("");
  const [refinement, setRefinement] = useState("");
  const [session, setSession] = useState<AgentResponse | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [heading, setHeading] = useState("");
  const { processMessages } = useA2UI();
  const gallery = useMemo(() => Boolean(session?.a2ui_operations?.length), [session]);

  async function submit(path: string, payload: object) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(path, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "The recommender could not complete that request.");
      processMessages(body.a2ui_operations);
      setHeading(body.a2ui_operations.find((item: A2UIMessage) => item.updateDataModel)?.updateDataModel?.value?.heading ?? "Products");
      setSession(body);
      setSelected([]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The recommender could not complete that request.");
    } finally {
      setBusy(false);
    }
  }

  function discover(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    void submit("/api/discover", { query: query.trim() });
  }

  function finalize(event: React.FormEvent) {
    event.preventDefault();
    if (!session) return;
    void submit("/api/recommend", {
      thread_id: session.thread_id,
      selected_product_ids: selected,
      refinement: refinement.trim(),
    });
  }

  function toggle(productId: string) {
    setSelected((current) => current.includes(productId)
      ? current.filter((id) => id !== productId)
      : current.length < 3 ? [...current, productId] : current);
  }

  return (
    <main className="page-shell">
      <header className="topbar"><a className="wordmark" href="#">fieldnote<span>®</span></a><span className="top-note">PRODUCT DISCOVERY, WITH CONTEXT</span></header>
      <section className="hero">
        <div className="eyebrow"><i /> YOUR NEXT GOOD FIND</div>
        <h1>Good things,<br /><em>found thoughtfully.</em></h1>
        <p className="intro">Describe what you have in mind. Pick a few things that feel right. We’ll use both to find your next favorite.</p>
        <form className="search-box" onSubmit={discover}>
          <label htmlFor="query">WHAT ARE YOU LOOKING FOR?</label>
          <div className="search-row"><input id="query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. lightweight tools for a small garden" maxLength={1000} />
            <button className="dark-button" disabled={busy || !query.trim()}>{busy && !session ? "Finding…" : "Find my starting point"}<span aria-hidden>↗</span></button></div>
        </form>
      </section>

      {gallery && <section className="results" aria-live="polite">
        <ProductActions.Provider value={{ selected, recommendations: heading === "Recommended for you", toggle }}>
          <A2UIRenderer surfaceId={session!.a2ui_surface_id} fallback={<p className="busy-line">Loading product cards…</p>} />
        </ProductActions.Provider>
        {heading !== "Recommended for you" && <form className="refine-panel" onSubmit={finalize}>
          <div className="refine-copy"><div className="eyebrow">MAKE IT MORE YOU</div><h2>What should we look for?</h2><p>Select up to three examples, then add any details that matter.</p></div>
          <textarea value={refinement} onChange={(e) => setRefinement(e.target.value)} placeholder="A color, material, size, use case, or anything else…" maxLength={1000} />
          <div className="submit-row"><span>We’ll combine your examples with your description.</span><button className="dark-button" disabled={busy || (!selected.length && !refinement.trim())}>{busy ? "Finding matches…" : "Finalize and recommend"}<span aria-hidden>↗</span></button></div>
        </form>}
      </section>}

      {busy && session && <div className="busy-line"><span /> Updating your recommendations…</div>}
      {error && <p className="error-message" role="alert">{error}</p>}
      <footer><span>Made for the way you look.</span><span>LOCAL CATALOGUE · PRIVATE BY DESIGN</span></footer>
    </main>
  );
}

export default function Home() {
  return <A2UIProvider catalog={catalog}><RecommendationApp /></A2UIProvider>;
}
