"use client";

import { useEffect, useRef, useState, type ComponentType, type ReactNode } from "react";
import { A2UIProvider } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/core/A2UIProvider.mjs";
import * as A2UIRendererExports from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/core/A2UIRenderer.mjs";
import { useA2UI } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/hooks/useA2UI.mjs";
import { catalog, ProductActions } from "./catalog";
import type { A2UIMessage, AgentTraceEvent, Product } from "./a2ui";

const A2UIRenderer = (A2UIRendererExports as unknown as {
  default: ComponentType<{ surfaceId: string; fallback?: ReactNode }>;
}).default;

type AgentResponse = {
  thread_id: string; a2ui_surface_id: string; products: Product[];
  selected_product_ids: string[]; status: string; message: string;
  query_context: string;
  search_intent?: { product_phrase: string; search_terms: string[];
    attributes: { name: string; value: string; evidence: string; excluded: boolean }[] } | null;
  categories: { name: string; count: number }[];
  cart: { id: string; title: string }[];
  preferences: { requirement: string; strength: string }[];
  filters: { brand?: string | null; is_secondhand?: boolean | null; category_prefixes?: string[]; title_color?: string | null; title_product?: string | null };
  trace: AgentTraceEvent[];
  a2ui_operations: A2UIMessage[];
};

const EXAMPLE_SEARCHES = ["Blue clothes", "Black cotton dress", "Pink bow tie"];

function RecommendationApp() {
  const [query, setQuery] = useState("");
  const [session, setSession] = useState<AgentResponse | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const initialized = useRef(false);
  const { processMessages } = useA2UI();

  async function submit(path: string, payload: object) {
    setBusy(true); setError("");
    try {
      const response = await fetch(`http://127.0.0.1:8000${path}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Unable to complete this request.");
      processMessages(body.a2ui_operations);
      setSession(body); setSelected(body.selected_product_ids ?? []);
      localStorage.setItem("recommender-thread", body.thread_id);
      setQuery("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to complete this request.");
    } finally { setBusy(false); }
  }

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void submit("/api/session", { thread_id: localStorage.getItem("recommender-thread") });
  }, []);

  function chat(event: React.FormEvent) {
    event.preventDefault();
    if (!session || (!query.trim() && !selected.length)) return;
    void submit("/api/recommend", { thread_id: session.thread_id,
      selected_product_ids: selected, refinement: query.trim() });
  }

  function toggle(id: string) {
    setSelected(current => current.includes(id) ? current.filter(x => x !== id) : current.length < 3 ? [...current, id] : current);
  }

  function cart(action: "cart_add" | "cart_remove", id: string) {
    if (session) void submit("/api/cart", { thread_id: session.thread_id, action, product_ids: [id] });
  }

  return <main className="page-shell">
    <header className="topbar"><a className="wordmark" href="#">fieldnote</a><span className="top-note">PRODUCT DISCOVERY, WITH CONTEXT</span></header>
    <section className="hero">
      <h1>Good things,<br /><em>found thoughtfully.</em></h1>
      <p className="intro">Describe your next find, choose examples, and refine what matters.</p>
      {session && <button className="new-session-button" disabled={busy} onClick={() => void submit("/api/session", {})}>New session</button>}
      {session?.status === "categories" && <section aria-label="Example searches" className="example-searches">
        <p>Try a search</p>
        <div>{EXAMPLE_SEARCHES.map(example => <button key={example} disabled={busy}
          onClick={() => void submit("/api/discover", { query: example })}>{example}</button>)}</div>
      </section>}
      <form className="search-box" onSubmit={chat}>
        <label htmlFor="query">WHAT ARE YOU LOOKING FOR?</label>
        <div className="search-row"><input id="query" value={query} onChange={e => setQuery(e.target.value)}
          placeholder="Describe a product or refine your preferences" maxLength={1000} />
          <button className="dark-button" disabled={busy || !session || (!query.trim() && !selected.length)}>{busy ? "Working..." : "Send"}</button></div>
      </form>
      {selected.length > 0 && <p>{selected.length} visual example(s) active. <button disabled={busy} onClick={() => setSelected([])}>Clear examples</button></p>}
      {session && session.status !== "categories" && <section className="search-applied" aria-label="Search applied">
        <strong>Search applied</strong>
        <p><span>Search:</span> {session.query_context || "Selected visual examples"}</p>
        {session.search_intent && <details><summary>How your request was understood</summary>
          <p><span>Product phrase:</span> {session.search_intent.product_phrase || "No specific product identified"}</p>
          {session.search_intent.attributes.map((a, index) => <p key={index}><span>{a.name}:</span> {a.excluded ? "Exclude " : ""}{a.value} <small>(from “{a.evidence}”)</small></p>)}
          {session.search_intent.search_terms.length > 0 && <p><span>Search terms:</span> {session.search_intent.search_terms.join(" · ")}</p>}
          <p>These describe your request. Product checks below use listing evidence.</p>
        </details>}
        <p><span>Catalog:</span> Apparel &amp; Accessories</p>
        {session.filters?.category_prefixes?.map(c => <p key={c}><span>Category:</span> {c}</p>)}
        {session.filters?.brand && <p><span>Brand:</span> {session.filters.brand}</p>}
        {session.filters?.is_secondhand != null && <p><span>Condition:</span> {session.filters.is_secondhand ? "Secondhand" : "Brand new"}</p>}
        {!session.search_intent && session.filters?.title_product && <p><span>Product:</span> {session.filters.title_product}</p>}
        {session.filters?.title_color && <p><span>Title color:</span> {session.filters.title_color}</p>}
        {session.preferences?.map(p => <p key={p.requirement}><span>Preference:</span> {p.requirement} ({p.strength})</p>)}
        {session.selected_product_ids.length > 0 && <p><span>Visual examples:</span> {session.selected_product_ids.length} selected</p>}
      </section>}
    </section>
    {session?.message && <p className="agent-message" aria-live="polite">{session.status === "categories" ? "Choose an example or describe what you want to find." : session.message}</p>}
    {Boolean(session?.trace.length) && <details className="agent-trace">
      <summary>Agent trace ({session!.trace.length} steps)</summary>
      <ol>{session!.trace.map((event, index) => <li key={`${event.node}-${index}`}>
        <span><strong>{event.node}</strong>{event.route ? ` → ${event.route}` : ""}{event.status ? ` · ${event.status}` : ""}</span>
        <small>{event.duration_ms} ms{event.result_count != null ? ` · ${event.result_count} products` : ""}{event.category_count != null ? ` · ${event.category_count} categories` : ""}{event.text_count != null ? ` · text ${event.text_count}` : ""}{event.clip_count != null ? ` · CLIP ${event.clip_count}` : ""}{event.example_count ? ` · examples ${event.example_count}` : ""}</small>
      </li>)}</ol>
    </details>}
    {Boolean(session?.products.length) && <section className="results">
      <ProductActions.Provider value={{ selected, toggle, addToCart: id => cart("cart_add", id), busy }}>
        <A2UIRenderer surfaceId={session!.a2ui_surface_id} fallback={<p>Loading products...</p>} />
      </ProductActions.Provider>
    </section>}
    {session && <section className="cart-panel" aria-label="Shopping cart"><h2>Your cart ({session.cart.length})</h2>
      {session.cart.length ? session.cart.map(p => <p key={p.id}>{p.title} <button disabled={busy} onClick={() => cart("cart_remove", p.id)}>Remove</button></p>) : <p>Your cart is empty.</p>}
    </section>}
    {error && <p className="error-message" role="alert">{error}</p>}
    {!session && !busy && <button onClick={() => void submit("/api/session", { thread_id: localStorage.getItem("recommender-thread") })}>Reconnect</button>}
  </main>;
}

export default function Home() {
  return <A2UIProvider catalog={catalog}><RecommendationApp /></A2UIProvider>;
}
