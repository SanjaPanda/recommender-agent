"use client";

import { useEffect, useRef, useState, type ComponentType, type ReactNode } from "react";
import { A2UIProvider } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/core/A2UIProvider.mjs";
import * as A2UIRendererExports from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/core/A2UIRenderer.mjs";
import { useA2UI } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/hooks/useA2UI.mjs";
import { catalog, ProductActions } from "./catalog";
import type { A2UIMessage, Product } from "./a2ui";

const A2UIRenderer = (A2UIRendererExports as unknown as {
  default: ComponentType<{ surfaceId: string; fallback?: ReactNode }>;
}).default;

type AgentResponse = {
  thread_id: string; a2ui_surface_id: string; products: Product[];
  selected_product_ids: string[]; status: string; message: string;
  categories: { name: string; count: number }[];
  cart: { id: string; title: string }[];
  preferences: { requirement: string; strength: string }[];
  filters: { brand?: string | null; is_secondhand?: boolean | null; category_prefixes?: string[] };
  a2ui_operations: A2UIMessage[];
};

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
      {session && <button disabled={busy} onClick={() => void submit("/api/discover", { thread_id: session.thread_id, query: "#buynew" })}>Start a new search</button>}
      {session?.status === "categories" && <section aria-label="Product categories" className="category-list">
        {session.categories.map(category => <button key={category.name} disabled={busy}
          onClick={() => void submit("/api/discover", { thread_id: session.thread_id, query: `Show products in ${category.name}` })}>
          {category.name} ({category.count})</button>)}
      </section>}
      <form className="search-box" onSubmit={chat}>
        <label htmlFor="query">WHAT ARE YOU LOOKING FOR?</label>
        <div className="search-row"><input id="query" value={query} onChange={e => setQuery(e.target.value)}
          placeholder="Describe a product or refine your preferences" maxLength={1000} />
          <button className="dark-button" disabled={busy || !session || (!query.trim() && !selected.length)}>{busy ? "Working..." : "Send"}</button></div>
      </form>
      {selected.length > 0 && <p>{selected.length} visual example(s) active. <button disabled={busy} onClick={() => setSelected([])}>Clear examples</button></p>}
      {session && <div aria-label="Active preferences">
        {session.filters?.brand && <span className="preference-tag">Brand: {session.filters.brand}</span>}
        {session.filters?.is_secondhand != null && <span className="preference-tag">{session.filters.is_secondhand ? "Secondhand" : "Brand new"}</span>}
        {session.filters?.category_prefixes?.map(c => <span key={c} className="preference-tag">{c}</span>)}
        {session.preferences?.map(p => <span key={p.requirement} className="preference-tag">{p.requirement} ({p.strength})</span>)}
      </div>}
    </section>
    {session?.message && <p className="agent-message" aria-live="polite">{session.message}</p>}
    {Boolean(session?.products.length) && <section className="results">
      <ProductActions.Provider value={{ selected, recommendations: false, toggle, addToCart: id => cart("cart_add", id), busy }}>
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
