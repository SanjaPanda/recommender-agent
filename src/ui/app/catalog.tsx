"use client";

import { createContext, useContext } from "react";
import { createCatalog, type CatalogDefinitions, type CatalogRenderers } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/create-catalog.mjs";
import { z } from "../node_modules/@copilotkit/a2ui-renderer/node_modules/zod";
import type { Product } from "./a2ui";
import { similarityPercent } from "./a2ui";

const DynamicString = z.union([z.string(), z.object({ path: z.string() })]);
const ProductSchema = z.object({
  id: z.string(), title: z.string(), description: z.string().nullable().optional(),
  brand: z.string().nullable().optional(), category: z.string().nullable().optional(),
  image_url: z.string().nullable().optional(),
  bm25_score: z.number().nullable().optional(),
  rrf_score: z.number().nullable().optional(),
  rrf_k: z.number().optional(),
  rrf_contributions: z.array(z.object({ signal: z.string(), rank: z.number(), weight: z.number(), contribution: z.number() })).optional(),
  text_similarity: z.number().nullable().optional(),
  image_similarity: z.number().nullable().optional(),
  example_similarity: z.number().nullable().optional(),
  match_checks: z.array(z.object({ label: z.string(), status: z.enum(["match", "mismatch", "unknown"]), evidence: z.string() })).optional(),
  listing: z.object({ description: z.string().nullable().optional(), features: z.array(z.string()),
    details: z.record(z.string(), z.string()), average_rating: z.number().nullable().optional(),
    rating_number: z.number().nullable().optional(), store: z.string().nullable().optional() }).optional(),
});

export const ProductActions = createContext<{
  selected: string[];
  toggle: (id: string) => void;
  addToCart: (id: string) => void;
  busy: boolean;
}>({ selected: [], toggle: () => undefined, addToCart: () => undefined, busy: false });

const definitions = {
  ProductGallery: {
    description: "A five-item product image gallery for selecting similar-product examples.",
    props: z.object({ heading: DynamicString, products: z.union([z.array(ProductSchema), z.object({ path: z.string() })]) }),
  },
} satisfies CatalogDefinitions;

const renderers: CatalogRenderers<typeof definitions> = {
  ProductGallery: ({ props }) => {
    const actions = useContext(ProductActions);
    const products = Array.isArray(props.products) ? props.products as Product[] : [];
    const heading = typeof props.heading === "string" ? props.heading : "Products";
    const eyebrow = heading === "Closest alternatives" ? "CLOSEST AVAILABLE OPTIONS"
      : heading === "Recommended for you" ? "A BETTER FIT, FOUND" : "A GOOD PLACE TO START";
    return <section className="a2ui-gallery">
      <div className="section-head"><div><div className="eyebrow">{eyebrow}</div><h2>{heading}</h2></div>
        <span className="selection-count">{actions.selected.length}/3 selected</span></div>
      <p className="similarity-help">Similarity compares product text and images with your search. It does not verify material, color, size, or suitability.</p>
      <div className="product-grid">{products.map((product, index) => {
        const active = actions.selected.includes(product.id);
        const signals = [
          { label: "Text similarity", value: similarityPercent(product.text_similarity) },
          { label: "Search-to-image similarity", value: similarityPercent(product.image_similarity) },
          { label: "Example-image similarity", value: similarityPercent(product.example_similarity) },
        ];
        const primary = signals.find(signal => signal.value !== null);
        return <div key={product.id}><button className={`product-card ${active ? "is-selected" : ""}`} onClick={() => actions.toggle(product.id)} aria-pressed={active} disabled={actions.busy}>
          <div className="image-wrap">{product.image_url ? <img src={product.image_url} alt={product.title} loading={index < 3 ? "eager" : "lazy"} /> : <div className="image-placeholder">Image unavailable</div>}
            <span className={`pick-indicator ${active ? "checked" : ""}`}>{active ? "✓" : "+"}</span></div>
          <div className="product-info"><span className="product-number">{String(index + 1).padStart(2, "0")}</span><div><h3>{product.title || "Untitled product"}</h3><p>{product.brand || product.category || "Catalogue find"}</p></div></div>
          <div className="similarity-badge">{primary ? <><span>{primary.label}</span><strong>{primary.value}</strong></> : <span>Similarity unavailable</span>}</div>
        </button>
        {Boolean(product.match_checks?.length) && <div className="listing-checks" aria-label="Listing checks">
          {product.match_checks!.map(check => <span key={check.label} className={`listing-check ${check.status}`} title={check.evidence}>
            <span aria-hidden="true">{check.status === "match" ? "✓" : check.status === "mismatch" ? "✕" : "?"}</span> {check.label}
            <span className="sr-only">: {check.status}</span>
          </span>)}
        </div>}
        <details className="product-details"><summary>Product details</summary>
          <h4>{product.title}</h4>
          <p><strong>Listing rating:</strong> {product.listing?.average_rating != null ? `${product.listing.average_rating} / 5` : "Not provided"}
            {product.listing?.rating_number != null ? ` (${product.listing.rating_number.toLocaleString()} ratings)` : ""}</p>
          {product.listing?.store && <p><strong>Store:</strong> {product.listing.store}</p>}
          {product.category && <p><strong>Category:</strong> {product.category}</p>}
          {Boolean(product.match_checks?.length) && <div><strong>Why these checks?</strong>{product.match_checks!.map(check => <p key={check.label}><strong>{check.label}:</strong> {check.evidence}</p>)}</div>}
          <strong>Description</strong><p className="source-description">{product.listing?.description || product.description || "Not provided"}</p>
          <strong>Features</strong>{product.listing?.features.length ? <ul>{product.listing.features.map((feature, i) => <li key={i}>{feature.replace(/<[^>]*>/g, " ")}</li>)}</ul> : <p>Not provided</p>}
          <strong>Listing attributes</strong>{product.listing && Object.keys(product.listing.details).length ? <dl>{Object.entries(product.listing.details).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl> : <p>Not provided</p>}
          <p>Checks use title and structured listing text. Multiple or missing colors are unknown. Image color and variant applicability are unverified. Ratings are from the source snapshot.</p>
        </details>
        <details className="similarity-details"><summary>Similarity details</summary>
          <p>Position {index + 1} in this result list. Available rankings are combined to order products.</p>
          <dl>{signals.map(signal => <div key={signal.label}><dt>{signal.label}</dt><dd>{signal.value ?? "Unavailable"}</dd></div>)}</dl>
          <dl><div><dt>BM25 keyword score</dt><dd>{product.bm25_score != null ? product.bm25_score.toFixed(3) : "Not in keyword results"}</dd></div>
            <div><dt>RRF score</dt><dd>{product.rrf_score != null ? product.rrf_score.toFixed(6) : "Unavailable for browsing"}</dd></div></dl>
          {Boolean(product.rrf_contributions?.length) && <ul>{product.rrf_contributions!.map(part => <li key={part.signal}>
            {({ bm25_score: "Keywords", text_similarity: "Text", image_similarity: "Search image", example_similarity: "Selected examples" } as Record<string, string>)[part.signal] ?? part.signal}: rank {part.rank}, weight {(part.weight * 100).toFixed(1)}%, contribution {part.contribution.toFixed(6)}
          </li>)}</ul>}
          <p>RRF adds weight / ({product.rrf_k ?? 60} + signal rank). BM25 rewards matching words in the title, description and features. These ranking scores are not percentages; higher ranks first within the same eligibility rules.</p>
          <p>Percentages are cosine similarity × 100. Each signal has its own scale; compare products within the same signal. They are not match probabilities. List position is not calculated by averaging these percentages.</p>
          {product.category && <p><strong>Category:</strong> {product.category}</p>}
          {product.description && <details><summary>Source description</summary><p className="source-description">{product.description}</p></details>}
        </details>
        <button type="button" disabled={actions.busy} onClick={() => actions.addToCart(product.id)}>Add to cart</button></div>;
      })}</div>
      {!products.length && <p className="empty-state">No matches surfaced. Try describing the product a little differently.</p>}
    </section>;
  },
};

export const catalog = createCatalog(definitions, renderers, {
  catalogId: "copilotkit://recommender-product-catalog",
  includeBasicCatalog: true,
});
