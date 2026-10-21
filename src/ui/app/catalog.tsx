"use client";

import { createContext, useContext } from "react";
import { createCatalog, type CatalogDefinitions, type CatalogRenderers } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/create-catalog.mjs";
import { z } from "../node_modules/@copilotkit/a2ui-renderer/node_modules/zod";
import type { Product } from "./a2ui";

const DynamicString = z.union([z.string(), z.object({ path: z.string() })]);
const ProductSchema = z.object({
  id: z.string(), title: z.string(), description: z.string().nullable().optional(),
  brand: z.string().nullable().optional(), category: z.string().nullable().optional(),
  image_url: z.string().nullable().optional(),
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
      <div className="product-grid">{products.map((product, index) => {
        const active = actions.selected.includes(product.id);
        return <div key={product.id}><button className={`product-card ${active ? "is-selected" : ""}`} onClick={() => actions.toggle(product.id)} aria-pressed={active} disabled={actions.busy}>
          <div className="image-wrap">{product.image_url ? <img src={product.image_url} alt={product.title} loading={index < 3 ? "eager" : "lazy"} /> : <div className="image-placeholder">Image unavailable</div>}
            <span className={`pick-indicator ${active ? "checked" : ""}`}>{active ? "✓" : "+"}</span></div>
          <div className="product-info"><span className="product-number">{String(index + 1).padStart(2, "0")}</span><div><h3>{product.title || "Untitled product"}</h3><p>{product.brand || product.category || "Catalogue find"}</p></div></div>
        </button><button type="button" disabled={actions.busy} onClick={() => actions.addToCart(product.id)}>Add to cart</button></div>;
      })}</div>
      {!products.length && <p className="empty-state">No matches surfaced. Try describing the product a little differently.</p>}
    </section>;
  },
};

export const catalog = createCatalog(definitions, renderers, {
  catalogId: "copilotkit://recommender-product-catalog",
  includeBasicCatalog: true,
});
