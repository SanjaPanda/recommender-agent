"use client";

import type { ComponentType, ReactNode } from "react";
import * as RendererExports from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/core/A2UIRenderer.mjs";
import { z } from "../node_modules/@copilotkit/a2ui-renderer/node_modules/zod";
export { z };
export { A2UIProvider } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/core/A2UIProvider.mjs";
export { useA2UI } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/hooks/useA2UI.mjs";
export { createCatalog, type CatalogDefinitions, type CatalogRenderers } from "../node_modules/@copilotkit/a2ui-renderer/dist/react-renderer/create-catalog.mjs";

// Keep the installed renderer's internal import/export shape at one boundary.
export const A2UIRenderer = (RendererExports as unknown as {
  default: ComponentType<{ surfaceId: string; fallback?: ReactNode }>;
}).default;

export const ProductSchema = z.object({
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

export type Product = z.infer<typeof ProductSchema>;

export type AgentResponse = {
  thread_id: string; a2ui_surface_id: string; products: Product[];
  selected_product_ids: string[]; status: string; message: string;
  query_context: string;
  search_intent?: { product_phrase: string; search_terms: string[];
    attributes: { name: string; value: string; evidence: string; excluded: boolean }[] } | null;
  categories: { name: string; count: number }[];
  cart: { id: string; title: string }[];
  preferences: { requirement: string; strength: string }[];
  filters: { brand?: string | null; category_prefixes?: string[]; title_color?: string | null; title_product?: string | null };
  removed_filters?: string[];
  trace: AgentTraceEvent[];
  a2ui_operations: A2UIMessage[];
};

/** Display cosine similarity on a percentage scale, preserving negative values. */
export function similarityPercent(value: number | null | undefined): string | null {
  return typeof value === "number" && Number.isFinite(value) && value >= -1 && value <= 1
    ? `${(value * 100).toFixed(1)}%` : null;
}

export type A2UIMessage = Record<string, any>;

export type AgentTraceEvent = {
  node: "router" | "retrieve" | "respond";
  duration_ms: number;
  route?: string | null;
  status?: string | null;
  result_count?: number | null;
  category_count?: number | null;
  text_count?: number | null;
  clip_count?: number | null;
  example_count?: number | null;
};
