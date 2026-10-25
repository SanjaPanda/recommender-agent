"use client";

/** Small A2UI v0.9.1 client for the fixed ProductGallery catalog component. */
export type Product = {
  id: string;
  title: string;
  description?: string | null;
  brand?: string | null;
  category?: string | null;
  image_url?: string | null;
  bm25_score?: number | null;
  rrf_score?: number | null;
  rrf_k?: number;
  rrf_contributions?: { signal: string; rank: number; weight: number; contribution: number }[];
  text_similarity?: number | null;
  image_similarity?: number | null;
  example_similarity?: number | null;
  match_checks?: { label: string; status: "match" | "mismatch" | "unknown"; evidence: string }[];
  listing?: {
    description?: string | null; features: string[]; details: Record<string, string>;
    average_rating?: number | null; rating_number?: number | null; store?: string | null;
  };
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
