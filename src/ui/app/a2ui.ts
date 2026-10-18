"use client";

/** Small A2UI v0.9.1 client for the fixed ProductGallery catalog component. */
export type Product = {
  id: string;
  title: string;
  description?: string | null;
  brand?: string | null;
  category?: string | null;
  image_url?: string | null;
};

export type A2UIMessage = Record<string, any>;

export function readProductGallery(operations: A2UIMessage[]) {
  const update = operations.find((operation) => operation.updateDataModel)?.updateDataModel;
  const components = operations.find((operation) => operation.updateComponents)?.updateComponents?.components ?? [];
  const hasGallery = components.some((component: any) => component.component === "ProductGallery");
  return hasGallery ? { heading: update?.value?.heading ?? "Products", products: (update?.value?.products ?? []) as Product[] } : null;
}
