"""A2UI v0.9.1 operations for the fixed product-gallery surface."""
from __future__ import annotations

from typing import Any


CATALOG_ID = "copilotkit://recommender-product-catalog"


def product_gallery_operations(
    products: list[dict[str, Any]], *, heading: str, surface_id: str, create_surface: bool
) -> list[dict[str, Any]]:
    """Create a gallery surface once, then update its cards for later turns."""
    operations = []
    if create_surface:
        # Session resume, reset, discovery and cart responses can reuse an ID.
        # The renderer rejects duplicate createSurface and skips the whole batch.
        # Deletion is a no-op when absent; replace before sending the snapshot.
        operations.append({
            "version": "v0.9",
            "deleteSurface": {"surfaceId": surface_id},
        })
        operations.append({
            "version": "v0.9",
            "createSurface": {"surfaceId": surface_id, "catalogId": CATALOG_ID},
        })
    operations.extend([
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "id": "root",
                        "component": "ProductGallery",
                        "heading": {"path": "/heading"},
                        "products": {"path": "/products"},
                    },
                ],
            },
        },
        {
            "version": "v0.9",
            "updateDataModel": {
                "surfaceId": surface_id,
                "path": "/",
                "value": {"heading": heading, "products": products},
            },
        },
    ])
    return operations
