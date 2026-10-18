"""Normalize Shopify catalogue source rows without filling missing data."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def clean_text(value: Any) -> str | None:
    """Collapse internal whitespace and preserve empty or absent values as None."""
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


def clean_categories(value: Any) -> str | None:
    """Normalize category lists into a readable, pipe-separated string."""
    if isinstance(value, (list, tuple)):
        values = [clean_text(item) for item in value]
        return " | ".join(item for item in values if item) or None
    return clean_text(value)


@dataclass(frozen=True, slots=True)
class ProductRecord:
    """Normalized catalogue product fields plus the untouched source attributes."""

    id: str
    title: str | None
    description: str | None
    brand: str | None
    category: str | None
    is_secondhand: bool | None
    image: dict[str, Any] | None
    raw_source: dict[str, Any]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ProductRecord:
        """Parse one saved sample row; keep original attributes in ``raw_source``."""
        source = {key: value for key, value in row.items() if key != "id"}
        product_id = row.get("id")
        if not product_id:
            canonical = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            product_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        category = clean_categories(source.get("potential_product_categories"))
        if category is None:
            category = clean_text(source.get("ground_truth_category"))
        return cls(
            id=product_id,
            title=clean_text(source.get("product_title")),
            description=clean_text(source.get("product_description")),
            brand=clean_text(source.get("ground_truth_brand")),
            category=category,
            is_secondhand=source.get("ground_truth_is_secondhand"),
            image=source.get("product_image"),
            raw_source=source,
        )
