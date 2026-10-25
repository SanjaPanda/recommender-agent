"""Listing-text checks and source details for explainable product cards.

Checks describe explicit title/structured-color evidence, not image verification
or inferred attribute truth. Missing or conflicting evidence stays unknown.
"""
import math
import re
from typing import Any

from rag.product_types import product_type, requested_product_type
from rag.phrases import phrase_in_title
from .relevance import COLOR_TERMS


def refresh_product_facts(product: dict[str, Any], facts: dict[str, Any],
                          *, invalidate_changed: bool = True) -> tuple[dict[str, Any], bool]:
    """Refresh a saved card and clear scores when its source identity changed.

    Args:
        product: Historical card or cart entry; never mutated.
        facts: Current database facts, including source and image hashes.
        invalidate_changed: False only for newly ranked, revision-checked cards.

    Returns:
        Updated card and whether its source hashes were stale. Unsupported condition
        facts are omitted; unknown condition is never presented as new.
    """
    stale = invalidate_changed and any(product.get(key) != facts.get(key)
                for key in ('source_record_sha256', 'image_sha256'))
    updated = {**product, **facts}
    updated.pop('is_secondhand', None)
    if stale:
        for key in ('score', 'rrf_score', 'bm25_score', 'text_similarity',
                    'image_similarity', 'example_similarity'):
            updated[key] = None
        updated['rrf_contributions'] = []
    return updated, stale


def listing_checks(title: str, details: dict[str, Any], query: str, filters: dict[str, Any],
                   product_phrase: str | None = None) -> list[dict[str, str]]:
    """Compare requested color/type with explicit listing evidence only."""
    colors = set(re.findall(r"\b[a-z]+\b", query.casefold())) & COLOR_TERMS
    negative = {m[1].lower() for m in re.finditer(r"\b(?:not|no|without|avoid|exclude)\s+([a-z]+)\b", query, re.I)} & COLOR_TERMS
    if filters.get("title_color"):
        colors.add(filters["title_color"].casefold())
    title_colors = set(re.findall(r"\b[a-z]+\b", title.casefold())) & COLOR_TERMS
    detail_colors = set()
    for key, value in details.items():
        if key.casefold() in {"color", "colour"} and isinstance(value, str):
            detail_colors |= set(re.findall(r"\b[a-z]+\b", value.casefold())) & COLOR_TERMS
    canonical = lambda values: {"gray" if v == "grey" else v for v in values}
    known = canonical(title_colors | detail_colors)
    checks = []
    for color in sorted(colors):
        target = "gray" if color == "grey" else color
        excluded = color in negative
        label = ("No " if excluded else "") + color.capitalize()
        if len(known) != 1:
            status, reason = "unknown", "Color is missing or multiple colors are listed; the selected image is not verified."
        else:
            matches = target in known
            status = "match" if matches != excluded else "mismatch"
            reason = "Title or structured color field lists " + ", ".join(sorted(known)) + "; image color is not verified."
        checks.append({"label": label, "status": status, "evidence": reason})
    if product_phrase:
        matches = phrase_in_title(title, product_phrase)
        checks.append({'label': product_phrase.capitalize(),
                       'status': 'match' if matches else 'unknown',
                       'evidence': 'Requested phrase ends a main title segment; semantic product identity and images are unverified.'
                       if matches else
                       'The main listing title does not establish the requested product phrase; synonyms are not verified.'})
        return checks
    wanted = requested_product_type(query, filters.get("title_product")) if product_phrase is None else None
    if wanted:
        actual = product_type(title)
        label = "Bow tie" if wanted == "bowtie" else wanted.capitalize()
        checks.append({"label": label, "status": "unknown" if actual is None else "match" if actual == wanted else "mismatch",
                       "evidence": "Product type is based on the listing title" + (f": {actual}." if actual else "; the title type is unresolved.")})
    return checks


def source_listing(raw: dict[str, Any], description: str | None, features: list[str], details: dict[str, Any]) -> dict[str, Any]:
    """Expose listing facts, preserving missing ratings as null rather than zero."""
    rating, count = raw.get("average_rating"), raw.get("rating_number")
    valid_rating = isinstance(rating, (int, float)) and not isinstance(rating, bool) and math.isfinite(rating) and 0 <= rating <= 5
    valid_count = isinstance(count, int) and not isinstance(count, bool) and count >= 0
    return {"description": description, "features": [v for v in features if isinstance(v, str)],
            "details": {str(k): str(v) for k, v in details.items() if isinstance(v, (str, int, float, bool))},
            "average_rating": rating if valid_rating else None, "rating_number": count if valid_count else None,
            "store": raw.get("store") if isinstance(raw.get("store"), str) else None}
