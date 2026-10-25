"""Deterministic relevance gates for retrieved apparel and fallback suggestions.

The ranker supplies Nomic text, CLIP text-to-image, and optional selected-example
similarities. This module applies conservative cutoffs plus title/description
term checks over source listing text and retrieval scores.
"""
from __future__ import annotations

import math
import re
from typing import Any
from rag.product_types import product_type, requested_product_type
from rag.phrases import phrase_in_title


TEXT_SIMILARITY_MIN = 0.61
CLIP_SIMILARITY_MIN = 0.28
EXAMPLE_SIMILARITY_MIN = 0.28
COLOR_TERMS = frozenset({
    "black", "white", "red", "blue", "green", "yellow", "pink", "purple",
    "orange", "brown", "grey", "gray", "beige", "silver", "gold",
})
GENERIC_TERMS = frozenset({
    "a", "an", "the", "and", "or", "for", "with", "in", "of", "to",
    "me", "my", "i", "want", "need", "find", "show", "get", "looking",
    "clothes", "clothing", "apparel", "wear", "outfit", "item", "items",
    "product", "products", "something", "nice", "good", "not", "no",
    "without", "never", "exclude", "avoid",
})
WORD_PATTERN = re.compile(r"[a-z0-9]+")
NEGATION_PATTERN = re.compile(r"\b(?:not|no|without|exclude|avoid|never)\s+([a-z]+)\b")


def is_relevant_product(
    product: dict[str, Any],
    query: str,
    *,
    has_visual_examples: bool = False,
    required_title_color: str | None = None,
    required_product_type: str | None = None,
    ignore_query_attributes: bool = False,
    product_phrase: str | None = None,
) -> bool:
    """Decide whether one ranked product is strong enough to call a match.

    Args:
        product: Fused retrieval result, including per-modality similarities.
        query: Positive search text and preserved positive preferences.
        has_visual_examples: Whether the rank also used selected catalog images.
        required_title_color: Explicit title color filter from the shopping plan.
        required_product_type: Explicit product type filter from the shopping plan.
        ignore_query_attributes: Relax color/product terms for fallback suggestions.
        product_phrase: Grounded title phrase; None allows conservative family
            inference, while an empty phrase deliberately leaves type unresolved.

    Returns:
        True when the product clears a retrieval cutoff and the available text
        supports the requested color/product terms. Fallback callers may relax
        query terms while retaining the same SQL category/brand scope.
    """
    title = product.get("title") or ""
    description = product.get("description") or ""
    title_tokens = _tokens(title)
    all_tokens = title_tokens | _tokens(description)
    # Feature words support lexical intent; colors retain the existing evidence rules.
    feature_tokens = _tokens(" ".join(feature for feature in product.get("features", [])
                                     if isinstance(feature, str)))
    lexical_match = False
    excluded_colors = _negated_colors(query)
    if excluded_colors & all_tokens:
        return False
    if product_phrase and not phrase_in_title(title, product_phrase):
        return False

    if not ignore_query_attributes:
        wanted_type = requested_product_type(query, required_product_type) if product_phrase is None else None
        if wanted_type and product_type(title) != wanted_type:
            return False
        colors = _positive_colors(query)
        if required_title_color:
            if required_title_color.casefold() not in title_tokens:
                return False
            colors.discard(required_title_color.casefold())
        if colors and not colors.issubset(all_tokens):
            # CLIP can support a visible color when the catalog text omits it.
            if (product.get("image_similarity") or 0.0) < CLIP_SIMILARITY_MIN:
                return False

        query_terms = _tokens(query) - COLOR_TERMS - GENERIC_TERMS - excluded_colors
        if product_phrase:
            # The complete product phrase has already been checked in the title.
            query_terms -= _tokens(product_phrase)
            lexical_match = True
        elif required_product_type and product_phrase is None:
            query_terms.add(_canonical(required_product_type.casefold()))
        if query_terms:
            overlap = query_terms & (all_tokens | feature_tokens)
            required_overlap = max(1, math.ceil(len(query_terms) * 0.5))
            if len(overlap) < required_overlap:
                return False
            lexical_match = True
        if colors or required_title_color:
            lexical_match = True

    if query.strip():
        return lexical_match or (
            (product.get("text_similarity") or 0.0) >= TEXT_SIMILARITY_MIN
            or (product.get("image_similarity") or 0.0) >= CLIP_SIMILARITY_MIN
        )
    if has_visual_examples:
        return (product.get("example_similarity") or 0.0) >= EXAMPLE_SIMILARITY_MIN
    return True


def _tokens(value: str) -> set[str]:
    """Tokenize catalog/query text and normalize basic English plurals."""
    return {
        _canonical(token)
        for token in WORD_PATTERN.findall(value.casefold())
        if token not in GENERIC_TERMS
    }


def _canonical(token: str) -> str:
    """Normalize regular plural forms without stemming unrelated words."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ies"):
        return token[:-1]
    if len(token) > 5 and token.endswith("es") and token[-3] in {"s", "x", "z", "c"}:
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _positive_colors(query: str) -> set[str]:
    """Return recognized color words that are stated positively."""
    tokens = WORD_PATTERN.findall(query.casefold())
    negative = _negated_colors(query)
    return (set(tokens) & COLOR_TERMS) - negative


def _negated_colors(query: str) -> set[str]:
    """Return colors explicitly excluded by phrases such as 'not blue'."""
    return {
        match.group(1)
        for match in NEGATION_PATTERN.finditer(query.casefold())
        if match.group(1) in COLOR_TERMS
    }
