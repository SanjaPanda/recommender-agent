"""Shared apparel catalog boundary and maintenance lock for ingestion/retrieval."""

APPAREL_CATEGORY = "Apparel & Accessories"

# Serialize catalog imports and embedding writes across maintenance processes.
CATALOG_WRITE_LOCK = 0x524147


def validate_category_prefixes(prefixes: list[str]) -> list[str]:
    """Validate optional subcategory paths against the fixed apparel root.

    Args:
        prefixes: Inclusive taxonomy paths, or an empty list for the full catalog.

    Returns:
        The supplied list, unchanged. This checks scope, not catalog membership.

    Raises:
        ValueError: If an entry is not a string or is outside the apparel root.
    """
    if any(not isinstance(prefix, str) or
           (prefix != APPAREL_CATEGORY and not prefix.startswith(APPAREL_CATEGORY + " > "))
           for prefix in prefixes):
        raise ValueError("Category filters must stay within Apparel & Accessories.")
    return prefixes
