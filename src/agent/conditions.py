"""Recognize unsupported product-condition requests before retrieval."""
import re

UNSUPPORTED_CONDITION = (
    'Condition filtering is not supported: this catalog does not reliably identify '
    'new, used, or pre-owned products. Please search without a condition requirement.'
)


def requests_condition(text: str) -> bool:
    """Recognize explicit condition language without treating New Balance as new.

    Args:
        text: Current request or effective saved query/preferences.

    Returns:
        Whether condition support is required. Common new-search commands and
        proper names remain ordinary requests; ambiguous language uses the router.
    """
    return bool(re.search(
        r'\b(?:brand[ -]new|unused|unworn|never (?:been )?worn|worn once|'
        r'second[ -]?hand|pre[ -]?owned|pre[ -]?loved|thrifted|refurbished|condition|'
        r'used\b(?!\s+to\b)|'
        r'new\b(?!\s+(?:balance|york|era|look|search|query|session|thread|request|'
        r'arrivals?|collection|styles?|designs?)\b))\b', text, re.I))
