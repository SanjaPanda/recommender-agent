"""Conservative title/query product families for retrieval eligibility.

Only explicit positive product nouns are used. Ambiguous or unknown types are
left unresolved; these rules do not use extracted attribute candidates.
"""
import re

PATTERNS = {
    "shirt": r"(?:t[- ]?)?shirts?|blouses?",
    "dress": r"dresses|dress",
    "jacket": r"jackets?|coats?|windbreakers?|outwear",
    "sweater": r"sweaters?|cardigans?",
    "jeans": r"jeans",
    "pants": r"pants|trousers",
    "shorts": r"shorts",
    "sandal": r"sandals?",
    "shoe": r"shoes?|sneakers?|trainers?",
    "boot": r"boots?",
    "loafer": r"loafers?",
    "hat": r"hats?|caps?|beanies?|toques?",
    "glove": r"gloves?|mittens?",
    "sock": r"socks?",
    "necklace": r"necklaces?",
    "bracelet": r"bracelets?|bangles?",
    "brooch": r"brooch(?:es)?",
    "bowtie": r"bow[- ]?ties?",
    "tie": r"ties?|neckties?",
    "belt": r"belts?",
    "bra": r"bras?|brassieres?",
}

def product_type(text: str) -> str | None:
    """Return one explicit family, or None for unknown, negated or mixed types.

    Compatibility clauses after for/with/fits and similar-to are excluded. Handle
    dress as a modifier in dress shirts/shoes and shirt as a modifier in shirt
    dresses. This deliberately leaves other multi-type titles unresolved.
    """
    head = re.split(r"\s+(?:for|with|fits|similar to)\s+", text, maxsplit=1, flags=re.I)[0]
    found = set()
    for family, pattern in PATTERNS.items():
        for match in re.finditer(r"\b(?:" + pattern + r")\b", head, re.I):
            prefix, suffix = head[max(0, match.start() - 30):match.start()], head[match.end():]
            if re.search(r"\b(?:not|no|without|avoid|exclude)\s+(?:\w+\s+){0,2}$", prefix, re.I):
                continue
            if family == "dress" and re.match(r"\s+(?:shirts?|shoes?|sandals?|pants)\b", suffix, re.I):
                continue
            if family == "shirt" and re.match(r"\s+dress(?:es)?\b", suffix, re.I):
                continue
            if family == "jacket" and re.match(r"\s+accessor(?:y|ies)\b", suffix, re.I):
                continue
            if family == "bowtie" and re.search(r"\bcostume\s+kits?\b", head, re.I):
                continue
            if family == "tie" and "bowtie" in found and not match.group().lower().startswith("neck"):
                # Repeated generic tie nouns do not erase an explicit bow tie.
                if not re.search(r"\b(?:and|or|versus|vs)\s+$", prefix, re.I):
                    continue
            found.add(family)
    return next(iter(found)) if len(found) == 1 else None


def requested_product_type(query: str, explicit: str | None = None) -> str | None:
    """Retain a query's bow-tie subtype when a stored filter only says tie."""
    query_type = product_type(query)
    explicit_type = product_type(explicit or "")
    if explicit_type == "tie" and query_type == "bowtie":
        return "bowtie"
    return product_type(explicit) if explicit else query_type


def sql_pattern(family: str) -> str:
    """Return a parameter value for PostgreSQL title-word eligibility checks."""
    return r"\m(?:" + PATTERNS[family] + r")\M"
