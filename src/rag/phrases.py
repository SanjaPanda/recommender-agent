"""Literal product-phrase evidence with generic spelling/plural handling."""
import re


def phrase_pattern(phrase: str) -> str:
    """Build an escaped pattern; no product-specific types or synonyms."""
    words = re.findall(r"[^\W_]+", phrase.casefold())
    if not words or len(words) > 8 or len(phrase) > 100:
        raise ValueError('Product phrase must contain one to eight words.')
    last = words[-1]
    variants = {last}
    if last.endswith('s') and not last.endswith(('ss', 'us')):
        variants.add(last[:-1])
        if last.endswith(('ses', 'xes', 'zes', 'ches', 'shes')):
            variants.add(last[:-2])
        if last.endswith('ies'):
            variants.add(last[:-3] + 'y')
    else:
        variants.add(last + ('es' if last.endswith(('s', 'x', 'z', 'ch', 'sh')) else 's'))
        if last.endswith('y'):
            variants.add(last[:-1] + 'ies')
    tail = '(?:' + '|'.join(re.escape(word) for word in sorted(variants)) + ')'
    return r'[-\s]*'.join([*(re.escape(word) for word in words[:-1]), tail])


def phrase_in_title(title: str, phrase: str) -> bool:
    """Require a phrase at the end of a primary title segment.

    A following unpunctuated word leaves the noun phrase unresolved: ``dress``
    in ``Dress Shoes`` cannot establish a dress. Commas, parentheses, numeric
    sizes and other explicit listing separators delimit a segment. This is
    conservative text evidence, not visual or semantic product verification.
    """
    head = re.split(r'\s+(?:for|with|fits|similar to)\s+', title, maxsplit=1, flags=re.I)[0]
    for match in re.finditer(r'\b' + phrase_pattern(phrase) + r'\b', head, re.I):
        prefix = head[max(0, match.start() - 40):match.start()]
        suffix = head[match.end():].lstrip()
        boundary = not suffix or not suffix[0].isalpha()
        if boundary and not re.search(r'\b(?:not|no|without|never|avoid|exclude)\s+(?:\w+\s+){0,2}$', prefix, re.I):
            return True
    return False
