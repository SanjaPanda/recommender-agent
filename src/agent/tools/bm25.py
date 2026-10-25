"""Exact BM25 for the bounded local catalog; no external search service."""
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from dataclasses import dataclass
from types import MappingProxyType
import math
import re


def tokens(text: str) -> list[str]:
    """Tokenize Unicode words after removing source HTML markup."""
    return re.findall(r"[^\W_]+", re.sub(r"<[^>]*>", " ", text).casefold())


@lru_cache(maxsize=4096)
def _document_counts(document: str):
    """Content-keyed immutable tokens; edits naturally receive a new key."""
    counts = Counter(tokens(document))
    return MappingProxyType(dict(counts)), sum(counts.values())


@dataclass(frozen=True)
class Corpus:
    """Immutable token statistics for one ordered, SQL-eligible document corpus."""
    counts: tuple[Mapping[str, int], ...]
    lengths: tuple[int, ...]
    average: float
    frequencies: Mapping[str, int]


@lru_cache(maxsize=8)
def _prepare_corpus(documents: tuple[str, ...]) -> Corpus:
    """Index exactly the eligible corpus, including document order/duplicates."""
    prepared = [_document_counts(document) for document in documents]
    counts = tuple(item[0] for item in prepared)
    lengths = tuple(item[1] for item in prepared)
    frequencies = Counter()
    for count in counts:
        frequencies.update(count.keys())
    return Corpus(counts, lengths, sum(lengths) / len(lengths) if lengths else 0,
                  MappingProxyType(dict(frequencies)))


def rank(query: str, documents: list[str], *, k1: float = 1.2, b: float = 0.75) -> list[tuple[int, float]]:
    """Return positive Okapi BM25 scores in stable descending order.

    Statistics are calculated over the eligible filtered corpus. Repeated query
    words count once. Scores are lexical relevance, not cosine or probability.

    Args:
        query: Keyword request, optionally expanded with equivalent phrases.
        documents: Ordered text of every eligible product in the filtered corpus.
        k1: Term-frequency saturation parameter; defaults to 1.2.
        b: Document-length normalization weight; defaults to 0.75.

    Returns:
        Positive (document index, score) pairs, descending by score with stable
        index order for ties. Empty/no-match queries return an empty list.
    """
    terms = set(tokens(query))
    if not terms:
        return []
    corpus = _prepare_corpus(tuple(documents))
    counts, lengths, average = corpus.counts, corpus.lengths, corpus.average
    if not terms or not average:
        return []
    frequencies = corpus.frequencies
    scores = []
    for index, count in enumerate(counts):
        score = 0.0
        for term in terms:
            tf = count.get(term, 0)
            if tf:
                df = frequencies[term]
                idf = math.log(1 + (len(counts) - df + 0.5) / (df + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[index] / average))
        if score > 0:
            scores.append((index, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))
