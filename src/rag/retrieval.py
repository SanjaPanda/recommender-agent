"""Apparel-only PostgreSQL retrieval and hybrid ranking."""
from __future__ import annotations

import logging
import re
import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from agent.config import settings
from . import APPAREL_CATEGORY, validate_category_prefixes
from .clip_runtime import text_embedding as clip_text_embedding
from .clip_runtime import model_version as clip_model_version
from .embed_text import MODEL as TEXT_MODEL, embed_texts, model_digest as text_model_version
from .bm25 import rank as bm25_rank
from .product_types import product_type, requested_product_type, sql_pattern
from .phrases import phrase_in_title, phrase_pattern

IMAGE_MODEL = settings.image_embedding_model
logger = logging.getLogger(__name__)
_category_cache = OrderedDict()
_category_cache_lock = Lock()


def retrieve_products(
    *,
    query_context: str = "",
    selected_product_ids: list[str] | None = None,
    category_prefixes: list[str] | None = None,
    limit: int = 5,
    filters: dict[str, Any] | None = None,
    product_phrase: str | None = None,
    search_terms: list[str] | None = None,
    excluded_product_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank apparel by product text, CLIP images, and selected examples.

    Query ranking fuses BM25, Nomic product-text and CLIP text-to-image
    reciprocal ranks with equal base weights (0.50 each), using k=60. Missing modes are renormalized.
    Selected catalog products add a CLIP image-example rank. SQL filters are
    applied before candidate limits, and every result stays under the apparel root.

    Args:
        query_context: Current query and preserved natural-language preferences.
        selected_product_ids: Up to three previously displayed apparel IDs.
        category_prefixes: Optional narrower apparel taxonomy paths.
        limit: Maximum results to return; must be positive.
        filters: Validated brand, product-title, and category filters.
        product_phrase: LLM-extracted literal product phrase; empty disables title-family inference.
        search_terms: LLM-proposed equivalent phrases used for lexical ranking only.
        excluded_product_ids: Session cart IDs omitted from every mode before limits.

    Returns:
        Products in fused ranking order with stable database IDs and per-mode scores.

    Raises:
        ValueError: If inputs or selected product IDs violate the retrieval contract.
        RuntimeError: If PostgreSQL or the Nomic embedding endpoint is unavailable.

    Example:
        >>> retrieve_products(query_context="blue cotton dress", limit=5)
    """
    from agent.conditions import requests_condition, UNSUPPORTED_CONDITION
    if requests_condition(query_context):
        raise ValueError(UNSUPPORTED_CONDITION)
    if limit < 1:
        raise ValueError("limit must be at least 1")
    refs = selected_product_ids or []
    if len(refs) > 3 or len(set(refs)) != len(refs):
        raise ValueError("select up to three unique product examples")
    active_filters = filters or {}
    exclude_refs = [] if active_filters.get("title_color") else refs
    wanted_type = requested_product_type(query_context, active_filters.get("title_product")) if product_phrase is None else None
    sql_filters = active_filters
    if product_phrase is not None or (wanted_type == "bowtie" and product_type(active_filters.get("title_product") or "") == "tie"):
        # The interpreted phrase takes precedence over single-word type inference.
        sql_filters = {**active_filters, "title_product": None}
    category_clause, category_params = filter_clause(sql_filters, category_prefixes or [])
    if excluded_product_ids:
        category_clause += " AND p.id <> ALL(%s)"
        category_params.append(excluded_product_ids)
    if product_phrase:
        category_clause += " AND (regexp_split_to_array(COALESCE(p.title, ''), '\\s+(?:for|with|fits|similar to)\\s+', 'i'))[1] ~* %s"
        # Apply the same conservative segment boundary before candidate limits;
        # otherwise high-ranked "Dress Shoes" can crowd actual dresses out.
        category_params.append(r"\m" + phrase_pattern(product_phrase) + r"\M(?=$|\s*[^[:alpha:][:space:]])")
    candidate_limit = max(limit * 10, 50)

    query_vector = None
    clip_vector = None
    text_version = image_version = None
    if query_context.strip():
        try:
            text_version = text_model_version()
            query_vector = embed_texts("search_query: " + query_context.strip())[0]
            if text_model_version() != text_version:
                raise RuntimeError('Text encoder changed during inference. Retry after pinning the installed tag.')
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
        try:
            clip_vector = clip_text_embedding(query_context)
            image_version = clip_model_version()
        except (RuntimeError, ValueError) as exc:
            clip_vector = None
            logger.warning("CLIP text retrieval unavailable; using Nomic text ranking: %s", exc)

    bm25_rows: list[tuple[Any, ...]] = []
    text_rows: list[tuple[Any, ...]] = []
    clip_rows: list[tuple[Any, ...]] = []
    example_rows: list[tuple[Any, ...]] = []
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
        register_vector(connection)
        if query_vector is None and not refs:
            rows = connection.execute(
                "SELECT p.id,p.title,p.description,p.brand,"
                "COALESCE(p.raw_source->>'ground_truth_category',p.category),p.image "
                "FROM products p WHERE true" + category_clause + " ORDER BY p.id LIMIT %s",
                (*category_params, limit),
            ).fetchall()
            return [
                {
                    "id": row[0], "title": row[1], "description": row[2], "brand": row[3],
                    "category": row[4], "image_meta": row[5],
                    "score": 0.0, "text_similarity": None,
                    "image_similarity": None, "example_similarity": None,
                    "visual_source": "hybrid",
                }
                for row in rows
            ]

        if refs:
            example_vector, image_version, titles = _selected_example_vector(connection, refs, image_version)
            if product_phrase is None and not query_context.strip() and not active_filters.get("title_product"):
                seed_types = {product_type(title) for title in titles}
                if len(seed_types) == 1 and None not in seed_types:
                    wanted_type = next(iter(seed_types))
        if wanted_type:
            category_clause += " AND COALESCE(p.title, '') ~* %s"
            category_params.append(sql_pattern(wanted_type))
        if refs:
            example_rows = _search_embedding(
                connection, vector=example_vector, modality="image", model=IMAGE_MODEL,
                model_version=image_version,
                category_clause=category_clause, category_params=category_params,
                excluded_ids=exclude_refs, limit=candidate_limit,
            )

        if query_vector is not None:
            bm25_rows = _search_bm25(
                connection, query=' '.join([query_context, *(search_terms or [])]), category_clause=category_clause,
                category_params=category_params, excluded_ids=exclude_refs, limit=candidate_limit,
            )
            text_rows = _search_embedding(
                connection, vector=Vector(query_vector), modality="text", model=TEXT_MODEL,
                model_version=text_version,
                category_clause=category_clause, category_params=category_params,
                excluded_ids=exclude_refs, limit=candidate_limit,
            )
            if clip_vector is not None:
                clip_rows = _search_embedding(
                    connection, vector=Vector(clip_vector), modality="image", model=IMAGE_MODEL,
                    model_version=image_version,
                    category_clause=category_clause, category_params=category_params,
                    excluded_ids=exclude_refs, limit=candidate_limit,
                )

    ranked_lists: list[tuple[list[tuple[Any, ...]], str, float]] = []
    if query_vector is not None:
        ranked_lists.extend([
            (bm25_rows, "bm25_score", 0.50),
            (text_rows, "text_similarity", 0.50),
            (clip_rows, "image_similarity", 0.50),
        ])
    if example_rows:
        ranked_lists.append((example_rows, "example_similarity",
                             0.20 if query_vector is not None else 1.0))
    return _fuse_rankings(ranked_lists, title_product=active_filters.get("title_product"),
                          wanted_type=wanted_type, product_phrase=product_phrase, limit=limit)


def _selected_example_vector(connection: psycopg.Connection, refs: list[str],
                             image_version: str | None) -> tuple[Vector, str, list[str]]:
    """Read eligible CLIP examples and validate revisions/dimensions before averaging.

    Args:
        connection: Retrieval connection with pgvector registered.
        refs: One to three unique selected apparel product IDs.
        image_version: Query CLIP revision, if available, to match against seeds.

    Returns:
        Normalized mean vector, immutable seed revision and seed titles.

    Raises:
        ValueError: If examples are missing, incompatible or have a zero mean.
    """
    reference_rows = connection.execute(
        """SELECT e.product_id,e.embedding::real[],p.title,e.model_version
           FROM product_embeddings e JOIN products p ON p.id=e.product_id
           WHERE e.product_id=ANY(%s) AND e.model_name=%s AND e.modality='image'
             AND e.status='ready'
             AND (p.raw_source->>'ground_truth_category'=%s OR
                  starts_with(p.raw_source->>'ground_truth_category',%s))""",
        (refs, IMAGE_MODEL, APPAREL_CATEGORY, APPAREL_CATEGORY + " > "),
    ).fetchall()
    available = {row[0] for row in reference_rows}
    missing = [product_id for product_id in refs if product_id not in available]
    if missing:
        raise ValueError("Selected apparel products have no CLIP image vectors: " + ", ".join(missing))
    versions = {row[3] for row in reference_rows}
    if len(versions) != 1 or not next(iter(versions)):
        raise ValueError('Selected examples use incompatible CLIP revisions. Rebuild their image vectors.')
    seed_version = next(iter(versions))
    if image_version is not None and image_version != seed_version:
        raise ValueError('Selected examples do not match the current CLIP encoder revision. Rebuild image vectors.')
    dimension = len(reference_rows[0][1])
    if any(len(row[1]) != dimension for row in reference_rows):
        raise ValueError('Selected examples have incompatible vector dimensions.')
    mean_vector = [0.0] * dimension
    for _, values, _, _ in reference_rows:
        for index, value in enumerate(values):
            mean_vector[index] += float(value) / len(reference_rows)
    norm = sum(value * value for value in mean_vector) ** 0.5
    if not norm:
        raise ValueError("Could not form a similarity vector from selected examples.")
    example_vector = Vector([value / norm for value in mean_vector])
    return example_vector, seed_version, [row[2] or "" for row in reference_rows]


def _fuse_rankings(ranked_lists: list[tuple[list[tuple[Any, ...]], str, float]],
                   *, title_product: str | None, wanted_type: str | None,
                   product_phrase: str | None, limit: int) -> list[dict[str, Any]]:
    """Fuse nonempty ranked lists with existing weights and conservative title gates.

    Args:
        ranked_lists: Candidate rows, score key and base weight per list.
        title_product: Explicit title filter used for primary-title ordering.
        wanted_type: Conservative title-family gate, or None for phrase searches.
        product_phrase: Required interpreted phrase, if present.
        limit: Maximum number of eligible products to return.

    Returns:
        Ranked product dictionaries with raw scores and individual RRF contributions.
    """
    available_weight = sum(weight for rows, _, weight in ranked_lists if rows)

    fused: dict[str, dict[str, Any]] = {}
    for rows, score_key, raw_weight in ranked_lists:
        if not rows or not available_weight:
            continue
        weight = raw_weight / available_weight
        for rank, row in enumerate(rows, start=1):
            product_id, title, description, brand, category, image_meta, similarity = row[:7]
            product = fused.setdefault(product_id, {
                "id": product_id, "title": title, "description": description, "brand": brand,
                "category": category, "image_meta": image_meta,
                "image_similarity": None, "example_similarity": None,
                "text_similarity": None, "bm25_score": None, "rrf_score": 0.0,
                "rrf_k": 60, "rrf_contributions": [], "score": 0.0, "visual_source": "hybrid",
            })
            if len(row) > 7:
                product["features"] = row[7]
            product[score_key] = float(similarity)
            contribution = weight / (60 + rank)
            product["score"] += contribution
            product["rrf_score"] = product["score"]
            product["rrf_contributions"].append({
                "signal": score_key, "rank": rank, "weight": weight, "contribution": contribution,
            })

    def ranking_key(item: dict[str, Any]) -> tuple[bool, float]:
        """Prioritize a requested type in the primary title, then retrieval score.

        Args:
            item: Fused candidate containing a product title and rank score.

        Returns:
            A descending sort tuple of title-match flag and fused score. When
            no type filter is active, candidates are ordered by score alone.
        """
        if not title_product:
            return False, item["score"]
        head = re.split(r"\s+(?:for|with|fits)\s+", item.get("title") or "",
                        maxsplit=1, flags=re.IGNORECASE)[0].strip()
        primary = bool(re.search(r"\b" + re.escape(title_product) + r"(?:s|es)?$", head, re.IGNORECASE))
        return primary, item["score"]

    eligible = [item for item in fused.values()
                if (wanted_type is None or product_type(item.get("title") or "") == wanted_type)
                and (not product_phrase or phrase_in_title(item.get("title") or "", product_phrase))]
    return sorted(eligible, key=ranking_key, reverse=True)[:limit]


def _search_bm25(connection, *, query, category_clause, category_params, excluded_ids, limit):
    """Score eligible source title, description and all feature bullets exactly."""
    exclusion = " AND p.id <> ALL(%s)" if excluded_ids else ""
    params = [*category_params, *([excluded_ids] if excluded_ids else [])]
    rows = connection.execute(
        "SELECT p.id,p.title,p.description,p.brand,"
        "COALESCE(p.raw_source->>'ground_truth_category',p.category),p.image,p.features "
        "FROM products p WHERE true" + category_clause + exclusion + " ORDER BY p.id", params,
    ).fetchall()
    documents = [" ".join([row[1] or "", row[2] or "", *(row[6] or [])]) for row in rows]
    return [(*rows[index][:6], score, rows[index][6]) for index, score in bm25_rank(query, documents)[:limit]]


def _search_embedding(
    connection: psycopg.Connection,
    *,
    vector: Vector,
    modality: str,
    model: str,
    model_version: str,
    category_clause: str,
    category_params: list[Any],
    excluded_ids: list[str],
    limit: int,
) -> list[tuple[Any, ...]]:
    """Run one exact vector search with shared apparel and product filters."""
    exclusion_clause = ""
    exclusion_params: list[Any] = []
    if excluded_ids:
        exclusion_clause = " AND e.product_id <> ALL(%s)"
        exclusion_params = [excluded_ids]
    return connection.execute(
        """SELECT p.id,p.title,p.description,p.brand,
                  COALESCE(p.raw_source->>'ground_truth_category',p.category),p.image,
                  1 - (e.embedding <=> %s) AS similarity
           FROM product_embeddings e JOIN products p ON p.id=e.product_id
           WHERE e.model_name=%s AND e.modality=%s AND e.status='ready'
             AND e.model_version=%s AND e.dimensions=%s"""
        + exclusion_clause + category_clause + " ORDER BY e.embedding <=> %s LIMIT %s",
        (vector, model, modality, model_version, vector.dimensions(), *exclusion_params, *category_params, vector, limit),
    ).fetchall()


def filter_clause(filters: dict[str, Any], category_prefixes: list[str]) -> tuple[str, list[Any]]:
    """Compile validated apparel filters to parameterized SQL predicates."""
    allowed = {"brand", "category_prefixes", "title_color", "title_product"}
    if set(filters) - allowed:
        raise ValueError("Unsupported catalogue filter")
    clauses: list[str] = []
    params: list[Any] = []
    prefixes = validate_category_prefixes(filters.get("category_prefixes") or category_prefixes)
    clauses.append(
        "((p.raw_source->>'ground_truth_category' = %s) OR "
        "starts_with(p.raw_source->>'ground_truth_category', %s))"
    )
    params.extend([APPAREL_CATEGORY, APPAREL_CATEGORY + " > "])
    prefixes = [prefix for prefix in prefixes if prefix != APPAREL_CATEGORY]
    if prefixes:
        terms = []
        for prefix in prefixes:
            terms.append("(p.raw_source->>'ground_truth_category' = %s OR "
                         "starts_with(p.raw_source->>'ground_truth_category', %s))")
            params.extend([prefix, prefix + " > "])
        clauses.append("(" + " OR ".join(terms) + ")")
    if filters.get("title_product") is not None:
        product = filters["title_product"]
        if not isinstance(product, str) or not re.fullmatch(r"[a-zA-Z]{1,100}", product):
            raise ValueError("Product type must be a single word")
        clauses.append("(regexp_split_to_array(COALESCE(p.title, ''), "
                       "'\\s+(?:for|with|fits)\\s+', 'i'))[1] ~* %s")
        params.append(sql_pattern("bowtie") if product_type(product) == "bowtie"
                      else r"\m" + product + r"(?:s|es)?\M")
    if filters.get("title_color") is not None:
        color = filters["title_color"]
        if not isinstance(color, str) or not re.fullmatch(r"[a-zA-Z]{1,30}", color):
            raise ValueError("Color must be a single color word")
        clauses.append("COALESCE(p.title, '') ~* %s")
        params.append(r"\m" + color + r"\M")
    if filters.get("brand") is not None:
        clauses.append("lower(p.brand) = lower(%s)")
        params.append(filters["brand"])
    return " AND " + " AND ".join(clauses), params


def list_categories() -> list[dict[str, Any]]:
    """Return snapshot-consistent counts, invalidated by committed catalog writes."""
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
        # The revision and miss query must see the same committed snapshot.
        connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        revision = connection.execute('SELECT revision FROM catalog_revision WHERE singleton = true').fetchone()[0]
        database = hashlib.sha256(settings.database_url.get_secret_value().encode()).digest()
        key = (database, revision)
        with _category_cache_lock:
            cached = _category_cache.get(key)
            if cached is not None:
                _category_cache.move_to_end(key)
                return [{'name': name, 'count': count} for name, count in cached]
        rows = connection.execute(
            "SELECT CASE WHEN split_part(raw_source->>'ground_truth_category', ' > ', 2) = '' "
            "THEN %s ELSE %s || ' > ' || split_part(raw_source->>'ground_truth_category', ' > ', 2) END,count(*) "
            "FROM products WHERE raw_source->>'ground_truth_category' = %s "
            "OR starts_with(raw_source->>'ground_truth_category', %s) "
            "GROUP BY 1 ORDER BY 1",
            (APPAREL_CATEGORY, APPAREL_CATEGORY, APPAREL_CATEGORY, APPAREL_CATEGORY + " > "),
        ).fetchall()
        # Cache only immutable values; caller mutation cannot corrupt counts.
        with _category_cache_lock:
            _category_cache[key] = tuple((row[0], row[1]) for row in rows)
            _category_cache.move_to_end(key)
            while len(_category_cache) > 8:
                _category_cache.popitem(last=False)
    return [{"name": name, "count": count} for name, count in rows]
