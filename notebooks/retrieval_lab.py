"""Small shared data, scoring and gallery functions for the retrieval notebooks.

Reads PostgreSQL in a read-only repeatable-read transaction; never updates data,
sessions or application vectors. Reuses application encoders, BM25 and RRF. New
experiment embeddings are cached privately in .data/experiments/retrieval/.
Run the notebooks with ``python scripts/run_retrieval_experiments.py`` after
``uv sync --extra image``. See notebooks/README.md for methodology and limits.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import psycopg

from agent.config import ROOT, settings
from rag import APPAREL_CATEGORY
from rag.catalog import local_asset
from agent.tools.embeddings import document_text, embed_texts, model_digest
from agent.tools.clip_runtime import text_embedding, model_version
from agent.tools.retrieval import _fuse_rankings

RESULTS = ROOT / 'observations/experiments/retrieval'
CACHE = ROOT / '.data/experiments/retrieval/embeddings'


def unit_rows(values: np.ndarray) -> np.ndarray:
    """Return normalized finite nonzero rows; reject invalid vectors explicitly."""
    matrix = np.atleast_2d(np.asarray(values, dtype=np.float64))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('Experiment vectors must be finite and nonzero.')
    return matrix / norms


def centroid(values: np.ndarray) -> np.ndarray:
    """Average normalized example vectors and normalize the resulting direction."""
    return unit_rows(unit_rows(values).mean(axis=0))[0]


def load_catalog() -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    """Load the common apparel universe with both ready, single-revision vectors.

    Returns products in stable ID order, normalized text/image matrices and a
    provenance manifest. Fails for incomplete coverage or mixed revisions, rather
    than giving one method a different candidate universe.
    """
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
        connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        boundary = "(p.raw_source->>'ground_truth_category'=%s OR starts_with(p.raw_source->>'ground_truth_category',%s))"
        scope = (APPAREL_CATEGORY, APPAREL_CATEGORY + ' > ')
        total = connection.execute('SELECT count(*) FROM products p WHERE ' + boundary, scope).fetchone()[0]
        rows = connection.execute(
            'SELECT p.id,p.title,p.description,p.features,p.brand,p.image,'
            "p.raw_source->>'ground_truth_category',p.source_record_sha256,p.image_sha256,"
            't.embedding::real[],i.embedding::real[],t.model_version,i.model_version '
            'FROM products p JOIN product_embeddings t ON t.product_id=p.id '
            "AND t.modality='text' AND t.model_name=%s AND t.status='ready' "
            'JOIN product_embeddings i ON i.product_id=p.id '
            "AND i.modality='image' AND i.model_name=%s AND i.status='ready' WHERE "
            + boundary + ' ORDER BY p.id', (settings.embedding_model, settings.image_embedding_model, *scope)).fetchall()
        counts = connection.execute('SELECT (SELECT count(*) FROM products),'
                                    '(SELECT count(*) FROM product_embeddings),'
                                    '(SELECT count(*) FROM agent_sessions)').fetchone()
        revision = connection.execute('SELECT revision FROM catalog_revision WHERE singleton=true').fetchone()[0]
        sources = connection.execute('SELECT id,name,revision FROM dataset_sources ORDER BY id').fetchall()
    if not rows or len(rows) != total or len({row[0] for row in rows}) != len(rows):
        raise ValueError('The experiment requires one ready text/image pair for every apparel product.')
    text_versions, image_versions = {row[11] for row in rows}, {row[12] for row in rows}
    if len(text_versions) != 1 or len(image_versions) != 1 or not all((*text_versions, *image_versions)):
        raise ValueError('Mixed or missing vector revisions: complete the existing embedding build first.')
    keys = ('id', 'title', 'description', 'features', 'brand', 'image', 'category',
            'source_record_sha256', 'image_sha256')
    products = [dict(zip(keys, row[:9])) for row in rows]
    text_raw, image_raw = np.asarray([r[9] for r in rows]), np.asarray([r[10] for r in rows])
    fingerprint = hashlib.sha256(json.dumps(products, sort_keys=True, ensure_ascii=True).encode())
    fingerprint.update(text_raw.tobytes())
    fingerprint.update(image_raw.tobytes())
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'catalog_sha256': fingerprint.hexdigest(),
        'catalog_revision': revision, 'candidate_count': len(products), 'database_counts': list(counts),
        'sources': sources, 'text_model': settings.embedding_model, 'text_revision': next(iter(text_versions)),
        'image_model': settings.image_embedding_model, 'image_revision': next(iter(image_versions)),
        'text_dimensions': text_raw.shape[1], 'image_dimensions': image_raw.shape[1],
        'python': platform.python_version(), 'packages': {name: metadata.version(name)
            for name in ('numpy', 'Pillow', 'torch', 'transformers', 'psycopg', 'pgvector')},
        'cases_sha256': hashlib.sha256((ROOT / 'notebooks/cases.json').read_bytes()).hexdigest(),
        'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'notebooks/retrieval_lab.py', 'src/agent/tools/bm25.py', 'src/rag/embed_text.py',
            'src/agent/tools/clip_runtime.py', 'src/agent/tools/retrieval.py')},
        'parameters': {'top_k': 5, 'rrf_k': 60, 'rrf_base_weights': [0.50, 0.50, 0.50],
                       'rrf_component_pool': 50, 'clip_device': 'cpu',
                       'embedding_num_gpu': settings.embedding_num_gpu, 'encoder_options': 'existing application defaults'},
        'design': 'Exact components, common apparel universe, no router/title gates/relevance thresholds; not production end-to-end.'}
    return products, unit_rows(text_raw), unit_rows(image_raw), manifest


def embeddings(inputs: list[str], kind: str, manifest: dict) -> np.ndarray:
    """Encode exact input strings with application defaults and revision-keyed caching.

    Args:
        inputs: Strings including Nomic task prefixes when kind is text.
        kind: text for Nomic, image for CLIP text-to-image queries.
        manifest: Catalog model revisions, which must match the live local encoders.

    Returns:
        Unit vectors in input order. Cache files contain only experiment inputs/vectors.
    """
    if kind not in {'text', 'image'} or not inputs or any(not value.strip() for value in inputs):
        raise ValueError('Expected nonempty text inputs and a supported embedding kind.')
    revision = manifest[kind + '_revision']
    current = model_digest() if kind == 'text' else model_version()
    if current != revision:
        raise ValueError('Installed encoder revision differs from the stored candidate vectors.')
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = [CACHE / (hashlib.sha256(json.dumps([kind, revision, value]).encode()).hexdigest() + '.json')
             for value in inputs]
    missing = [index for index, path in enumerate(paths) if not path.exists()]
    for start in range(0, len(missing), 4):
        batch = missing[start:start + 4]
        vectors = (embed_texts([inputs[index] for index in batch]) if kind == 'text'
                   else [text_embedding(inputs[index]) for index in batch])
        unit_rows(vectors)
        for index, vector in zip(batch, vectors, strict=True):
            paths[index].write_text(json.dumps({'input': inputs[index], 'revision': revision, 'vector': vector}), encoding='utf-8')
        print(f'{kind}: encoded {min(start + 4, len(missing))}/{len(missing)} uncached inputs', flush=True)
    if (model_digest() if kind == 'text' else model_version()) != revision:
        raise ValueError('Encoder revision changed during the experiment.')
    cached = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if any(item['input'] != value or item['revision'] != revision for item, value in zip(cached, inputs, strict=True)):
        raise ValueError('Experiment embedding cache identity mismatch.')
    result = unit_rows([item['vector'] for item in cached])
    if result.shape[1] != manifest[kind + '_dimensions']:
        raise ValueError('Experiment query/candidate dimensions disagree.')
    return result


def ranked(scores: np.ndarray, products: list[dict], excluded: set[str] | None = None) -> list[tuple[int, float]]:
    """Sort exact scores descending, breaking ties by stable product order; omit seeds."""
    scores = np.asarray(scores)
    if scores.shape != (len(products),) or not np.isfinite(scores).all():
        raise ValueError('One finite score per candidate is required.')
    return [(int(index), float(scores[index])) for index in np.argsort(-scores, kind='stable')
            if products[index]['id'] not in (excluded or set())]


def audit_text_sample(products: list[dict], stored: np.ndarray, seed_ids: list[str], manifest: dict) -> dict:
    """Check six fixed seed vectors against fresh current-format inputs; expose drift."""
    by_id = {p['id']: i for i, p in enumerate(products)}
    selected = [products[by_id[pid]] for pid in seed_ids]
    inputs = [document_text(p['title'], p['description'], p['features']) for p in selected]
    fresh = embeddings(inputs, 'text', manifest)
    cosines = np.sum(fresh * stored[[by_id[pid] for pid in seed_ids]], axis=1)
    result = {'seed_ids': seed_ids, 'cosines': cosines.tolist(), 'match_tolerance': 0.999,
              'matching': int(np.sum(cosines >= 0.999)), 'sample_count': len(seed_ids),
              'all_match': bool(np.all(cosines >= 0.999)), 'input_characters': [len(value) for value in inputs],
              'input_sha256': [hashlib.sha256(value.encode()).hexdigest() for value in inputs]}
    print('Stored/current-format text-vector sample:', json.dumps(result))
    if not result['all_match']:
        print('WARNING: stored text inputs do not all reproduce. Treat full-catalog results as index-snapshot behavior; '
              'use the freshly encoded bounded control for representation comparisons.')
    return result


def exclude_examples(products: list[dict], seed_ids: list[str]) -> set[str]:
    """Exclude every group seed and identical-photo siblings for every example count."""
    selected = [product for product in products if product['id'] in seed_ids]
    if len(selected) != len(set(seed_ids)):
        raise ValueError('A fixed seed is absent from this catalog snapshot.')
    hashes = {product['image_sha256'] for product in selected if product['image_sha256']}
    return {product['id'] for product in products
            if product['id'] in seed_ids or product['image_sha256'] in hashes}


def fuse_components(methods: dict[str, list[tuple[int, float]]], products: list[dict]) -> list[tuple[int, float]]:
    """Reuse production RRF k=60 and equal 0.50 component weights, with top-50 pools."""
    score_keys = {'bm25': 'bm25_score', 'nomic_text': 'text_similarity', 'clip_text': 'image_similarity'}
    lists = []
    for name, ranking in methods.items():
        rows = []
        for index, score in ranking[:50]:
            p = products[index]
            rows.append((p['id'], p['title'], p['description'], p['brand'], p['category'], p['image'], score))
        lists.append((rows, score_keys[name], 0.50))
    fused = _fuse_rankings(lists, title_product=None, wanted_type=None, product_phrase=None, limit=5)
    by_id = {p['id']: index for index, p in enumerate(products)}
    return [(by_id[p['id']], p['rrf_score']) for p in fused]


def evidence(product: dict, case: dict) -> tuple[bool, bool | None]:
    """Return transparent title-family and literal listing-support proxies, not relevance labels."""
    title = product['title'] or ''
    family = bool(re.search(case['family'], title, re.I)) and not (
        case.get('reject') and re.search(case['reject'], title, re.I))
    text = ' '.join([title, product['description'] or '', product['brand'] or '', *(product['features'] or [])])
    supported = bool(family and all(re.search(rule, text, re.I) for rule in case['requirements']))
    return bool(family), None if case['id'] == 'exclude_blue' else supported


def summaries(methods: dict[str, list[tuple[int, float]]], products: list[dict], case: dict) -> list[dict]:
    """Summarize top-five listing proxies and category spread without comparing raw score scales."""
    rows = []
    for method, ranking in methods.items():
        checks = [evidence(products[index], case) for index, _ in ranking[:5]]
        rows.append({'case': case['id'], 'method': method, 'returned': len(checks),
            'family_at_5': sum(check[0] for check in checks) / 5,
            'listing_support_at_5': None if case['id'] == 'exclude_blue' else sum(bool(check[1]) for check in checks) / 5,
            'categories': dict(Counter(products[index]['category'].split(' > ')[-1] for index, _ in ranking[:5]))})
    return rows


def candidate_coverage(products: list[dict], cases: list[dict]) -> dict:
    """Count available title-family/literal supports so an absent target is visible."""
    result = {}
    for case in cases:
        checks = [evidence(product, case) for product in products]
        result[case['id']] = {'title_family_count': sum(check[0] for check in checks),
            'literal_listing_count': None if case['id'] == 'exclude_blue' else sum(bool(check[1]) for check in checks)}
    return result


def visual_review_summary(top5: dict, review: dict) -> list[dict]:
    """Score one explicit photo rubric; preserve unjudged/ambiguous results as bounds."""
    result = []
    for method, products in top5.items():
        supported = unknown = 0
        for product in products[:5]:
            label = review['labels'].get(product['id'], {})
            values = [label.get('mostly_white'), label.get('low_top')]
            if any(value is False for value in values):
                continue
            if all(value is True for value in values):
                supported += 1
            else:
                unknown += 1
        unknown += max(0, 5 - len(products[:5]))
        result.append({'method': method, 'supported': supported, 'unknown': unknown,
                       'support_at_5_bounds': [supported/5, (supported + unknown)/5]})
    return result


def top_results(methods: dict[str, list[tuple[int, float]]], products: list[dict]) -> dict:
    """Export stable IDs, raw method scores and source text for the reviewed top five."""
    return {method: [{'id': products[index]['id'], 'title': products[index]['title'], 'score': score,
                     'description_excerpt': (products[index]['description'] or '')[:220]}
                    for index, score in ranking[:5]] for method, ranking in methods.items()}


def save_gallery(methods: dict[str, list[tuple[int, float]]], products: list[dict], name: str) -> str:
    """Save a contact sheet with five local catalog images per method; never fetch remote images."""
    target = RESULTS / 'figures' / (name + '.jpg')
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = 220, 280
    sheet = Image.new('RGB', (5 * width, len(methods) * height), 'white')
    draw = ImageDraw.Draw(sheet)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    for row, (method, ranking) in enumerate(methods.items()):
        draw.text((5, row * height + 3), method, fill='black', font=font)
        for col, (index, score) in enumerate(ranking[:5]):
            product = products[index]
            meta = product['image'] or {}
            if meta.get('local_path'):
                path = local_asset(ROOT / '.data/catalog', meta['local_path'])
                with Image.open(path) as original:
                    thumb = ImageOps.contain(original.convert('RGB'), (210, 185))
                    sheet.paste(thumb, (col * width + (width - thumb.width)//2, row * height + 24))
            label = f'{col + 1}. {product["id"].split(":")[-1]}  {score:.3f}\n'
            label += '\n'.join(textwrap.wrap(product['title'] or '', 31)[:3])
            draw.multiline_text((col * width + 4, row * height + 212), label, fill='black', font=font)
    sheet.save(target, quality=85)
    return str(target.relative_to(ROOT)).replace('\\', '/')


def save_json(name: str, value: dict) -> None:
    """Persist compact experiment evidence under observations, without application writes."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def print_metrics(rows: list[dict]) -> None:
    """Print comparable rank/listing proxies; cosine scales are deliberately kept separate."""
    print(f'{"case":23} {"method":26} {"family@5":>9} {"listing@5":>10}')
    for row in rows:
        support = row['listing_support_at_5']
        print(f'{row["case"]:23} {row["method"]:26} {row["family_at_5"]:9.2f} '
              f'{"not scored" if support is None else f"{support:.2f}":>10}')
