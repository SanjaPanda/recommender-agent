"""Check authored interpretation cases against configured local Ollama.

Run ``uv run python scripts/check_search_intent.py --case compound`` from the
repository root. Omit --case for all eight cases (several minutes on CPU).
Requires configured Ollama and installed AGENT_MODEL; reads authored data under
data/ and writes .codex-work/search-intent-checks.json by default. It never
changes catalog products, vectors or sessions. These are structural intent
checks, not held-out human judgments of ranked product relevance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from time import perf_counter

from agent.config import ROOT, settings
from agent.models import SearchIntent
from agent.search_intent import extract_search_intent
from rag.phrases import phrase_pattern


def check_intent(case: dict, intent: SearchIntent) -> dict:
    """Compare compound preservation and required quoted attribute meanings.

    Args:
        case: Authored query, expected product_phrase and required attributes.
        intent: A schema/evidence-validated interpretation returned by runtime.

    Returns:
        Pass status, product match, missing required attributes and extra
        attributes for review. Extra source-supported attributes do not fail a
        case; their semantic precision requires independent review.
    """
    expected = case['product_phrase']
    product_matches = (bool(re.fullmatch(phrase_pattern(expected), intent.product_phrase, re.I))
                       if expected else not intent.product_phrase)
    normalize = lambda value: ' '.join(value.casefold().split())
    key = lambda attr: (normalize(attr['name']), normalize(attr['value']), attr.get('excluded', False))
    required = {key(attr) for attr in case['attributes']}
    actual = {key(attr.model_dump()) for attr in intent.attributes}
    return {'passed': product_matches and required <= actual, 'product_matches': product_matches,
            'missing_attributes': sorted(required - actual), 'extra_attributes': sorted(actual - required)}


def main() -> None:
    """Run selected authored cases and save actual outcomes/timings without retries."""
    cases = [json.loads(line) for line in (ROOT / 'data/search_intent_cases.jsonl').read_text(encoding='utf-8').splitlines() if line]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[case['id'] for case in cases], help='Run one case; default runs all')
    parser.add_argument('--output', type=Path, default=ROOT / '.codex-work/search-intent-checks.json', help='JSON report path')
    args = parser.parse_args()
    report = {'model': settings.agent_model,
              'limits': 'Authored interpretation checks, not judged retrieval quality or latency percentiles.', 'runs': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if args.case and case['id'] != args.case:
            continue
        run = {'id': case['id']}
        started = perf_counter()
        try:
            intent = extract_search_intent(case['query'])
            run.update(check_intent(case, intent), intent=intent.model_dump())
        except Exception as error:
            run.update(passed=False, error_type=type(error).__name__)
        run['seconds'] = round(perf_counter() - started, 3)
        report['runs'].append(run)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({key: value for key, value in run.items() if key != 'intent'}), flush=True)
    if any(not run['passed'] for run in report['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
