"""Evaluate labeled live refinements without retrieval or session writes.

Use --list, --case NAME (repeatable), or --all. Defaults to the original case.
Requires PostgreSQL category metadata and Ollama.
"""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from agent.models import Plan
from agent.config import settings
from agent import workflow

ROOT = Path(__file__).resolve().parents[1]


def shopping(plan):
    """Compare persistent constraints, excluding response metadata."""
    validated = Plan.model_validate(plan).model_dump()
    result = {k: validated[k] for k in ('query', 'filters', 'preferences')}
    if result['filters']['brand'] is not None:
        result['filters']['brand'] = result['filters']['brand'].casefold()
    return result


def main():
    """Run selected labeled user-turn cases against the configured local router.

    Run ``.venv/Scripts/python.exe scripts/check_router_edits.py --list`` to list
    cases, or repeat ``--case NAME`` to evaluate them without retrieval writes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--case', action='append', dest='cases')
    group.add_argument('--all', action='store_true')
    group.add_argument('--list', action='store_true')
    parser.add_argument('--model', help='Override AGENT_MODEL for this evaluation only')
    parser.add_argument('--output', type=Path, default=ROOT / '.data/router-evaluation.json')
    args = parser.parse_args()
    if args.model:
        settings.agent_model = args.model
    cases = [json.loads(line) for line in (ROOT / 'data/agent_routing_cases.jsonl').read_text().splitlines() if line.strip()]
    known = {case['id'] for case in cases}
    if args.list:
        print('\n'.join(case['id'] for case in cases))
        return
    requested = known if args.all else set(args.cases or ['brand_and_permission'])
    if requested - known:
        parser.error('Unknown cases: ' + ', '.join(sorted(requested - known)))
    report = {'model': settings.agent_model, 'think': settings.agent_think,
              'num_gpu': settings.agent_num_gpu if settings.agent_num_gpu is not None else 'automatic',
              'router_timeout_seconds': settings.agent_router_timeout_seconds, 'results': []}
    try:
        for case in cases:
            if case['id'] not in requested:
                continue
            previous = Plan.model_validate(case['previous']).model_dump()
            expected = shopping(case['expected'])
            started = time.monotonic()
            result = {'id': case['id']}
            try:
                real_complete = workflow.complete

                def observe(*positional, **keywords):
                    output = real_complete(*positional, **keywords)
                    result['model_output'] = output.model_dump()
                    return output

                # Observe real responses; no canned model output is supplied.
                with patch('agent.workflow.complete', side_effect=observe):
                    state = workflow._route({'plan': previous, 'turn': {'query': case['message']}})
                result.update(status=state['status'], message=state['message'], actual=shopping(state['plan']), expected=expected)
                result['passed'] = state['status'] == case['expected_status'] and result['actual'] == expected
                if case['expected_status'] == 'clarify':
                    result['passed'] = result['passed'] and state['route'] == 'respond'
            except Exception as exc:
                result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
            result['seconds'] = round(time.monotonic() - started, 2)
            report['results'].append(result)
            print(json.dumps(result), flush=True)
    finally:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    passed = sum(result['passed'] for result in report['results'])
    print(f'{passed}/{len(report["results"])} live cases passed.')
    if passed != len(requested):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
