"""Profile the real local search pipeline without changing application behavior.

Creates isolated profiling sessions. Runs two authored queries, each followed by
an identical-query repeat, sequentially in one fresh Python process. Output has
timings and provider counters, never product records, prompts or credentials.
Run ``uv run python scripts/profile_search_latency.py`` from the repository
root with local services configured as in docs/SETUP.md. Optional ``--output``
changes the default .codex-work/search-latency-profile.json destination.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import urllib.parse
import urllib.request
from uuid import uuid4


class Recorder:
    """Collect nested function timings and permitted local provider counters."""
    def __init__(self):
        """Initialize independent function, HTTP and nesting records."""
        self.events = []
        self.http = []
        self.stack = []

    @contextmanager
    def span(self, name):
        """Record duration/error type for a named context, preserving exceptions."""
        event = {'name': name, 'parent': self.stack[-1] if self.stack else None}
        index = len(self.events)
        self.events.append(event)
        self.stack.append(index)
        started = perf_counter()
        try:
            yield
        except Exception as error:
            event['error_type'] = type(error).__name__
            raise
        finally:
            event['ms'] = round((perf_counter() - started) * 1000, 3)
            self.stack.pop()

    def wrap(self, name, function):
        """Return an argument-preserving timed wrapper around a callable."""
        @wraps(function)
        def timed(*args, **kwargs):
            """Record one invocation and its permitted retrieval metadata."""
            with self.span(name):
                if name == 'bm25_python_rank':
                    self.events[self.stack[-1]]['document_count'] = len(args[1] if len(args) > 1 else kwargs['documents'])
                if name == 'vector_search':
                    self.events[self.stack[-1]]['modality'] = kwargs.get('modality')
                return function(*args, **kwargs)
        return timed

    def urlopen(self, original):
        """Wrap urllib calls, recording counters only for local model endpoints."""
        recorder = self

        def opened(request, *args, **kwargs):
            """Forward the request unchanged and time supported response reads."""
            url = request.full_url if hasattr(request, 'full_url') else request
            endpoint = urllib.parse.urlparse(url).path
            if endpoint not in {'/api/chat', '/api/embed'}:
                return original(request, *args, **kwargs)
            started = perf_counter()
            response = original(request, *args, **kwargs)

            class Response:
                """Delegate context/read behavior while capturing provider counters."""
                def __enter__(self):
                    """Enter the underlying HTTP response context."""
                    response.__enter__()
                    return self

                def __exit__(self, *exception):
                    """Close the underlying response and preserve exception handling."""
                    return response.__exit__(*exception)

                def read(self, *read_args, **read_kwargs):
                    """Return original response bytes and extract permitted JSON counters."""
                    body = response.read(*read_args, **read_kwargs)
                    data = json.loads(body)
                    fields = ('total_duration', 'load_duration', 'prompt_eval_count',
                              'prompt_eval_duration', 'eval_count', 'eval_duration')
                    recorder.http.append({'endpoint': endpoint,
                                          'wall_ms': round((perf_counter() - started) * 1000, 3),
                                          **{key: data[key] for key in fields if key in data}})
                    return body

            return Response()

        return opened


def main() -> None:
    """Run authored live queries and save measured timings to the output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.codex-work/search-latency-profile.json'))
    args = parser.parse_args()
    imported = perf_counter()
    import psycopg
    import agent.api as api
    import agent.workflow as workflow
    import agent.sessions as sessions
    import agent.tools.retrieval as retrieval
    from agent.config import settings

    report = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Sequential real session pipeline and API serialization, fresh profiler process; no forced service/cache reset',
        'application_import_ms': round((perf_counter() - imported) * 1000, 3),
        'settings': {'agent_model': settings.agent_model, 'agent_num_gpu': settings.agent_num_gpu,
                     'embedding_model': settings.embedding_model, 'embedding_num_gpu': settings.embedding_num_gpu,
                     'router_timeout_seconds': settings.agent_router_timeout_seconds},
        'runs': [],
        'limits': ['Two authored queries, not latency percentiles or a relevance benchmark.',
                   'Profiler process has its own initially cold CLIP cache; running API process is separate.',
                   'Host/service/OS cache states are not forced cold. No browser rendering or HTTP client transport measured.',
                   'Nested timings overlap; sum only direct children. Instrumentation has small overhead.',
                   'LLM extraction releases its model after each new query, matching application behavior.',
                   'Unique profiling sessions are persisted; catalog facts and vectors are not modified.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for query in ('blue bow tie', 'black cotton dress'):
        thread_id = 'latency-profile-' + uuid4().hex
        for repeat in (False, True):
            recorder = Recorder()
            run = {'query': query, 'repeat_same_session': repeat}
            started = perf_counter()
            try:
                with ExitStack() as patches:
                    for module, attribute, label in (
                        (workflow, 'extract_search_intent', 'llm_intent'),
                        (workflow, 'list_categories', 'category_lookup'),
                        (workflow, 'retrieve_products', 'hybrid_retrieval'),
                        (retrieval, 'embed_texts', 'nomic_query_embedding'),
                        (retrieval, 'clip_text_embedding', 'clip_query_embedding'),
                        (retrieval, '_search_bm25', 'bm25_corpus_fetch_and_rank'),
                        (retrieval, 'bm25_rank', 'bm25_python_rank'),
                        (retrieval, '_search_embedding', 'vector_search'),
                        (psycopg, 'connect', 'database_connect'),
                        (psycopg.Connection, 'execute', 'database_execute'),
                    ):
                        patches.enter_context(patch.object(module, attribute,
                            recorder.wrap(label, getattr(module, attribute))))
                    patches.enter_context(patch.object(urllib.request, 'urlopen', recorder.urlopen(urllib.request.urlopen)))
                    with recorder.span('session_turn'):
                        state = sessions.run_recommendation_turn(
                                                                thread_id=thread_id, turn={'query': query})
                    with recorder.span('api_serialization'):
                        response = api._response(state, thread_id, 'Profile results', create_surface=True)
                    run.update(result_count=len(response['products']), status=response['status'],
                               trace=[event.model_dump() for event in state.get('trace', [])])
            except Exception as error:
                # Provider exceptions can contain content; record only their class.
                run['error_type'] = type(error).__name__
            run.update(wall_ms=round((perf_counter() - started) * 1000, 3),
                       events=recorder.events, ollama=recorder.http)
            report['runs'].append(run)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({key: value for key, value in run.items()
                              if key not in {'events', 'ollama', 'trace'}}), flush=True)
            if 'error_type' in run:
                # A repeat cannot represent cache reuse after a failed initial search.
                break

    if any('error_type' in run for run in report['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
