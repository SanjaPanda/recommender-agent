"""Execute the two plain-Python retrieval notebooks without a Jupyter installation.

From the repository root, after ``uv sync --extra image``, run:
``.venv/Scripts/python.exe scripts/run_retrieval_experiments.py``.
Optional positional notebook filenames select a subset, for example
``... scripts/run_retrieval_experiments.py 01_text_queries.ipynb``.
Requires the existing catalog/vectors, local Ollama Nomic and cached CLIP weights.

Cells use ordinary Python only: no magics, rich display or kernel-specific state.
This runner executes them in order in a shared namespace, captures stdout/errors,
and saves real cell outputs in the notebooks. It is not a Jupyter kernel test.
Figures/results go to observations/experiments/retrieval and private embedding
caches to .data/experiments/retrieval. Database access is read-only.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('01_text_queries.ipynb', '02_image_examples.ipynb')


def main() -> None:
    """Run selected notebooks, saving executed cells and stopping on the first failure."""
    sys.stdout.reconfigure(encoding='utf-8')  # Keep redirected Windows logs consistent with notebook JSON.
    names = sys.argv[1:] or NAMES
    if any(name not in NAMES for name in names):
        raise SystemExit('Choose filenames from: ' + ', '.join(NAMES))
    for name in names:
        path = ROOT / 'notebooks' / name
        notebook = json.loads(path.read_text(encoding='utf-8'))
        namespace = {'__name__': '__main__'}
        count = 0
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                cell.update(outputs=[], execution_count=None)
        notebook['metadata']['execution'] = {'runner': 'stdlib sequential Python',
            'jupyter_kernel_verified': False, 'status': 'running', 'utc': datetime.now(timezone.utc).isoformat()}
        try:
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                count += 1
                print(f'{name}: executing cell {count}', flush=True)
                output = io.StringIO()
                cell['execution_count'] = count
                try:
                    with redirect_stdout(output):
                        source = cell['source']
                        exec(compile(''.join(source) if isinstance(source, list) else source,
                                     str(path) + '#' + cell['id'], 'exec'), namespace)
                except Exception as exc:
                    cell['outputs'].append({'output_type': 'error', 'ename': type(exc).__name__,
                        'evalue': str(exc), 'traceback': traceback.format_exc().splitlines()})
                    raise
                finally:
                    if output.getvalue():
                        cell['outputs'].insert(0, {'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue()})
                        print(output.getvalue(), end='', flush=True)
            notebook['metadata']['execution']['status'] = 'passed'
            print(f'{name}: PASS ({count} executed Python cells)', flush=True)
        except Exception:
            notebook['metadata']['execution']['status'] = 'failed'
            raise
        finally:
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
