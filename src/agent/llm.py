"""Bounded, validated JSON calls to local Ollama language and vision models."""
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from pydantic import BaseModel


class ModelError(RuntimeError):
    """Local model unavailable or response violates its contract."""


def complete(schema: type[BaseModel], system: str, data: dict, *, image: Path | None = None,
             timeout: float = 120):
    """Make one local model call with no hidden retry or agent loop."""
    if timeout <= 0:
        raise ModelError('Model call time budget exhausted.')
    setting = 'AGENT_VISION_MODEL' if image else 'AGENT_MODEL'
    model = os.environ.get(setting, '').strip()
    if not model:
        raise ModelError(f'Configure {setting} with an installed Ollama model.')
    endpoint = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
    if urllib.parse.urlparse(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ModelError('This agent is configured for local Ollama inference only.')
    message = {'role': 'user', 'content': json.dumps(data, ensure_ascii=True)}
    if image:
        message['images'] = [base64.b64encode(image.read_bytes()).decode('ascii')]
    payload = {
        'model': model, 'stream': False, 'format': schema.model_json_schema(),
        'messages': [{'role': 'system', 'content': system}, message],
        'options': {'temperature': 0, 'num_predict': {'Plan': 640, 'Decision': 160, 'Summary': 256}.get(schema.__name__, 640),
                    'num_ctx': 4096}, 'keep_alive': '1m',
    }
    request = urllib.request.Request(endpoint + '/api/chat', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = json.load(response)['message']['content']
        return schema.model_validate_json(content)
    except (urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError) as exc:
        raise ModelError(f'Local model {model!r} could not produce a valid {schema.__name__} response.') from exc
