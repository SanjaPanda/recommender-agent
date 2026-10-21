"""Bounded, validated JSON calls to local Ollama language models."""
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .config import settings

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ModelError(RuntimeError):
    """Local model unavailable or response violates its contract."""


def complete(
    schema: type[ResponseT],
    system: str,
    data: BaseModel,
    *,
    timeout: float | None = None,
) -> ResponseT:
    """Make one local model call with no hidden retry or agent loop."""
    if timeout is None:
        timeout = settings.agent_router_timeout_seconds if schema.__name__ in {'Plan', 'TurnEdits'} else 120
    if not math.isfinite(timeout) or timeout <= 0:
        raise ModelError('Model call time budget exhausted.')
    model = settings.agent_model.strip()
    endpoint = settings.ollama_url.rstrip('/')
    if urllib.parse.urlparse(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ModelError('This agent is configured for local Ollama inference only.')
    message = {'role': 'user', 'content': data.model_dump_json()}
    payload = {
        'model': model, 'think': settings.agent_think, 'stream': False, 'format': schema.model_json_schema(),
        'messages': [{'role': 'system', 'content': system}, message],
        'options': {'temperature': 0, 'num_predict': {'Plan': 640, 'Summary': 256}.get(schema.__name__, 640),
                    'num_ctx': 4096}, 'keep_alive': '1m',
    }
    # Some small GPUs make partial offload slower than CPU-only inference.
    if settings.agent_num_gpu is not None:
        payload['options']['num_gpu'] = settings.agent_num_gpu
    request = urllib.request.Request(endpoint + '/api/chat', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = json.load(response)['message']['content']
        return schema.model_validate_json(content)
    except ValidationError as exc:
        invalid_fields = ", ".join(
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors(include_input=False)
        )
        raise ModelError(
            f"Local model {model!r} returned invalid {schema.__name__} fields: {invalid_fields}."
        ) from exc
    except TimeoutError as exc:
        raise ModelError(f'Local model {model!r} timed out after {timeout:g} seconds producing {schema.__name__}.') from exc
    except (urllib.error.URLError, KeyError, TypeError, ValueError) as exc:
        raise ModelError(f'Local model {model!r} could not produce a valid {schema.__name__} response.') from exc
