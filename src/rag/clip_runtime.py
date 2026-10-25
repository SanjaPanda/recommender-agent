"""Cached CPU text encoder for queries against the persistent CLIP image index."""
from __future__ import annotations

from functools import lru_cache
import math

from agent.config import settings


@lru_cache(maxsize=1)
def _clip_encoder(model_name: str):
    """Load one local CLIP model and processor for this application process.

    Downloads are disabled in request handling. Install the configured model
    before startup when it is not already in the Hugging Face cache.
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("Install the optional image dependencies with `uv sync --extra image`.") from exc
    try:
        model = CLIPModel.from_pretrained(model_name, local_files_only=True).to("cpu").eval()
        revision = getattr(model.config, '_commit_hash', None)
        if not revision:
            raise RuntimeError('Cached CLIP model has no immutable revision; reinstall its snapshot.')
        processor = CLIPProcessor.from_pretrained(model_name, revision=revision,
                                                 local_files_only=True, use_fast=False)
    except OSError as exc:
        raise RuntimeError(
            f"CLIP model {model_name!r} is not cached locally; install it before using hybrid retrieval."
        ) from exc
    return torch, model, processor


def text_embedding(query: str) -> list[float]:
    """Encode a natural-language query in the image index's CLIP space.

    Args:
        query: Non-empty user request or preference refinement.

    Returns:
        Unit-normalized 512-dimensional CLIP text features.

    Raises:
        ValueError: If the query is blank or the generated vector is invalid.
        RuntimeError: If optional dependencies or locally cached model weights are absent.
    """
    if not query.strip():
        raise ValueError("CLIP query must not be empty.")
    torch, model, processor = _clip_encoder(settings.image_embedding_model)
    inputs = processor(text=[query.strip()], return_tensors="pt", padding=True,
                       truncation=True, max_length=model.config.text_config.max_position_embeddings)
    with torch.inference_mode():
        features = model.get_text_features(**inputs)[0].cpu().tolist()
    norm = math.sqrt(sum(float(value) * float(value) for value in features))
    if not math.isfinite(norm) or norm == 0 or not all(math.isfinite(float(v)) for v in features):
        raise ValueError("CLIP returned an invalid query vector.")
    return [float(value) / norm for value in features]


def model_version() -> str:
    """Return the revision of the cached CLIP encoder used by query inference.

    Loads the configured local encoder if necessary; never downloads weights.
    Raises RuntimeError when immutable revision metadata is unavailable.
    """
    _, model, _ = _clip_encoder(settings.image_embedding_model)
    revision = getattr(model.config, '_commit_hash', None)
    if not revision:
        raise RuntimeError('Cached CLIP model has no immutable revision; reinstall its snapshot.')
    return revision
