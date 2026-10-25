"""Validated local Nomic encoding shared by retrieval and offline vector jobs.

No database writes or model downloads occur here. ``embed_texts`` calls the
configured local Ollama endpoint; ``model_digest`` identifies its installed tag.
Run focused checks with ``.venv/Scripts/python.exe -m unittest discover -s tests
-p test_embeddings.py -v`` from the repository root.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError, field_validator
from agent.config import settings

OLLAMA_URL = settings.ollama_url
MODEL = settings.embedding_model


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one bounded JSON request to the configured local Ollama endpoint.

    Args:
        path: Ollama API path, such as /api/embed.
        payload: JSON-compatible request fields, without hosted credentials.

    Returns:
        Decoded provider response; embed_texts validates the embedding contract.

    Raises:
        RuntimeError: If connection or the 300-second socket timeout fails.
        ValueError: If the response body is not valid JSON.
    """
    request = urllib.request.Request(
        f"{OLLAMA_URL.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL}: {exc}") from exc


class EmbeddingOptions(BaseModel):
    """Optional local hardware override; provider defaults otherwise apply."""
    model_config = ConfigDict(extra="forbid")
    num_gpu: int = Field(ge=0)


class EmbeddingRequest(BaseModel):
    """Typed Ollama text-embedding request."""

    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    input: str | list[str] = Field(description="One query/document or a batch of texts.")
    options: EmbeddingOptions | None = None


class EmbeddingResponse(BaseModel):
    """Validated finite vectors returned by Ollama's embedding endpoint."""

    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    embeddings: list[Annotated[list[FiniteFloat], Field(min_length=1)]] = Field(min_length=1)
    total_duration: int | None = Field(default=None, ge=0)
    load_duration: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    eval_duration: int | None = Field(default=None, ge=0)

    @field_validator("embeddings")
    @classmethod
    def finite_nonzero_consistent_vectors(
        cls, vectors: list[list[float]]
    ) -> list[list[float]]:
        """Require equal-sized nonzero vectors before retrieval or storage."""
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or any(not any(value != 0 for value in vector) for vector in vectors):
            raise ValueError("Embedding vectors must share one dimension and be nonzero.")
        return vectors


def embed_texts(inputs: str | list[str]) -> list[list[float]]:
    """Embed one or more strings with the configured local Nomic model.

    Args:
        inputs: Non-empty search query or batch of product texts.

    Returns:
        Finite embedding vectors in input order.

    Raises:
        ValueError: If no text is provided.
        RuntimeError: If Ollama returns a malformed response or wrong vector count.
    """
    values = [inputs] if isinstance(inputs, str) else inputs
    if not values or any(not value.strip() for value in values):
        raise ValueError("Embedding input must contain non-empty text.")
    request = EmbeddingRequest(model=MODEL, input=inputs, options=(
        EmbeddingOptions(num_gpu=settings.embedding_num_gpu) if settings.embedding_num_gpu is not None else None))
    try:
        response = EmbeddingResponse.model_validate(
            post_json("/api/embed", request.model_dump(exclude_none=True))
        )
    except ValidationError as exc:
        locations = ", ".join(
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors(include_input=False)
        )
        raise RuntimeError(f"Ollama embedding response failed validation at: {locations}.") from exc
    if len(response.embeddings) != len(values):
        raise RuntimeError(
            f"Ollama returned {len(response.embeddings)} vectors for {len(values)} inputs."
        )
    return response.embeddings


def model_digest() -> str:
    """Resolve the exact installed Ollama tag to its immutable digest.

    Untagged names select ``:latest``. Explicit tags and registry host ports
    are preserved. Raises RuntimeError for invalid metadata and SystemExit if
    the configured tag is absent. Performs one local metadata request.
    """
    request = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            tags = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not read Ollama model metadata: {exc}") from exc
    name = MODEL if ':' in MODEL.rsplit('/', 1)[-1] else MODEL + ':latest'
    for model in tags.get("models", []):
        installed = model.get('name', '')
        installed = installed if ':' in installed.rsplit('/', 1)[-1] else installed + ':latest'
        if installed == name:
            digest = model.get('digest')
            if not isinstance(digest, str) or not digest:
                raise RuntimeError('Installed embedding model has no revision digest.')
            return digest
    raise SystemExit(f"Model {MODEL!r} is not installed. Run: ollama pull {MODEL}")


def document_text(title: str | None, description: str | None, features: list[str] | None = None) -> str:
    """Format preserved listing text using Nomic's document task prefix.

    Args:
        title: Listing title, or None when absent.
        description: Source description, without generated substitute facts.
        features: Optional source feature bullets; empty strings are skipped.

    Returns:
        ``search_document: `` followed by nonblank fields separated by newlines.

    Example:
        >>> document_text('Dress', None, ['Cotton'])
        'search_document: Dress\\nCotton'
    """
    parts = [part.strip() for part in (title, description) if part and part.strip()]
    parts.extend(feature.strip() for feature in (features or []) if isinstance(feature, str) and feature.strip())
    return "search_document: " + "\n".join(parts)


