"""Generate revisioned Nomic product vectors through local Ollama.

Run ``uv run python -m rag.embed_text --help`` from the repository root.
Normal execution reads PostgreSQL products, calls Ollama and writes vectors in
batches. Configure services and private assets as documented in docs/SETUP.md.
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
import json
from typing import Annotated, Any

import psycopg
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError, field_validator
from pgvector.psycopg import register_vector
from pgvector import Vector

from agent.config import ROOT, settings
from . import CATALOG_WRITE_LOCK

OLLAMA_URL = settings.ollama_url
MODEL = settings.embedding_model


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one JSON request to the configured local Ollama endpoint."""
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
    """Format title, source description and bullet features for Nomic retrieval."""
    parts = [part.strip() for part in (title, description) if part and part.strip()]
    parts.extend(feature.strip() for feature in (features or []) if isinstance(feature, str) and feature.strip())
    return "search_document: " + "\n".join(parts)


def main() -> None:
    """Embed catalogue product text and upsert model-versioned vectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum products to process")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Recompute all product embeddings")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.batch_size < 1 or args.batch_size > 32:
        parser.error("--batch-size must be between 1 and 32")

    database_url = settings.database_url.get_secret_value()
    digest = model_digest()
    limit_sql = " LIMIT %s" if args.limit is not None else ""
    existing_filter = "" if args.force else """
        AND NOT EXISTS (
            SELECT 1 FROM product_embeddings e
            WHERE e.product_id = p.id AND e.modality = 'text'
              AND e.model_name = %s AND e.model_version = %s AND e.status = 'ready'
        )
    """
    select_params: list[Any] = []
    if not args.force:
        select_params.extend([MODEL, digest])
    if args.limit is not None:
        select_params.append(args.limit)
    query = """
        SELECT p.id, p.title, p.description, p.features
        FROM products p
        WHERE (NULLIF(trim(p.title), '') IS NOT NULL
               OR NULLIF(trim(p.description), '') IS NOT NULL)
    """ + existing_filter + " ORDER BY p.id" + limit_sql

    started = time.perf_counter()
    processed = 0
    dimensions: int | None = None
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(%s)", (CATALOG_WRITE_LOCK,))
        register_vector(connection)
        products = connection.execute(query, select_params).fetchall()
        if not products:
            print(f"No products need embeddings for {MODEL} ({digest or 'unknown version'}).")
            return

        for offset in range(0, len(products), args.batch_size):
            batch = products[offset : offset + args.batch_size]
            texts = [document_text(title, description, features) for _, title, description, features in batch]
            vectors = embed_texts(texts)
            for vector in vectors:
                if dimensions is None:
                    dimensions = len(vector)
                elif len(vector) != dimensions:
                    raise RuntimeError("Ollama returned inconsistent embedding dimensions.")
            with connection.transaction():
                for (product_id, _, _, _), vector in zip(batch, vectors, strict=True):
                    connection.execute(
                        """
                        INSERT INTO product_embeddings
                            (product_id, modality, model_name, model_version, embedding, status, error)
                        VALUES (%s, 'text', %s, %s, %s, 'ready', NULL)
                        ON CONFLICT (product_id, modality, model_name) DO UPDATE SET
                            model_version = EXCLUDED.model_version,
                            embedding = EXCLUDED.embedding,
                            status = 'ready', error = NULL, updated_at = now()
                        """,
                        (product_id, MODEL, digest, Vector(vector)),
                    )
            processed += len(batch)
            print(f"Embedded {processed}/{len(products)} products")

    elapsed = time.perf_counter() - started
    print(
        f"Stored {processed} {MODEL} embeddings ({dimensions} dimensions, "
        f"model digest {digest or 'unknown'}) in {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
