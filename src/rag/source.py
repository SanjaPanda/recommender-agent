"""Bounded local source manifests and safe asset access for offline jobs.

Paths resolve inside the source directory; record bytes are returned unchanged so
callers can hash the original input. Import and reporting keep their own policy
for exhausted budgets and product counts.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class Manifest(BaseModel):
    """Versioned local source decision, bounded scan and image inventory.

    Paths are relative to the manifest directory. Image entries are indexed by
    parent_asin and contain local_path and the selected upstream source_url.
    """

    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    source_name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    rights_status: Literal["permitted", "personal_use", "authored"]
    rights_evidence: str = Field(min_length=1)
    records: str
    images: dict[str, dict[str, str]]
    source_lines: dict[str, PositiveInt] = Field(default_factory=dict)
    max_products: int = Field(default=2000, ge=1, le=2000)
    max_lines: int = Field(default=100000, ge=1, le=100000)
    max_bytes: int = Field(default=250_000_000, ge=1, le=250_000_000)
    quality_policy: Literal["evidence-v2"] | None = None
    min_description_words: int = Field(default=50, ge=50)

    @model_validator(mode="after")
    def require_pinned_source(self) -> "Manifest":
        """Reject blank provenance and mutable revision aliases."""
        for value in (self.source_id, self.source_name, self.revision, self.source_url, self.rights_evidence):
            if not value.strip():
                raise ValueError("Source provenance must not be blank")
        if self.revision.lower() in {"main", "master", "latest"}:
            raise ValueError("Use an immutable source revision")
        return self


def local_asset(root: Path, relative: str) -> Path:
    """Resolve a local file within root, rejecting traversal and symlink escapes."""
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root.resolve()):
        raise ValueError("asset_path")
    if not resolved.is_file():
        raise ValueError("missing_asset")
    return resolved


def load_manifest(path: Path) -> Manifest:
    """Validate a local manifest of at most 2 MB before reading source assets."""
    if path.stat().st_size > 2_000_000:
        raise ValueError("manifest_byte_limit")
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def records(manifest_path: Path, manifest: Manifest) -> Iterator[bytes]:
    """Yield original local JSONL bytes within the manifest line/byte budgets.

    Stops at EOF or max_lines. Raises ValueError on a line over 1 MB or input
    exceeding max_bytes. Callers decide how many products to assess and how to
    handle invalid JSON. No line is decoded, reserialized or newline-normalized.
    """
    consumed = 0
    with local_asset(manifest_path.parent, manifest.records).open("rb") as stream:
        for _ in range(manifest.max_lines):
            line = stream.readline(min(1_000_000, manifest.max_bytes - consumed) + 1)
            if not line:
                return
            consumed += len(line)
            if consumed > manifest.max_bytes or len(line) > 1_000_000:
                raise ValueError("Input exceeds manifest budget")
            yield line
