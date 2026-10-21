"""Validated settings shared by the local recommendation application."""
from pathlib import Path

from pydantic import Field, NonNegativeInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and project `.env`.

    Required database connectivity is validated at import/startup. Secrets are
    represented as secret types so accidental model representations redact them.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: SecretStr = Field(description="PostgreSQL connection URL.")
    ollama_url: str = Field(default="http://127.0.0.1:11434", description="Local Ollama HTTP endpoint.")
    agent_model: str = Field(default="gemma3:4b", min_length=1, description="One local Ollama model for routing and optional summaries.")
    embedding_model: str = Field(default="nomic-embed-text", min_length=1, description="Ollama model used for product/query text vectors.")
    image_embedding_model: str = Field(default="openai/clip-vit-base-patch32", min_length=1, description="Cached local CLIP model used for product and query image-space vectors.")
    agent_think: bool = Field(default=False, description="Enable Ollama reasoning mode.")
    agent_router_timeout_seconds: float = Field(
        default=180, gt=0, le=600,
        description="Maximum wait for one local router inference, including cold model load.",
    )
    agent_generate_summaries: bool = Field(
        default=False, description="Whether to make an optional grounded summary call."
    )
    agent_num_gpu: NonNegativeInt | None = Field(
        default=None, description="Optional Ollama GPU-layer count; None uses the provider default."
    )

    @field_validator("ollama_url")
    @classmethod
    def require_local_ollama(cls, value: str) -> str:
        """Reject hosted Ollama endpoints so inference remains local-only."""
        from urllib.parse import urlparse

        if urlparse(value).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama URL must use a loopback host.")
        return value


settings = Settings()
