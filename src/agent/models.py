"""Validated contracts shared by routing, retrieval and verification."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    """Reject unexpected model-generated fields."""
    model_config = ConfigDict(extra="forbid")


class Filters(Contract):
    """Only catalogue-backed hard SQL constraints."""
    brand: str | None = Field(default=None, max_length=200)
    is_secondhand: bool | None = None
    category_prefixes: list[str] = Field(default_factory=list, max_length=10)


class Preference(Contract):
    """A requirement stated positively, including exclusions such as 'not blue'."""
    requirement: str = Field(min_length=1, max_length=300)
    strength: Literal["hard", "soft"] = "hard"
    visual: bool = False


class Plan(Contract):
    """Complete updated shopping context, not a concatenation of old turns."""
    intent: Literal["recommend", "clarify", "categories", "cart_add", "cart_remove", "cart_show"]
    query: str = Field(default="", max_length=1000)
    filters: Filters = Field(default_factory=Filters)
    preferences: list[Preference] = Field(default_factory=list, max_length=10)
    message: str = Field(default="", max_length=1000)
    product_ids: list[str] = Field(default_factory=list, max_length=20)


class Decision(Contract):
    """One evidence-based decision for one preference."""
    verdict: Literal["matches", "violates", "uncertain"]
    evidence: str = Field(min_length=1, max_length=600)


class Summary(Contract):
    """Plain-text explanation; product identity/order stays application-owned."""
    message: str = Field(min_length=1, max_length=2000)
