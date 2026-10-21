"""Validated contracts shared by routing, retrieval and responses."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    """Reject unexpected model-generated fields."""
    model_config = ConfigDict(extra="forbid")


class AgentTraceEvent(BaseModel):
    """Safe summary of one completed LangGraph node, without prompt contents."""

    node: Literal["router", "retrieve", "respond"] = Field(description="LangGraph node that completed.")
    duration_ms: int = Field(ge=0, description="Node wall time in milliseconds.")
    route: str | None = Field(default=None, description="Router branch selected, when applicable.")
    status: str | None = Field(default=None, description="Validated shopping workflow status, when available.")
    result_count: int | None = Field(default=None, ge=0, description="Candidate or recommendation count produced by the node.")
    category_count: int | None = Field(default=None, ge=0, description="Available category count returned by the router node.")
    text_count: int | None = Field(default=None, ge=0, description="Displayed candidates with a product-text score.")
    clip_count: int | None = Field(default=None, ge=0, description="Displayed candidates with a query-to-image CLIP score.")
    example_count: int | None = Field(default=None, ge=0, description="Displayed candidates with a selected-example CLIP score.")


class Filters(Contract):
    """Only catalogue-backed hard SQL constraints."""
    title_product: str | None = Field(default=None, max_length=100, description="Single product type matched against the main product title.")
    title_color: str | None = Field(default=None, max_length=30, description="Single color word matched against the product title.")
    brand: str | None = Field(default=None, max_length=200, description="Exact catalogue brand filter.")
    is_secondhand: bool | None = Field(default=None, description="Whether the user requested a pre-owned item.")
    category_prefixes: list[str] = Field(default_factory=list, max_length=10, description="Apparel taxonomy paths used as inclusive category filters.")


class Preference(Contract):
    """A requirement stated positively, including exclusions such as 'not blue'."""
    requirement: str = Field(min_length=1, max_length=300, description="User-stated product preference or exclusion.")
    strength: Literal["hard", "soft"] = Field(default="hard", description="Hard constraints require exact support; soft preferences guide ranking.")
    visual: bool = Field(default=False, description="Whether the preference depends on visible appearance.")


class Plan(Contract):
    """Complete updated shopping context, not a concatenation of old turns."""
    intent: Literal["recommend", "clarify", "categories", "cart_add", "cart_remove", "cart_show"] = Field(description="Action for the current shopping turn.")
    result_limit: int = Field(default=5, ge=1, le=5, description="Maximum number of catalogue products to return.")
    query: str = Field(default="", max_length=1000, description="Current natural-language product need.")
    filters: Filters = Field(default_factory=Filters, description="Supported apparel-only deterministic filters.")
    preferences: list[Preference] = Field(default_factory=list, max_length=10, description="Preferences preserved across turns.")
    message: str = Field(default="", max_length=1000, description="Clarification or user-facing intent message.")
    product_ids: list[str] = Field(default_factory=list, max_length=20, description="Displayed product IDs for explicit cart actions.")


class PlanEdit(Contract):
    """One explicit change supported by a quote from the latest message."""
    field: Literal['brand', 'is_secondhand', 'category', 'query', 'preference'] = Field(
        description='Shopping-plan field changed by this edit.')
    operation: Literal['add', 'remove', 'replace'] = Field(
        description='Whether to add, remove, or replace the named constraint.')
    target: str = Field(default='', max_length=300, description='For preference removal/replacement, copy the existing requirement exactly.')
    value: str = Field(default='', max_length=1000, description='New value; empty for removal. Condition values: new or used.')
    evidence: str = Field(min_length=1, max_length=1000, description='Exact supporting quote from the latest user message.')
    strength: Literal['hard', 'soft'] = 'hard'
    visual: bool = False


class AddEdit(PlanEdit):
    """A new constraint always needs a supplied, nonblank value."""
    operation: Literal['add']
    value: str = Field(min_length=1, max_length=1000, pattern=r'\S',
                       description='Required new constraint, for example lightweight. Never empty.')


class ReplaceEdit(PlanEdit):
    """Replacing a constraint always supplies its replacement."""
    operation: Literal['replace']
    value: str = Field(min_length=1, max_length=1000, pattern=r'\S',
                       description='Required replacement constraint. Never empty.')


class RemoveEdit(PlanEdit):
    """Removing a constraint cannot introduce a replacement value."""
    operation: Literal['remove']
    value: Literal[''] = ''


class TurnEdits(Contract):
    """Routing decision and edits, never a replacement session snapshot."""
    intent: Literal['recommend', 'clarify', 'categories', 'cart_add', 'cart_remove', 'cart_show'] = Field(
        description='Action for the current shopping turn.')
    edits: list[AddEdit | ReplaceEdit | RemoveEdit] = Field(max_length=20, description='Atomic changes supported by direct user-message evidence.')
    message: str = Field(default='', max_length=1000, description='Clarification text, when needed.')
    product_ids: list[str] = Field(default_factory=list, max_length=20, description='Displayed IDs for an explicit cart action.')


class Summary(Contract):
    """Plain-text explanation; product identity/order stays application-owned."""
    message: str = Field(min_length=1, max_length=2000, description='Grounded user-facing summary text.')


class ProductReference(Contract):
    """Small product reference supplied to a router call."""

    id: str = Field(min_length=1, max_length=128, description="Stable catalogue product identifier.")
    title: str = Field(default="", max_length=500, description="Catalogue product title.")


class ProductFacts(ProductReference):
    """Bounded catalogue facts supplied to a grounded explanation call."""

    description: str = Field(default="", max_length=1200, description="Truncated catalogue description, when available.")
    brand: str = Field(default="", max_length=200, description="Catalogue brand, when available.")


class RouterRequest(Contract):
    """Typed context for interpreting one new user shopping message."""

    previous_plan: Plan = Field(description="Previously validated shopping state.")
    user_message: str = Field(min_length=1, max_length=1000, description='Latest user message to interpret.')
    categories: list[str] = Field(max_length=100, description="Available apparel taxonomy paths.")
    displayed_products: list[ProductReference] = Field(max_length=5, description='Products visible to the user in this session.')
    cart: list[ProductReference] = Field(max_length=20, description='Current cart tracker contents.')


class ExplanationRequest(Contract):
    """Typed input for an optional factual recommendation summary."""

    plan: Plan = Field(description='Validated request and preserved shopping preferences.')
    products: list[ProductFacts] = Field(min_length=1, max_length=5)
