"""Validated contracts shared by routing, retrieval and responses."""
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from rag import validate_category_prefixes

CategoryPrefixes = Annotated[list[str], Field(max_length=10), AfterValidator(validate_category_prefixes)]
FilterName = Literal['brand', 'category_prefixes', 'title_color', 'title_product']


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
    category_prefixes: CategoryPrefixes = Field(
        default_factory=list,
        description="Inclusive paths within Apparel & Accessories; empty means the full catalog.")


class Preference(Contract):
    """A requirement stated positively, including exclusions such as 'not blue'."""
    requirement: str = Field(min_length=1, max_length=300, description="User-stated product preference or exclusion.")
    strength: Literal["hard", "soft"] = Field(default="hard", description="Hard constraints require exact support; soft preferences guide ranking.")
    visual: bool = Field(default=False, description="Whether the preference depends on visible appearance.")


class SearchAttribute(Contract):
    """A user-stated attribute, never an inferred catalogue fact."""
    name: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=100)
    evidence: str = Field(min_length=1, max_length=300)
    excluded: bool = False


class SearchIntent(Contract):
    """Open-vocabulary interpretation of a request, separate from SQL filters."""
    product_phrase: str = Field(default="", max_length=100, description="Exact core product noun phrase from the request. Preserve defining modifiers. Empty when unclear or generic.")
    search_terms: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=4, description="Short equivalent search phrases, preserving the specific product and all exclusions. No broader substitutes.")
    attributes: list[SearchAttribute] = Field(default_factory=list, max_length=10)


class SearchIntentRequest(Contract):
    """Bounded effective query sent to local structured intent extraction."""
    query: str = Field(min_length=1, max_length=5000)


class Plan(Contract):
    """Complete updated shopping context, not a concatenation of old turns."""
    intent: Literal["recommend", "clarify", "categories"] = Field(description="Action for the current shopping turn.")
    result_limit: int = Field(default=5, ge=1, le=5, description="Maximum number of catalogue products to return.")
    query: str = Field(default="", max_length=1000, description="Current natural-language product need.")
    filters: Filters = Field(default_factory=Filters, description="Supported apparel-only deterministic filters.")
    preferences: list[Preference] = Field(default_factory=list, max_length=10, description="Preferences preserved across turns.")
    message: str = Field(default="", max_length=1000, description="Clarification or user-facing intent message.")

    @property
    def effective_query(self) -> str:
        """Combine the product need and preserved preferences for interpretation/retrieval."""
        return ' '.join(part for part in [self.query, *(p.requirement for p in self.preferences)] if part)

    @model_validator(mode='after')
    def validate_preferences(self) -> 'Plan':
        """Reject duplicate or directly contradictory saved requirements."""
        requirements = [' '.join(p.requirement.casefold().split()) for p in self.preferences]
        if len(requirements) != len(set(requirements)):
            raise ValueError('Preferences must be unique.')
        for requirement in requirements:
            for prefix in ('not ', 'no ', 'without '):
                if requirement.startswith(prefix) and requirement[len(prefix):] in requirements:
                    raise ValueError('Preferences cannot require and exclude the same attribute.')
        return self


class RouterDecision(Plan):
    """Updated shopping state and removal flags returned by one router call."""
    removed_filters: list[FilterName] = Field(default_factory=list, max_length=4,
        description='Names of previously active filters explicitly cleared this turn.')
    search_intent: SearchIntent | None = Field(default=None,
        description='Grounded interpretation of the effective query plus preferences, for textual recommendations.')

    @model_validator(mode='after')
    def validate_decision(self, info: ValidationInfo) -> 'RouterDecision':
        """Check state consistency and interpretation evidence before any state update."""
        from .search_intent import validate_search_intent
        from .llm import ModelError
        previous = (info.context or {}).get('previous_plan')
        current_filters = self.filters.model_dump()
        if len(self.removed_filters) != len(set(self.removed_filters)):
            raise ValueError('Removed filter flags must be unique.')
        if any(current_filters[name] for name in self.removed_filters):
            raise ValueError('A removed filter must be empty in the updated state.')
        if previous is not None:
            removed = {name for name, value in previous.filters.model_dump().items()
                       if value and not current_filters[name]}
            if removed != set(self.removed_filters):
                raise ValueError('Removed filter flags must match the change from the previous plan.')
            if self.intent != 'recommend' and any(
                getattr(self, name) != getattr(previous, name) for name in ('query', 'filters', 'preferences')
            ):
                raise ValueError('Clarification and category actions must preserve shopping constraints.')
        query = self.effective_query
        if self.intent == 'recommend' and query:
            if self.search_intent is None:
                raise ValueError('Textual recommendations require a search interpretation.')
            try:
                validate_search_intent(query, self.search_intent)
            except ModelError as exc:
                raise ValueError(str(exc)) from exc
        return self


class SessionTurn(Contract):
    """Validated user text or explicit UI/terminal action for one session turn."""
    query: str = Field(default='', max_length=1000)
    refinement: str = Field(default='', max_length=1000)
    category_prefixes: CategoryPrefixes | None = None
    selected_product_ids: list[str] | None = Field(default=None, max_length=3)
    action: Literal['cart_add', 'cart_remove', 'cart_show'] | None = None
    product_ids: list[str] = Field(default_factory=list, max_length=20)
    reset_context: bool = False

    @field_validator('selected_product_ids')
    @classmethod
    def unique_examples(cls, value: list[str] | None) -> list[str] | None:
        """Reject duplicate visual examples before invoking retrieval."""
        if value is not None and len(value) != len(set(value)):
            raise ValueError('Select up to three unique products.')
        return value


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
    is_new_session: bool = Field(description='Whether this session has no prior user turns.')
    user_message: str = Field(min_length=1, max_length=1000, description='Latest user message to interpret.')
    categories: list[str] = Field(max_length=100, description="Available apparel taxonomy paths.")
    displayed_products: list[ProductReference] = Field(max_length=5, description='Products visible to the user in this session.')


class ExplanationRequest(Contract):
    """Typed input for an optional factual recommendation summary."""

    plan: Plan = Field(description='Validated request and preserved shopping preferences.')
    products: list[ProductFacts] = Field(min_length=1, max_length=5)
