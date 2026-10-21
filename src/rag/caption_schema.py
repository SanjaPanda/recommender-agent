"""Pydantic v2 schema for offline apparel image captions."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
ShortList = Annotated[list[ShortText], Field(min_length=1, max_length=12)]


class VisualAttributes(BaseModel):
    """Compact visible apparel attributes returned by a caption model.

    Every value is an image observation and is treated as unverified. Missing or
    unclear observations use null rather than inferred product attributes.
    """

    model_config = ConfigDict(extra="forbid")

    product_type: ShortText | None = Field(description="Visible apparel item type, if identifiable.")
    colours: ShortList | None = Field(description="Visible product colors ordered by coverage; null if unclear.")
    pattern: ShortText | None = Field(description="Pattern on the apparel item, when visible.")
    visible_components: ShortList | None = Field(description="Visible parts of the item, if useful.")
    visible_object_count: Annotated[int, Field(strict=True, ge=1)] | None = Field(
        description="Count of visible main product objects; not a sale quantity.")
    other_visual_details: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1, max_length=240)] | None = Field(
            description="One short useful visual detail, if present.")
