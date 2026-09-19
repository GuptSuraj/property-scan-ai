"""Visible observations, rule-backed concealed-risk flags, and unpriced scope."""

from enum import StrEnum

from pydantic import Field, JsonValue

from property_scanner.schemas.base import ConfidenceScore, ContractModel, Identifier, NonEmptyText
from property_scanner.schemas.measurements import AreaMeasurement, LengthMeasurement, QuantityMeasurement
from property_scanner.schemas.primitives import BoundingBox2D, Polygon2D


class DamageType(StrEnum):
    CRACK = "crack"
    WATER_DAMAGE = "water_damage"
    MOISTURE_STAIN = "moisture_stain"
    MOLD = "mold"
    HOLE = "hole"
    PEELING_PAINT = "peeling_paint"
    SURFACE_DAMAGE = "surface_damage"
    UNKNOWN = "unknown"


class DamageSeverity(StrEnum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class DamageRegion(ContractModel):
    """Visible observation; unknown metric size stays null, never zero-filled."""

    damage_id: Identifier
    damage_type: DamageType | NonEmptyText = Field(default=DamageType.UNKNOWN, description="Known category or nonempty custom category.")
    surface_id: Identifier | None = None
    room_id: Identifier | None = None
    polygon_2d: Polygon2D | None = Field(default=None, description="Metric surface-local coordinates; not image pixels. See data_model.md.")
    bounding_box: BoundingBox2D | None = Field(default=None, description="Metric bounds in the same frame as polygon_2d.")
    metric_area: AreaMeasurement | None = None
    metric_length: LengthMeasurement | None = None
    severity: DamageSeverity = DamageSeverity.UNKNOWN
    confidence: ConfidenceScore | None = None
    source_frames: list[NonEmptyText] = Field(default_factory=list, description="Source file/URI/frame references; not opened during validation.")
    notes: NonEmptyText | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ConcealedDamageFlag(ContractModel):
    """Rule-backed suspicion, not a confirmed observation of hidden damage."""

    flag_id: Identifier
    room_id: Identifier | None = None
    surface_id: Identifier | None = None
    suspected_damage_type: DamageType | NonEmptyText = DamageType.UNKNOWN
    rule_id: Identifier = Field(description="Required identifier of the rule that fired.")
    rule_description: NonEmptyText = Field(description="Required explanation of why the rule flags a risk.")
    evidence: list[NonEmptyText] = Field(default_factory=list, description="Descriptions or references to actual supporting observations.")
    confidence: ConfidenceScore | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ScopeLineItem(ContractModel):
    """Repair action with an optional measured quantity and no price fields."""

    line_item_id: Identifier
    room_id: Identifier | None = None
    surface_id: Identifier | None = None
    damage_id: Identifier | None = None
    action: NonEmptyText
    description: NonEmptyText
    quantity: QuantityMeasurement | None = None
    confidence: ConfidenceScore | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
