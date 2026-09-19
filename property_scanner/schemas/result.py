"""Versioned root result and reproducibility information for every input mode."""

from typing import Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from property_scanner.schemas.base import ContractModel, Identifier, NonEmptyText, NonNegativeNumber, require_unique
from property_scanner.schemas.capture import CaptureMetadata
from property_scanner.schemas.damage import ConcealedDamageFlag, DamageRegion, ScopeLineItem
from property_scanner.schemas.geometry import PropertyGeometry

SCHEMA_VERSION = "1.0.0"


class ResultWarning(ContractModel):
    """Actionable diagnostic with optional links to known geometry."""

    code: Identifier
    message: NonEmptyText
    room_id: Identifier | None = None
    surface_id: Identifier | None = None


class ProcessingIssue(ContractModel):
    """Processing error information; does not imply successful execution."""

    code: Identifier
    message: NonEmptyText
    module: NonEmptyText | None = None


class ProcessingInfo(ContractModel):
    """Optional, supplied facts about a run; no timestamps or durations fabricated."""

    pipeline_version: NonEmptyText | None = None
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    processing_seconds: NonNegativeNumber | None = None
    modules_used: list[NonEmptyText] = Field(default_factory=list)
    model_versions: dict[NonEmptyText, NonEmptyText] = Field(default_factory=dict)
    errors: list[ProcessingIssue] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def chronological_timestamps(self) -> Self:
        if self.started_at is not None and self.completed_at is not None:
            if self.completed_at < self.started_at:
                raise ValueError("completed_at must not precede started_at")
        return self


class PropertyScanResult(ContractModel):
    """Official output contract, shared by photos, video, and LiDAR.

    Collections contain only known entities; an empty collection does not prove
    that analysis ran or that a property has no damage. Consult processing_info
    and warnings. Optional values are null when unknown.
    """

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    capture: CaptureMetadata
    property: PropertyGeometry
    damages: list[DamageRegion] = Field(default_factory=list)
    concealed_damage_flags: list[ConcealedDamageFlag] = Field(default_factory=list)
    scope_line_items: list[ScopeLineItem] = Field(default_factory=list)
    warnings: list[ResultWarning] = Field(default_factory=list)
    processing_info: ProcessingInfo | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="Namespaced extension/provenance data, including synthetic fixture labels.")

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        require_unique([item.damage_id for item in self.damages], "damage IDs")
        require_unique([item.flag_id for item in self.concealed_damage_flags], "concealed-damage flag IDs")
        require_unique([item.line_item_id for item in self.scope_line_items], "scope line item IDs")
        rooms = {room.room_id for room in self.property.rooms}
        surfaces: dict[str, str] = {}
        for room in self.property.rooms:
            surfaces.update({wall.wall_id: room.room_id for wall in room.walls})
            for surface in (room.floor, room.ceiling):
                if surface is not None:
                    surfaces[surface.surface_id] = room.room_id

        def check_location(room_id: str | None, surface_id: str | None) -> None:
            if room_id is not None and room_id not in rooms:
                raise ValueError(f"Unknown room reference: {room_id}")
            if surface_id is not None:
                if surface_id not in surfaces:
                    raise ValueError(f"Unknown surface reference: {surface_id}")
                if room_id is not None and surfaces[surface_id] != room_id:
                    raise ValueError("Surface reference does not belong to the referenced room")

        for item in (*self.damages, *self.concealed_damage_flags, *self.scope_line_items, *self.warnings):
            check_location(item.room_id, item.surface_id)
        damages = {damage.damage_id: damage for damage in self.damages}

        def check_damage(damage_id: str, room_id: str | None, surface_id: str | None) -> None:
            if damage_id not in damages:
                raise ValueError(f"Unknown damage reference: {damage_id}")
            damage = damages[damage_id]
            damage_room = damage.room_id or surfaces.get(damage.surface_id or "")
            owner_room = room_id or surfaces.get(surface_id or "")
            if owner_room is not None and damage_room is not None and owner_room != damage_room:
                raise ValueError("Damage reference conflicts with its room")
            if surface_id is not None and damage.surface_id is not None and surface_id != damage.surface_id:
                raise ValueError("Damage reference conflicts with its surface")

        for room in self.property.rooms:
            require_unique(room.damage_ids, "room damage references")
            for damage_id in room.damage_ids:
                check_damage(damage_id, room.room_id, None)
            for surface in (*room.walls, room.floor, room.ceiling):
                if surface is None:
                    continue
                surface_id = surface.wall_id if hasattr(surface, "wall_id") else surface.surface_id
                require_unique(surface.damage_ids, "surface damage references")
                for damage_id in surface.damage_ids:
                    check_damage(damage_id, room.room_id, surface_id)
        for item in self.scope_line_items:
            if item.damage_id is not None:
                check_damage(item.damage_id, item.room_id, item.surface_id)
        return self
