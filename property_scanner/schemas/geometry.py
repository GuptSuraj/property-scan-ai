"""Canonical property geometry and room graph; no derived values are computed."""

from enum import StrEnum
from typing import Self

from pydantic import Field, JsonValue, model_validator

from property_scanner.schemas.base import (
    ConfidenceScore, ContractModel, Identifier, NonEmptyText, require_unique,
)
from property_scanner.schemas.measurements import AngleMeasurement, AreaMeasurement, LengthMeasurement
from property_scanner.schemas.primitives import Point2D, Polygon2D


class OpeningType(StrEnum):
    DOOR = "door"
    WINDOW = "window"
    OPEN_PASSAGE = "open_passage"
    UNKNOWN = "unknown"


class ConnectionType(StrEnum):
    DOOR = "door"
    OPEN_PASSAGE = "open_passage"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Wall(ContractModel):
    """Room-owned wall in the shared property XY frame; wall_id is a surface ID."""

    wall_id: Identifier
    room_id: Identifier
    start_point: Point2D | None = None
    end_point: Point2D | None = None
    length: LengthMeasurement | None = None
    height: LengthMeasurement | None = None
    thickness: LengthMeasurement | None = None
    orientation: AngleMeasurement | None = None
    opening_ids: list[Identifier] = Field(default_factory=list)
    damage_ids: list[Identifier] = Field(default_factory=list)
    confidence: ConfidenceScore | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class FloorSurface(ContractModel):
    """Room floor; polygon uses the property XY frame."""

    surface_id: Identifier
    polygon: Polygon2D | None = None
    area: AreaMeasurement | None = None
    material: NonEmptyText | None = None
    damage_ids: list[Identifier] = Field(default_factory=list)
    confidence: ConfidenceScore | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CeilingSurface(ContractModel):
    """Ceiling footprint with explicit height above the associated room floor."""

    surface_id: Identifier
    polygon: Polygon2D | None = None
    height: LengthMeasurement | None = Field(default=None, description="Ceiling height above the room floor, in meters; null if unavailable.")
    area: AreaMeasurement | None = None
    damage_ids: list[Identifier] = Field(default_factory=list)
    confidence: ConfidenceScore | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Opening(ContractModel):
    """Defined once at property level, optionally joining two rooms."""

    opening_id: Identifier
    type: OpeningType = OpeningType.UNKNOWN
    wall_id: Identifier | None = Field(default=None, description="Primary host wall, if identified.")
    room_ids: list[Identifier] = Field(default_factory=list, max_length=2)
    width: LengthMeasurement | None = None
    height: LengthMeasurement | None = None
    sill_height: LengthMeasurement | None = None
    position_along_wall: LengthMeasurement | None = Field(default=None, description="Offset from primary wall start toward its end, in meters.")
    confidence: ConfidenceScore | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_rooms(self) -> Self:
        require_unique(self.room_ids, "opening room IDs")
        return self


class Room(ContractModel):
    """Room with optional geometry; open vocabulary room_type accepts custom labels."""

    room_id: Identifier
    name: NonEmptyText | None = None
    room_type: NonEmptyText = "unknown"
    polygon: Polygon2D | None = None
    floor: FloorSurface | None = None
    ceiling: CeilingSurface | None = None
    walls: list[Wall] = Field(default_factory=list)
    opening_ids: list[Identifier] = Field(default_factory=list)
    damage_ids: list[Identifier] = Field(default_factory=list)
    confidence: ConfidenceScore | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ownership(self) -> Self:
        require_unique([wall.wall_id for wall in self.walls], "wall IDs")
        for wall in self.walls:
            if wall.room_id != self.room_id:
                raise ValueError(f"Wall {wall.wall_id} must belong to room {self.room_id}")
        return self


class RoomConnection(ContractModel):
    """Explicit edge in the property graph, optionally supported by an opening."""

    connection_id: Identifier
    room_a_id: Identifier
    room_b_id: Identifier
    opening_id: Identifier | None = None
    connection_type: ConnectionType = ConnectionType.UNKNOWN
    confidence: ConfidenceScore | None = None

    @model_validator(mode="after")
    def different_rooms(self) -> Self:
        if self.room_a_id == self.room_b_id:
            raise ValueError("A room connection cannot connect a room to itself")
        return self


class BoundingDimensions(ContractModel):
    """Axis-aligned property extents; no dimensions are derived automatically."""

    length: LengthMeasurement | None = None
    width: LengthMeasurement | None = None
    height: LengthMeasurement | None = None


class PropertyGeometry(ContractModel):
    """Stitched property in a local right-handed metric frame with Z up.

    Room floor area lives at floor.area and ceiling height at ceiling.height to
    avoid conflicting copies. Empty collections can represent incomplete work.
    """

    property_id: Identifier
    rooms: list[Room] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    room_connections: list[RoomConnection] = Field(default_factory=list)
    footprint_polygon: Polygon2D | None = None
    total_floor_area: AreaMeasurement | None = None
    bounding_dimensions: BoundingDimensions | None = None
    confidence: ConfidenceScore | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        require_unique([room.room_id for room in self.rooms], "room IDs")
        require_unique([opening.opening_id for opening in self.openings], "opening IDs")
        require_unique([connection.connection_id for connection in self.room_connections], "connection IDs")
        room_ids = {room.room_id for room in self.rooms}
        wall_owners = {wall.wall_id: room.room_id for room in self.rooms for wall in room.walls}
        surface_ids = [wall.wall_id for room in self.rooms for wall in room.walls]
        surface_ids += [surface.surface_id for room in self.rooms for surface in (room.floor, room.ceiling) if surface is not None]
        require_unique(surface_ids, "surface IDs (including walls)")
        openings = {opening.opening_id: opening for opening in self.openings}
        for opening in self.openings:
            if not set(opening.room_ids) <= room_ids:
                raise ValueError(f"Opening {opening.opening_id} references an unknown room")
            if opening.wall_id is not None:
                if opening.wall_id not in wall_owners:
                    raise ValueError(f"Opening {opening.opening_id} references an unknown wall")
                if opening.room_ids and wall_owners[opening.wall_id] not in opening.room_ids:
                    raise ValueError("Opening host wall must belong to one of its rooms")
        for room in self.rooms:
            for owner in (room, *room.walls):
                require_unique(owner.opening_ids, "opening references")
                for opening_id in owner.opening_ids:
                    if opening_id not in openings:
                        raise ValueError(f"Unknown opening reference: {opening_id}")
                    opening = openings[opening_id]
                    if opening.room_ids and room.room_id not in opening.room_ids:
                        raise ValueError("Opening reference conflicts with opening rooms")
                    if isinstance(owner, Wall) and opening.wall_id is not None:
                        # The opposite room may reference its side of a shared opening.
                        if wall_owners[opening.wall_id] == room.room_id and opening.wall_id != owner.wall_id:
                            raise ValueError("Opening reference conflicts with its host wall")
        for connection in self.room_connections:
            if connection.room_a_id not in room_ids or connection.room_b_id not in room_ids:
                raise ValueError("Room connection references an unknown room")
            if connection.opening_id is not None:
                if connection.opening_id not in openings:
                    raise ValueError("Room connection references an unknown opening")
                known_rooms = set(openings[connection.opening_id].room_ids)
                if known_rooms and not known_rooms <= {connection.room_a_id, connection.room_b_id}:
                    raise ValueError("Room connection conflicts with opening rooms")
        return self
