"""Validate supplied geometry and translate the whole scene without rescaling it."""
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from property_scanner.schemas.geometry import PropertyGeometry, Room, Wall
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.rendering.exceptions import InvalidGeometryError


@dataclass
class PositionedRoom:
    room: Room
    polygon: Polygon
    walls: list[tuple[Wall, np.ndarray, np.ndarray]]


@dataclass
class RenderScene:
    rooms: list[PositionedRoom]
    origin: np.ndarray
    width: float
    height: float


def build_scene(geometry: PropertyGeometry, config: RenderingConfig) -> RenderScene:
    if not geometry.rooms:
        raise InvalidGeometryError("Floor plan requires at least one room")
    polygons = []
    for room in geometry.rooms:
        if room.polygon is None:
            raise InvalidGeometryError(f"Room {room.room_id} has no room polygon")
        polygon = Polygon([(p.x, p.y) for p in room.polygon.points])
        if polygon.is_empty or not polygon.is_valid:
            raise InvalidGeometryError(f"Room {room.room_id} has an invalid polygon")
        if not room.walls:
            raise InvalidGeometryError(f"Room {room.room_id} has no walls")
        polygons.append(polygon)
    bounds = unary_union(polygons).bounds
    origin = np.array(bounds[:2])
    positioned = []
    for room, polygon in zip(geometry.rooms, polygons):
        local_polygon = Polygon(np.asarray(polygon.exterior.coords)-origin)
        walls, lines = [], []
        for wall in room.walls:
            if wall.start_point is None or wall.end_point is None:
                raise InvalidGeometryError(f"Wall {wall.wall_id} requires both endpoints")
            a = np.array([wall.start_point.x, wall.start_point.y])-origin
            b = np.array([wall.end_point.x, wall.end_point.y])-origin
            if np.linalg.norm(b-a) <= 1e-9:
                raise InvalidGeometryError(f"Wall {wall.wall_id} has coincident endpoints")
            line = LineString([a, b])
            if not local_polygon.boundary.buffer(config.geometry_tolerance).covers(line):
                raise InvalidGeometryError(f"Wall {wall.wall_id} does not follow the supplied polygon")
            walls.append((wall, a, b))
            lines.append(line)
        if not unary_union(lines).buffer(config.geometry_tolerance).covers(local_polygon.boundary):
            raise InvalidGeometryError(f"Room {room.room_id} walls do not cover its polygon boundary")
        positioned.append(PositionedRoom(room, local_polygon, walls))
    return RenderScene(positioned, origin, bounds[2]-bounds[0], bounds[3]-bounds[1])
