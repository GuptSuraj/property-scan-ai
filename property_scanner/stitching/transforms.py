"""SE(2) utilities and lossless schema geometry transformation."""

import math
import numpy as np

from property_scanner.schemas.geometry import Room
from property_scanner.schemas.primitives import Point2D, Polygon2D


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def se2(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def components(matrix: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(matrix, dtype=float)
    return float(matrix[0, 2]), float(matrix[1, 2]), wrap_angle(math.atan2(matrix[1, 0], matrix[0, 0]))


def validate_se2(matrix) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Room transform must be a finite 3x3 matrix")
    if not np.allclose(value[2], [0, 0, 1], atol=1e-6):
        raise ValueError("Room transform has an invalid homogeneous row")
    if not np.allclose(value[:2, :2].T @ value[:2, :2], np.eye(2), atol=1e-4) or not np.isclose(np.linalg.det(value[:2, :2]), 1, atol=1e-4):
        raise ValueError("Room transform must contain rotation without scale")
    return value


def transform_xy(point: Point2D, matrix: np.ndarray) -> Point2D:
    output = matrix @ np.array([point.x, point.y, 1.0])
    return Point2D(x=float(output[0]), y=float(output[1]))


def transform_polygon(polygon: Polygon2D | None, matrix: np.ndarray) -> Polygon2D | None:
    return None if polygon is None else Polygon2D(points=[transform_xy(point, matrix) for point in polygon.points])


def transform_room(room: Room, matrix: np.ndarray) -> Room:
    """Copy and rigidly place a room while retaining all supplied measurements."""
    matrix = validate_se2(matrix)
    placed = room.model_copy(deep=True)
    placed.polygon = transform_polygon(placed.polygon, matrix)
    if placed.floor:
        placed.floor.polygon = transform_polygon(placed.floor.polygon, matrix)
    if placed.ceiling:
        placed.ceiling.polygon = transform_polygon(placed.ceiling.polygon, matrix)
    yaw = math.degrees(components(matrix)[2])
    for wall in placed.walls:
        if wall.start_point:
            wall.start_point = transform_xy(wall.start_point, matrix)
        if wall.end_point:
            wall.end_point = transform_xy(wall.end_point, matrix)
        if wall.orientation:
            wall.orientation.value = (wall.orientation.value + yaw) % 360
    placed.metadata.update({"coordinate_scope": "property_shared", "room_to_property_transform": matrix.tolist()})
    return placed

