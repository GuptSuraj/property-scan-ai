"""Bounded wall intersections and conservative graph-based polygon closure."""
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union
from shapely.geometry.polygon import orient
from property_scanner.geometry.models import RoomCorner, WallSegment2D
from property_scanner.geometry.config import GeometryConfig
from property_scanner.schemas.primitives import Polygon2D
from property_scanner.schemas.measurements import LengthMeasurement


def build_polygon(segments: list[WallSegment2D], config: GeometryConfig) -> tuple[list[RoomCorner], Polygon2D | None, list[WallSegment2D], float | None]:
    corners = []
    def array(point):
        return np.array([point.x, point.y])
    for i, first in enumerate(segments):
        a, b = array(first.start), array(first.end)
        u = (b-a)/np.linalg.norm(b-a)
        for second in segments[i+1:]:
            c, d = array(second.start), array(second.end)
            v = (d-c)/np.linalg.norm(d-c)
            matrix = np.column_stack([u, -v])
            if abs(np.linalg.det(matrix)) < np.sin(np.deg2rad(config.min_intersection_angle)):
                continue
            t, s = np.linalg.solve(matrix, c-a)
            lengths = (np.linalg.norm(b-a), np.linalg.norm(d-c))
            if not all(-config.intersection_extension <= value <= length + config.intersection_extension
                       and min(abs(value), abs(value-length)) <= config.intersection_extension
                       for value, length in zip((t, s), lengths)):
                continue
            point = a+t*u
            nearby = next((corner for corner in corners if np.linalg.norm(array(corner.point)-point) <= config.corner_merge_tolerance), None)
            if nearby is None:
                corners.append(RoomCorner(point={"x": point[0], "y": point[1]}, wall_ids=[first.wall_id, second.wall_id],
                                          quality_score=min(first.quality_score, second.quality_score)))
            else:
                nearby.wall_ids = sorted(set(nearby.wall_ids + [first.wall_id, second.wall_id]))
                nearby.quality_score = min(nearby.quality_score, first.quality_score, second.quality_score)
    if len(corners) < 3 or any(len(c.wall_ids) != 2 for c in corners):
        return corners, None, segments, None
    final, lines, gaps = [], [], []
    for segment in segments:
        ends = [c.point for c in corners if segment.wall_id in c.wall_ids]
        if len(ends) != 2:
            return corners, None, segments, None
        a, b = map(array, ends)
        observed = [array(segment.start), array(segment.end)]
        gaps.extend(min(np.linalg.norm(endpoint-p) for p in observed) for endpoint in (a, b))
        if np.linalg.norm(b-a) <= config.corner_merge_tolerance:
            return corners, None, segments, None
        lines.append(LineString([a, b]))
        final.append(segment.model_copy(update={"start": ends[0], "end": ends[1],
                     "length": LengthMeasurement(value=float(np.linalg.norm(b-a)), method="bounded_wall_intersections")}))
    polygons = list(polygonize(unary_union(lines)))
    if len(polygons) != 1:
        return corners, None, segments, None
    polygon = orient(polygons[0], sign=1.0)
    if (not polygon.is_valid or polygon.interiors or polygon.area < config.min_polygon_area
            or len(polygon.exterior.coords)-1 != len(segments)):
        return corners, None, segments, None
    result = Polygon2D(points=[{"x": x, "y": y} for x, y in list(polygon.exterior.coords)[:-1]])
    return corners, result, final, float(max(gaps))
