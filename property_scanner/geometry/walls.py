"""Vertical plane selection, duplicate merging, and floor projection."""
import numpy as np
from property_scanner.geometry.planes import fit_plane, PlaneCandidate
from property_scanner.geometry.models import WallSegment2D, DetectedPlane
from property_scanner.geometry.config import GeometryConfig
from property_scanner.schemas.measurements import LengthMeasurement


def wall_extent(candidate: PlaneCandidate, points: np.ndarray, config: GeometryConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = candidate.normal[:2]
    direction = np.array([-normal[1], normal[0]]) / np.linalg.norm(normal)
    selected = points[candidate.indices]
    limits = np.quantile(selected[:, :2] @ direction, [config.extent_quantile, 1-config.extent_quantile])
    heights = np.quantile(selected[:, 2], [config.extent_quantile, 1-config.extent_quantile])
    return direction, limits, heights


def select_walls(candidates: list[PlaneCandidate], points: np.ndarray, config: GeometryConfig) -> tuple[list[PlaneCandidate], int]:
    walls, rejected = [], 0
    for plane in candidates:
        if abs(plane.normal[2]) > np.sin(np.deg2rad(config.vertical_angle_tolerance)):
            continue
        _, limits, heights = wall_extent(plane, points, config)
        if (limits[1]-limits[0] < config.min_wall_length
                or heights[1]-heights[0] < config.min_wall_height
                or abs(heights[0]) > config.wall_base_tolerance):
            rejected += 1
        else:
            walls.append(plane)
    return walls, rejected


def merge_walls(walls: list[PlaneCandidate], points: np.ndarray, config: GeometryConfig) -> tuple[list[PlaneCandidate], int]:
    walls = list(walls)
    merged = 0
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(walls):
            for j in range(i+1, len(walls)):
                second = walls[j]
                if abs(float(first.normal @ second.normal)) < np.cos(np.deg2rad(config.wall_merge_angle_tolerance)):
                    continue
                distance = max(abs(first.normal @ second.centroid + first.offset), abs(second.normal @ first.centroid + second.offset))
                if distance > config.wall_merge_distance_tolerance:
                    continue
                direction, extent, heights = wall_extent(first, points, config)
                other = points[second.indices]
                other_extent = np.quantile(other[:, :2] @ direction, [config.extent_quantile, 1-config.extent_quantile])
                other_height = np.quantile(other[:, 2], [config.extent_quantile, 1-config.extent_quantile])
                gap = max(extent[0], other_extent[0]) - min(extent[1], other_extent[1])
                height_gap = max(heights[0], other_height[0]) - min(heights[1], other_height[1])
                if max(gap, height_gap) > config.wall_merge_gap_tolerance:
                    continue
                walls[i] = fit_plane(points, np.union1d(first.indices, second.indices))
                walls.pop(j)
                merged += 1
                changed = True
                break
            if changed:
                break
    walls.sort(key=lambda p: tuple(p.centroid.tolist()))
    return walls, merged


def project_wall(candidate: PlaneCandidate, model: DetectedPlane, points: np.ndarray, config: GeometryConfig) -> WallSegment2D:
    direction, limits, _ = wall_extent(candidate, points, config)
    normal = candidate.normal[:2]
    origin = -candidate.offset * normal / (normal @ normal)
    start, end = origin + limits[0]*direction, origin + limits[1]*direction
    return WallSegment2D(wall_id=model.plane_id, source_plane_id=model.plane_id,
                         start={"x": start[0], "y": start[1]}, end={"x": end[0], "y": end[1]},
                         length=LengthMeasurement(value=float(np.linalg.norm(end-start)), method="observed_plane_extent"),
                         quality_score=model.quality_score)
