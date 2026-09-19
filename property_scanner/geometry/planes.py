"""Bounded iterative RANSAC and evidence-based horizontal plane selection."""
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from shapely.geometry import MultiPoint
from property_scanner.geometry.config import GeometryConfig
from property_scanner.geometry.exceptions import FloorNotDetectedError
from property_scanner.geometry.models import DetectedPlane


@dataclass
class PlaneCandidate:
    indices: np.ndarray
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    area: float
    rmse: float


def fit_plane(points: np.ndarray, indices: np.ndarray) -> PlaneCandidate:
    selected = points[indices]
    center = selected.mean(axis=0)
    _, _, vectors = np.linalg.svd(selected - center, full_matrices=False)
    normal = vectors[-1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    offset = -float(normal @ center)
    projected = (selected - center) @ vectors[:2].T
    area = float(MultiPoint(projected).convex_hull.area)
    residual = selected @ normal + offset
    return PlaneCandidate(indices, normal, offset, center, area, float(np.sqrt(np.mean(residual**2))))


def detect_planes(cloud: o3d.geometry.PointCloud, config: GeometryConfig) -> list[PlaneCandidate]:
    o3d.utility.random.seed(config.random_seed)
    points = np.asarray(cloud.points)
    remaining = np.arange(len(points))
    candidates = []
    for _ in range(config.max_planes):
        if len(remaining) < max(config.min_plane_inliers, config.ransac_n):
            break
        subset = cloud.select_by_index(remaining.tolist())
        _, inliers = subset.segment_plane(config.ransac_distance_threshold, config.ransac_n,
                                         config.ransac_iterations, probability=1.0)
        if len(inliers) < config.min_plane_inliers:
            break
        indices = remaining[inliers]
        candidates.append(fit_plane(points, indices))
        keep = np.ones(len(remaining), dtype=bool)
        keep[inliers] = False
        remaining = remaining[keep]
    return candidates


def horizontal(candidate: PlaneCandidate, degrees: float) -> bool:
    return abs(candidate.normal[2]) >= np.cos(np.deg2rad(degrees))


def choose_floor(candidates: list[PlaneCandidate], points: np.ndarray, config: GeometryConfig) -> PlaneCandidate:
    footprint = float(MultiPoint(points[:, :2]).convex_hull.area)
    eligible = [p for p in candidates if horizontal(p, config.horizontal_angle_tolerance)
                and p.area >= max(config.min_horizontal_area, footprint * config.min_floor_coverage)
                and abs(float(np.quantile(points @ p.normal + p.offset, config.extent_quantile)))
                <= config.floor_elevation_tolerance]
    if not eligible:
        raise FloorNotDetectedError("No sufficiently supported floor near the cloud base; check up_axis, scale, and coverage")
    return max(eligible, key=lambda p: (p.area * len(p.indices), -p.centroid[2]))


def floor_transform(floor: PlaneCandidate) -> np.ndarray:
    normal = floor.normal if floor.normal[2] > 0 else -floor.normal
    x = np.array([1., 0., 0.])
    x -= normal * (x @ normal)
    x /= np.linalg.norm(x)
    rotation = np.vstack([x, np.cross(normal, x), normal])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    # Keep a reproducible XY origin close to the source origin; floor elevation is zero.
    transform[2, 3] = -float(normal @ floor.centroid)
    return transform


def choose_ceiling(candidates: list[PlaneCandidate], floor: PlaneCandidate,
                   points: np.ndarray, config: GeometryConfig) -> PlaneCandidate | None:
    top = np.quantile(points[:, 2], 1 - config.extent_quantile)
    eligible = [p for p in candidates if p is not floor
                and horizontal(p, config.ceiling_parallel_tolerance)
                and config.min_ceiling_height <= p.centroid[2] <= config.max_ceiling_height
                and p.area >= max(config.min_horizontal_area, floor.area * config.min_ceiling_coverage)
                and p.centroid[2] >= top - config.ceiling_top_tolerance]
    return max(eligible, key=lambda p: (p.area * len(p.indices), p.centroid[2]), default=None)


def plane_model(candidate: PlaneCandidate, plane_id: str, count: int, points: np.ndarray, config: GeometryConfig) -> DetectedPlane:
    p = points[candidate.indices]
    normal = candidate.normal
    direction = np.array([-normal[1], normal[0], 0.])
    vertical = abs(normal[2]) <= np.sin(np.deg2rad(config.vertical_angle_tolerance))
    if vertical:
        direction /= np.linalg.norm(direction)
        limits = np.quantile(p @ direction, [config.extent_quantile, 1-config.extent_quantile])
        width = float(limits[1] - limits[0])
    else:
        width = float(np.linalg.norm(np.ptp(p[:, :2], axis=0)))
    return DetectedPlane(
        plane_id=plane_id, equation=(*normal.tolist(), candidate.offset),
        normal=dict(zip("xyz", normal.tolist())), centroid=dict(zip("xyz", candidate.centroid.tolist())),
        inlier_count=len(p), inlier_ratio=len(p)/count, area_estimate=candidate.area,
        orientation="vertical" if vertical else "horizontal" if horizontal(candidate, config.horizontal_angle_tolerance) else "other",
        residual_rmse=candidate.rmse, width=width,
        height_coverage=float(np.ptp(np.quantile(p[:, 2], [config.extent_quantile, 1-config.extent_quantile]))),
        orientation_degrees=float(np.degrees(np.arctan2(direction[1], direction[0])) % 360),
        quality_score=float(np.exp(-candidate.rmse / config.ransac_distance_threshold)),
    )
