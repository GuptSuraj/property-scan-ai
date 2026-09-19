"""Coordinate geometry steps without acquisition-specific branches."""
from pathlib import Path
import numpy as np
from property_scanner.geometry.config import GeometryConfig
from shapely.geometry import Polygon
from property_scanner.core.exceptions import ProcessingError
from property_scanner.geometry.preprocessing import load_cloud, preprocess, axis_rotation
from property_scanner.geometry.planes import detect_planes, choose_floor, floor_transform, fit_plane, choose_ceiling, plane_model
from property_scanner.geometry.walls import select_walls, merge_walls, project_wall
from property_scanner.geometry.polygon import build_polygon
from property_scanner.geometry.models import RoomGeometryResult, GeometryDiagnostics
from property_scanner.schemas.measurements import LengthMeasurement, AreaMeasurement
from property_scanner.schemas.result import ResultWarning


def process(path: Path, config: GeometryConfig, diagnostics_dir: Path | None) -> RoomGeometryResult:
    try:
        return _process(path, config, diagnostics_dir)
    except (RuntimeError, OSError) as exc:
        raise ProcessingError(f"Geometry processing failed: {exc}") from exc


def _process(path: Path, config: GeometryConfig, diagnostics_dir: Path | None) -> RoomGeometryResult:
    cloud = load_cloud(path, config)
    cloud, original, downsampled = preprocess(cloud, config)
    candidates = detect_planes(cloud, config)
    floor = choose_floor(candidates, np.asarray(cloud.points), config)
    floor_index = next(i for i, p in enumerate(candidates) if p is floor)
    leveling = floor_transform(floor)
    cloud.transform(leveling)
    points = np.asarray(cloud.points)
    candidates = [fit_plane(points, p.indices) for p in candidates]
    floor = candidates[floor_index]
    ceiling = choose_ceiling(candidates, floor, points, config)
    walls, rejected = select_walls([p for p in candidates if p is not floor and p is not ceiling], points, config)
    walls, merged = merge_walls(walls, points, config)
    count = len(points)
    floor_model = plane_model(floor, "floor", count, points, config)
    ceiling_model = plane_model(ceiling, "ceiling", count, points, config) if ceiling is not None else None
    models = [plane_model(p, f"wall_{i+1:02}", count, points, config) for i, p in enumerate(walls)]
    segments = [project_wall(p, model, points, config) for p, model in zip(walls, models)]
    corners, polygon, segments, closure = build_polygon(segments, config)
    warnings = []
    height = None
    if ceiling is None:
        warnings.append(ResultWarning(code="CEILING_UNAVAILABLE", message="No supported ceiling near the cloud top within configured height bounds."))
    else:
        distance = -(ceiling.normal @ floor.centroid + ceiling.offset) / (ceiling.normal @ floor.normal)
        height = LengthMeasurement(value=abs(float(distance)), method="plane_separation_along_floor_normal_at_floor_centroid")
    if polygon is None:
        warnings.append(ResultWarning(code="POLYGON_UNAVAILABLE", message="Wall intersections do not form one unambiguous closed room; observed wall extents retained."))
    structural = [floor, *walls] + ([ceiling] if ceiling is not None else [])
    explained = len(np.unique(np.concatenate([p.indices for p in structural]))) / count
    all_models = [floor_model, *models] + ([ceiling_model] if ceiling_model else [])
    transform = np.eye(4)
    transform[:3, :3] = axis_rotation(config.up_axis)
    transform = leveling @ transform
    result = RoomGeometryResult(
        floor_plane=floor_model, ceiling_plane=ceiling_model, ceiling_height=height,
        wall_planes=models, wall_segments_2d=segments, corners=corners, room_polygon=polygon,
        floor_area=AreaMeasurement(value=float(Polygon([(p.x, p.y) for p in polygon.points]).area), method="shapely_polygon_area") if polygon else None,
        diagnostics=GeometryDiagnostics(original_points=original, downsampled_points=downsampled, filtered_points=count,
            floor_inlier_ratio=len(floor.indices)/count, ceiling_inlier_ratio=len(ceiling.indices)/count if ceiling else None,
            wall_inlier_ratio=sum(len(p.indices) for p in walls)/count, explained_point_ratio=explained,
            detected_wall_count=len(walls), rejected_wall_candidates=rejected, merged_wall_candidates=merged,
            polygon_closure_error=closure, plane_residual_rmse={p.plane_id: p.residual_rmse for p in all_models}),
        warnings=warnings, source_to_floor_transform=transform.tolist(),
    )
    if diagnostics_dir is not None:
        from property_scanner.geometry.diagnostics import export_diagnostics
        export_diagnostics(diagnostics_dir, cloud, structural, all_models, result, config)
    return result
