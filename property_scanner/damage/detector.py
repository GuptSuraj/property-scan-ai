"""Shared metric damage detection, multi-view fusion, and deterministic scope."""

from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import MultiPoint

from property_scanner.damage.models import DamageCandidate, DamageConfig, DamageFrame
from property_scanner.damage.vision import DamageVisionModel
from property_scanner.pipeline.context import ScanContext
from property_scanner.schemas.damage import ConcealedDamageFlag, DamageRegion, DamageType, ScopeLineItem
from property_scanner.schemas.geometry import PropertyGeometry
from property_scanner.schemas.measurements import AreaMeasurement, LengthMeasurement, MeasurementUnit, QuantityMeasurement
from property_scanner.schemas.primitives import BoundingBox2D, Point2D, Polygon2D
from property_scanner.schemas.result import ResultWarning


def _backproject(mask, depth, frame, config):
    intr = frame.intrinsics
    if depth.shape != mask.shape:
        return np.empty((0, 3))
    yy, xx = np.nonzero(mask)
    xx, yy = xx[::config.depth_sample_stride], yy[::config.depth_sample_stride]
    z = depth[yy, xx].astype(float)
    valid = np.isfinite(z) & (z >= config.depth_min_m) & (z <= config.depth_max_m)
    xx, yy, z = xx[valid], yy[valid], z[valid]
    if not len(z): return np.empty((0, 3))
    camera = np.column_stack(((xx-float(intr["cx"]))*z/float(intr["fx"]),
        (yy-float(intr["cy"]))*z/float(intr["fy"]), z, np.ones(len(z))))
    return (np.asarray(frame.camera_to_property, dtype=float) @ camera.T).T[:, :3]


def _associate(points, geometry, room_id, max_distance):
    choices = []
    for room in geometry.rooms:
        if room_id and room.room_id != room_id: continue
        for wall in room.walls:
            if wall.start_point is None or wall.end_point is None: continue
            a = np.array([wall.start_point.x, wall.start_point.y]); b = np.array([wall.end_point.x, wall.end_point.y])
            length = np.linalg.norm(b-a)
            if length < 1e-8: continue
            tangent = (b-a)/length; normal = np.array([-tangent[1], tangent[0]])
            u = (points[:, :2]-a)@tangent; dist = np.abs((points[:, :2]-a)@normal)
            valid = (u >= -max_distance) & (u <= length+max_distance)
            choices.append((float(np.median(dist[valid])) if valid.any() else np.inf,
                room.room_id, wall.wall_id, np.column_stack((u, points[:, 2])), valid))
        if room.floor:
            choices.append((float(np.median(np.abs(points[:, 2]))), room.room_id,
                room.floor.surface_id, points[:, :2], np.ones(len(points), bool)))
        if room.ceiling and room.ceiling.height:
            choices.append((float(np.median(np.abs(points[:, 2]-room.ceiling.height.value))), room.room_id,
                room.ceiling.surface_id, points[:, :2], np.ones(len(points), bool)))
    if not choices: return None
    best = min(choices, key=lambda item: item[0])
    if best[0] > max_distance: return None
    return best[1], best[2], best[3][best[4]], best[0]


def _extent(coords, low, high):
    lo, hi = np.percentile(coords, [low, high], axis=0)
    clipped = coords[np.all((coords >= lo) & (coords <= hi), axis=1)]
    hull = MultiPoint(clipped).convex_hull
    if hull.geom_type != "Polygon" or hull.area <= 1e-6: return None, None, None
    polygon = Polygon2D(points=[Point2D(x=float(x), y=float(y)) for x, y in list(hull.exterior.coords)[:-1]])
    box = BoundingBox2D(min_point=Point2D(x=float(hull.bounds[0]), y=float(hull.bounds[1])),
        max_point=Point2D(x=float(hull.bounds[2]), y=float(hull.bounds[3])))
    return polygon, box, float(hull.area)


class DamageDetector:
    def __init__(self, config=None, vision_model=None):
        self.config = config or DamageConfig(); self.vision_model = vision_model

    def detect(self, geometry: PropertyGeometry, frames: list[DamageFrame], *, diagnostics_dir: Path | None = None):
        if self.vision_model is None: raise ValueError("DamageDetector requires a local vision model")
        candidates, warnings = [], []
        for frame in frames:
            image = Image.open(frame.image_path).convert("RGB"); depth = np.load(frame.depth_path).astype(np.float32)
            for index, prediction in enumerate(self.vision_model.predict(image)):
                if prediction.score < self.config.semantic_min_score or prediction.mask.sum() < self.config.minimum_pixel_area: continue
                points = _backproject(prediction.mask, depth, frame, self.config)
                if len(points) < self.config.minimum_3d_points:
                    warnings.append(ResultWarning(code="DAMAGE_DEPTH_UNAVAILABLE", message=f"Candidate in {frame.frame_id} lacked metric depth.", room_id=frame.room_id)); continue
                association = _associate(points, geometry, frame.room_id, self.config.surface_max_distance_m)
                if association is None:
                    warnings.append(ResultWarning(code="DAMAGE_SURFACE_ASSOCIATION_FAILED", message=f"Candidate in {frame.frame_id} did not align with a known surface.", room_id=frame.room_id)); continue
                room, surface, coordinates, residual = association
                candidates.append(DamageCandidate(f"candidate:{frame.frame_id}:{index}", frame.frame_id,
                    prediction.damage_type, prediction.score, room, surface, coordinates, residual))
        regions = self._fuse(candidates)
        flags, scope = build_scope(regions)
        updated = geometry.model_copy(deep=True); by_surface, by_room = defaultdict(list), defaultdict(list)
        for region in regions:
            by_surface[region.surface_id].append(region.damage_id); by_room[region.room_id].append(region.damage_id)
        for room in updated.rooms:
            room.damage_ids = by_room[room.room_id]
            for wall in room.walls: wall.damage_ids = by_surface[wall.wall_id]
            if room.floor: room.floor.damage_ids = by_surface[room.floor.surface_id]
            if room.ceiling: room.ceiling.damage_ids = by_surface[room.ceiling.surface_id]
        return updated, regions, flags, scope, warnings

    def _fuse(self, candidates):
        groups = []
        for item in candidates:
            center = np.median(item.coordinates, axis=0)
            group = next((g for g in groups if g[0].surface_id == item.surface_id and g[0].damage_type == item.damage_type and
                np.linalg.norm(np.median(g[0].coordinates, axis=0)-center) <= self.config.multi_view_merge_distance_m), None)
            if group is None: groups.append([item])
            else: group.append(item)
        output = []
        for index, group in enumerate(groups, 1):
            frames = sorted({item.frame_id for item in group})
            if len(frames) < self.config.minimum_supporting_views: continue
            extents = [_extent(item.coordinates, self.config.robust_lower_percentile, self.config.robust_upper_percentile) for item in group]
            valid = [item for item in extents if item[0] is not None]
            areas = [item[2] for item in valid]
            lengths = []
            for item in group:
                centered = item.coordinates-np.median(item.coordinates, axis=0)
                if len(centered) >= 2:
                    _, _, vh = np.linalg.svd(centered, full_matrices=False); lengths.append(float(np.ptp(centered@vh[0])))
            area = float(np.median(areas)) if areas else None
            length = float(np.median(lengths)) if lengths and group[0].damage_type == DamageType.CRACK else None
            semantic = float(np.median([item.score for item in group]))
            residual = float(np.median([i.residual for i in group]))
            quality = min(1.0, .45*semantic + .30*max(0.0, 1-residual/self.config.surface_max_distance_m)
                          + .25*min(1.0, len(frames)/3))
            output.append(DamageRegion(damage_id=f"damage_{index:03d}", damage_type=group[0].damage_type,
                surface_id=group[0].surface_id, room_id=group[0].room_id,
                polygon_2d=valid[0][0] if valid else None, bounding_box=valid[0][1] if valid else None,
                metric_area=AreaMeasurement(value=area, method="metric surface projection") if area is not None else None,
                metric_length=LengthMeasurement(value=length, method="surface principal extent") if length is not None else None,
                confidence=quality, source_frames=frames,
                metadata={"supporting_views": len(frames), "semantic_score": semantic, "surface_residual_m": residual,
                    "area_mad_m2": float(np.median(np.abs(np.asarray(areas)-np.median(areas)))) if areas else None}))
        return output

    def run(self, context: ScanContext) -> ScanContext:
        if context.result is None: raise NotImplementedError("Damage analysis is not implemented without a reconstructed result")
        return context


def build_scope(regions):
    actions = {
        DamageType.CRACK: [("inspect_crack", "Inspect crack and substrate"), ("repair_crack", "Fill and finish the affected section"), ("repaint", "Prepare and repaint affected surface")],
        DamageType.WATER_DAMAGE: [("inspect_moisture_source", "Inspect and correct the moisture source"), ("repair_finish", "Remove damaged finish where required and repair surface"), ("refinish", "Repaint or refinish affected surface")],
        DamageType.MOISTURE_STAIN: [("inspect_moisture_source", "Inspect possible moisture source"), ("prepare_surface", "Clean and prepare stained finish"), ("repaint", "Repaint affected surface")],
        DamageType.MOLD: [("inspect_moisture_source", "Inspect underlying moisture source"), ("remediate", "Treat and clean affected finish")],
        DamageType.HOLE: [("patch", "Patch damaged surface"), ("finish", "Finish and repaint patch")],
        DamageType.PEELING_PAINT: [("remove_loose_coating", "Remove loose coating"), ("prepare_surface", "Prepare affected surface"), ("repaint", "Repaint affected area")],
        DamageType.SURFACE_DAMAGE: [("inspect", "Inspect affected surface"), ("repair_finish", "Repair and refinish affected surface")],
        DamageType.UNKNOWN: [("inspect", "Inspect unclassified visible surface condition")],
    }
    flags, scope = [], []
    for region in regions:
        if region.damage_type in {DamageType.WATER_DAMAGE, DamageType.MOISTURE_STAIN}:
            flags.append(ConcealedDamageFlag(flag_id=f"flag:{region.damage_id}:moisture", room_id=region.room_id, surface_id=region.surface_id,
                suspected_damage_type=DamageType.WATER_DAMAGE, rule_id="RULE_WATER_STAIN_CONCEALED_MOISTURE",
                rule_description="Visible water or moisture staining may indicate concealed moisture behind the finish.", evidence=[region.damage_id]))
        if region.damage_type == DamageType.MOLD:
            flags.append(ConcealedDamageFlag(flag_id=f"flag:{region.damage_id}:source", room_id=region.room_id, surface_id=region.surface_id,
                suspected_damage_type=DamageType.WATER_DAMAGE, rule_id="RULE_MOLD_MOISTURE_SOURCE",
                rule_description="Visible mold may indicate an underlying moisture source.", evidence=[region.damage_id]))
        quantity = None
        if region.metric_area: quantity = QuantityMeasurement(value=region.metric_area.value, unit=MeasurementUnit.SQUARE_METER, method="visible metric damage extent")
        elif region.metric_length: quantity = QuantityMeasurement(value=region.metric_length.value, unit=MeasurementUnit.METER, method="visible metric damage extent")
        for number, (action, description) in enumerate(actions[region.damage_type], 1):
            scope.append(ScopeLineItem(line_item_id=f"scope:{region.damage_id}:{number}", room_id=region.room_id, surface_id=region.surface_id,
                damage_id=region.damage_id, action=action, description=description, quantity=quantity))
    return flags, scope
