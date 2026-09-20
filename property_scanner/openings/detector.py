"""Shared semantic/metric door and window detection on existing wall geometry."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon

from property_scanner.pipeline.context import ScanContext
from property_scanner.schemas.geometry import ConnectionType, Opening, OpeningType, PropertyGeometry
from property_scanner.schemas.measurements import LengthMeasurement
from property_scanner.schemas.result import ResultWarning
from property_scanner.openings.models import OpeningCandidate, OpeningConfig, OpeningDiagnostics, OpeningFrame
from property_scanner.openings.semantic import SemanticSurfaceDetector
from property_scanner.openings.occupancy import geometry_opening_candidates


@dataclass
class OpeningDetectionResult:
    property: PropertyGeometry
    candidates: list[OpeningCandidate]
    diagnostics: OpeningDiagnostics
    warnings: list[ResultWarning]


def _warning(code, message, room_id=None):
    return ResultWarning(code=code, message=message, room_id=room_id)


def _components(probability, kind, frame, config):
    binary = (probability >= config.semantic_min_score).astype(np.uint8)
    if config.mask_morphology_size > 1:
        kernel = np.ones((config.mask_morphology_size,)*2, np.uint8)
        binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < config.candidate_min_pixel_area:
            continue
        x, y, width, height = (int(stats[index, field]) for field in
            (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        mask = labels == index
        result.append((mask, OpeningCandidate(candidate_id=f"candidate:{frame.frame_id}:{kind}:{index}",
            frame_id=frame.frame_id, semantic_type=kind, bounding_box=(x, y, x+width, y+height),
            pixel_area=area, semantic_score=float(np.median(probability[mask])), room_id=frame.room_id)))
    return result


def _wall_records(geometry, room_id):
    records = []
    for room in geometry.rooms:
        if room_id is not None and room.room_id != room_id:
            continue
        for wall in room.walls:
            if wall.start_point is None or wall.end_point is None:
                continue
            start = np.array([wall.start_point.x, wall.start_point.y], dtype=float)
            end = np.array([wall.end_point.x, wall.end_point.y], dtype=float)
            length = float(np.linalg.norm(end-start))
            if length > 1e-6:
                records.append((room, wall, start, (end-start)/length, length))
    return records


def _metric_candidate(mask, candidate, frame, depth, geometry, config):
    k = frame.intrinsics
    width, height = int(k["width"]), int(k["height"])
    if depth.shape != (height, width) or mask.shape != depth.shape:
        return candidate.model_copy(update={"rejection_reason": "OPENING_DEPTH_UNAVAILABLE"})
    rows, cols = np.nonzero(mask)
    selected = np.arange(0, len(rows), config.depth_sample_stride)
    rows, cols = rows[selected], cols[selected]
    z = depth[rows, cols].astype(float)
    valid = np.isfinite(z) & (z >= config.depth_min_m) & (z <= config.depth_max_m)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if len(z) < config.minimum_3d_points:
        return candidate.model_copy(update={"sample_count": len(z), "rejection_reason": "OPENING_DEPTH_UNAVAILABLE"})
    camera = np.column_stack(((cols-float(k["cx"]))/float(k["fx"])*z,
        (rows-float(k["cy"]))/float(k["fy"])*z, z, np.ones(len(z))))
    pose = np.asarray(frame.camera_to_property, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return candidate.model_copy(update={"rejection_reason": "OPENING_WALL_ASSOCIATION_FAILED"})
    points = (pose @ camera.T).T[:, :3]
    best = None
    for room, wall, start, direction, length in _wall_records(geometry, frame.room_id):
        relative = points[:, :2]-start
        along = relative @ direction
        normal = np.abs(relative[:, 0]*direction[1]-relative[:, 1]*direction[0])
        usable = ((along >= -config.wall_endpoint_tolerance) & (along <= length+config.wall_endpoint_tolerance)
                  & (normal <= config.wall_association_max_distance))
        count = int(usable.sum())
        if count < config.minimum_3d_points:
            continue
        distance = float(np.median(normal[usable]))
        score = count/(1+distance/config.wall_association_max_distance)
        if best is None or score > best[0]:
            best = score, room, wall, length, along[usable], points[usable, 2], distance, count
    if best is None:
        return candidate.model_copy(update={"sample_count": len(points),
            "rejection_reason": "OPENING_WALL_ASSOCIATION_FAILED"})
    _, room, wall, wall_length, u, v, distance, count = best
    u_min, u_max = np.percentile(u, [config.robust_lower_percentile, config.robust_upper_percentile])
    v_min, v_max = np.percentile(v, [config.robust_lower_percentile, config.robust_upper_percentile])
    u_min, u_max = max(0.0, float(u_min)), min(wall_length, float(u_max))
    v_min, v_max = max(0.0, float(v_min)), float(v_max)
    opening_width, opening_height = u_max-u_min, v_max-v_min
    kind, reason = candidate.semantic_type, None
    if opening_width <= 0 or u_max > wall_length+1e-8:
        reason = "OPENING_EXCEEDS_WALL_BOUNDS"
    elif opening_height < config.opening_min_height:
        reason = "OPENING_DIMENSIONS_UNRESOLVED"
    elif kind == "door" and not config.door_min_width <= opening_width <= config.door_max_width:
        reason = "POSSIBLE_PHANTOM_OPENING"
    elif kind == "window" and not config.window_min_width <= opening_width <= config.window_max_width:
        reason = "POSSIBLE_PHANTOM_OPENING"
    association = max(0.0, 1-distance/config.wall_association_max_distance)
    geometry_score = 0.7*association+0.3*min(1.0, count/(config.minimum_3d_points*3))
    quality = 0.45*candidate.semantic_score+0.55*geometry_score
    if kind == "door" and v_min > config.floor_contact_tolerance:
        quality *= 0.75
    if kind == "window" and v_min <= config.floor_contact_tolerance:
        quality *= 0.5
        if quality < config.single_view_min_quality:
            reason = reason or "POSSIBLE_PHANTOM_OPENING"
    return candidate.model_copy(update={"wall_id": wall.wall_id, "room_id": room.room_id,
        "u_min": u_min, "u_max": u_max, "v_min": v_min, "v_max": v_max,
        "sample_count": count, "wall_distance_median": distance, "geometry_score": geometry_score,
        "quality": quality, "accepted": reason is None, "rejection_reason": reason,
        "metadata": {"wall_length_m": wall_length, "depth_source": frame.depth_source}})


def _mad(values):
    values = np.asarray(values, dtype=float)
    return float(np.median(np.abs(values-np.median(values))))


def _cluster(candidates, distance):
    groups = []
    for candidate in sorted(candidates, key=lambda item: (item.wall_id or "", item.u_min or 0)):
        center = (candidate.u_min+candidate.u_max)/2
        target = None
        for group in groups:
            other = group[0]
            if candidate.wall_id == other.wall_id and abs(center-(other.u_min+other.u_max)/2) <= distance:
                target = group; break
        if target is None:
            groups.append([candidate])
        else:
            target.append(candidate)
    return groups


def _fuse_group(group, index, config):
    frames = sorted({item.frame_id for item in group if item.frame_id != "geometry"})
    widths = [item.u_max-item.u_min for item in group]
    heights = [item.v_max-item.v_min for item in group]
    u_min, width = float(np.median([item.u_min for item in group])), float(np.median(widths))
    v_min, height = float(np.median([item.v_min for item in group])), float(np.median(heights))
    width_mad, height_mad = _mad(widths), _mad(heights)
    scores = Counter()
    for item in group:
        scores[item.semantic_type] += item.semantic_score
    ranked = scores.most_common()
    ambiguous = len(ranked) > 1 and ranked[0][1]-ranked[1][1] < 0.15*sum(scores.values())
    kind = "unknown" if ambiguous else ranked[0][0]
    quality = float(np.median([item.quality for item in group]))
    accepted = len(frames) >= config.minimum_supporting_views or (len(frames) == 1 and quality >= config.single_view_min_quality) or not frames
    reason = None; height_reliable = height_mad <= config.max_measurement_dispersion
    if width_mad > config.max_measurement_dispersion:
        accepted, reason = False, "OPENING_MULTI_VIEW_INCONSISTENT"
    elif not accepted:
        reason = "OPENING_LOW_SUPPORT"
    elif not height_reliable:
        reason = "OPENING_DIMENSIONS_UNRESOLVED"
    opening = Opening(opening_id=f"detected:{group[0].wall_id}:{index:02d}", type=OpeningType(kind),
        wall_id=group[0].wall_id, room_ids=[group[0].room_id],
        width=LengthMeasurement(value=width, method="median robust wall-local extent"),
        height=LengthMeasurement(value=height, method="median robust wall-local extent")
            if height_reliable and height >= config.opening_min_height else None,
        sill_height=LengthMeasurement(value=v_min, method="height above canonical floor") if kind == "window" else None,
        position_along_wall=LengthMeasurement(value=u_min, method="distance from deterministic wall start"),
        confidence=quality, metadata={"supporting_frame_ids": frames, "number_of_views": len(frames),
            "width_mad_m": width_mad, "height_mad_m": height_mad, "height_reliable": height_reliable,
            "internal_quality_not_calibrated_probability": quality})
    return opening, accepted, reason, ambiguous


def _link_opening(geometry, opening, config):
    owner = next((room for room in geometry.rooms if any(w.wall_id == opening.wall_id for w in room.walls)), None)
    if owner is None:
        return
    wall = next((wall for wall in owner.walls if wall.wall_id == opening.wall_id), None)
    if wall is None:
        return
    wall.opening_ids.append(opening.opening_id); owner.opening_ids.append(opening.opening_id)
    direction = np.array([wall.end_point.x-wall.start_point.x, wall.end_point.y-wall.start_point.y], dtype=float)
    direction /= np.linalg.norm(direction)
    midpoint = np.array([wall.start_point.x, wall.start_point.y])+direction*(opening.position_along_wall.value+opening.width.value/2)
    choices = []
    if geometry.metadata.get("coordinate_scope") == "property_shared":
        for connection in geometry.room_connections:
            if owner.room_id not in (connection.room_a_id, connection.room_b_id):
                continue
            if connection.opening_id is not None:
                continue
            other_id = connection.room_b_id if connection.room_a_id == owner.room_id else connection.room_a_id
            other = next((room for room in geometry.rooms if room.room_id == other_id), None)
            if other is None:
                continue
            if other.polygon:
                polygon = Polygon([(point.x, point.y) for point in other.polygon.points])
                choices.append((polygon.boundary.distance(Point(midpoint)), connection, other_id))
    if choices:
        distance, connection, other_id = min(choices, key=lambda item: item[0])
        if distance <= config.connection_boundary_tolerance:
            opening.room_ids.append(other_id); connection.opening_id = opening.opening_id
            if opening.type == OpeningType.DOOR: connection.connection_type = ConnectionType.DOOR
            elif opening.type == OpeningType.OPEN_PASSAGE: connection.connection_type = ConnectionType.OPEN_PASSAGE


class OpeningDetector:
    """One detector for photo, video, and LiDAR canonical evidence."""
    def __init__(self, config=None, semantic_detector=None):
        self.config = config or OpeningConfig(); self.semantic_detector = semantic_detector

    def detect(self, property_geometry, frames, *, diagnostics_dir=None, structural_points=None):
        if self.semantic_detector is None:
            raise ValueError("OpeningDetector requires a SemanticSurfaceDetector")
        geometry = property_geometry.model_copy(deep=True)
        old = {item.opening_id for item in geometry.openings if item.metadata.get("detector") == "shared_opening_detector"}
        geometry.openings = [item for item in geometry.openings if item.opening_id not in old]
        for connection in geometry.room_connections:
            if connection.opening_id in old:
                connection.opening_id = None
                if connection.connection_type in (ConnectionType.DOOR, ConnectionType.OPEN_PASSAGE):
                    connection.connection_type = ConnectionType.INFERRED
        for room in geometry.rooms:
            room.opening_ids = [item for item in room.opening_ids if item not in old]
            for wall in room.walls: wall.opening_ids = [item for item in wall.opening_ids if item not in old]
        candidates, warnings, depth_counts = [], [], Counter()
        mask_dir = Path(diagnostics_dir)/"semantic_masks" if diagnostics_dir else None
        overlay_dir = Path(diagnostics_dir)/"candidate_overlays" if diagnostics_dir else None
        if mask_dir: mask_dir.mkdir(parents=True, exist_ok=True)
        if overlay_dir: overlay_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            try:
                frame_start = len(candidates)
                with Image.open(frame.image_path) as source: image = source.convert("RGB")
                depth = np.load(frame.depth_path, allow_pickle=False)
                predictions = self.semantic_detector.predict(image); depth_counts[frame.depth_source] += 1
                for kind in ("door", "window"):
                    probability = np.asarray(predictions.get(kind), dtype=np.float32)
                    if probability.shape != depth.shape: raise ValueError("Semantic mask, depth, and calibration dimensions differ")
                    if mask_dir and self.config.cache_semantic_masks:
                        Image.fromarray(np.uint8(np.clip(probability*255, 0, 255))).save(mask_dir/f"{frame.frame_id.replace(':', '_')}_{kind}.png")
                    for mask, raw in _components(probability, kind, frame, self.config):
                        candidates.append(_metric_candidate(mask, raw, frame, depth, geometry, self.config))
                if overlay_dir:
                    overlay = image.copy(); drawing = ImageDraw.Draw(overlay)
                    for item in candidates[frame_start:]:
                        if item.bounding_box is None: continue
                        color = "#18864b" if item.accepted else "#ba3b35"
                        drawing.rectangle(item.bounding_box, outline=color, width=3)
                        label = f"{item.semantic_type} | {item.wall_id or 'no wall'} | {'accepted' if item.accepted else item.rejection_reason}"
                        drawing.text((item.bounding_box[0], max(0, item.bounding_box[1]-14)), label, fill=color)
                    overlay.save(overlay_dir/f"{frame.frame_id.replace(':', '_')}.jpg", quality=92)
            except (OSError, ValueError, RuntimeError) as exc:
                warnings.append(_warning("OPENING_DEPTH_UNAVAILABLE", f"Frame {frame.frame_id}: {exc}", frame.room_id))
        accepted_candidates = [item for item in candidates if item.accepted]
        if structural_points is not None and self.config.enable_geometry_fallback:
            geometric = geometry_opening_candidates(geometry, structural_points, self.config)
            # Retain geometry-only holes only when no semantic candidate already covers them.
            for item in geometric:
                center = (item.u_min+item.u_max)/2
                duplicate = any(other.wall_id == item.wall_id and
                    abs(center-(other.u_min+other.u_max)/2) <= self.config.multi_view_merge_distance
                    for other in accepted_candidates)
                if not duplicate:
                    candidates.append(item); accepted_candidates.append(item)
        fused, fusion_rejected = [], 0
        for index, group in enumerate(_cluster(accepted_candidates, self.config.multi_view_merge_distance), start=1):
            opening, accepted, reason, ambiguous = _fuse_group(group, index, self.config)
            if ambiguous: warnings.append(_warning("OPENING_TYPE_AMBIGUOUS", f"Conflicting semantic types on {opening.wall_id}.", opening.room_ids[0]))
            if not accepted:
                fusion_rejected += len(group); warnings.append(_warning(reason, f"Candidate on {opening.wall_id} rejected.", opening.room_ids[0])); continue
            if reason:
                warnings.append(_warning(reason, f"Opening {opening.opening_id} retained without reliable height.", opening.room_ids[0]))
            opening.metadata["detector"] = "shared_opening_detector"
            _link_opening(geometry, opening, self.config); fused.append(opening)
        geometry.openings.extend(fused)
        if diagnostics_dir and fused:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from matplotlib.patches import Rectangle
            wall_dir = Path(diagnostics_dir)/"wall_projection"; wall_dir.mkdir(parents=True, exist_ok=True)
            for wall_id in sorted({item.wall_id for item in fused}):
                wall = next((wall for room in geometry.rooms for wall in room.walls if wall.wall_id == wall_id), None)
                if wall is None:
                    continue
                length = wall.length.value if wall.length else float(np.hypot(
                    wall.end_point.x-wall.start_point.x, wall.end_point.y-wall.start_point.y))
                figure=Figure(figsize=(8,3)); FigureCanvasAgg(figure); axis=figure.subplots()
                axis.set(xlim=(0,length),ylim=(0,4),xlabel="Position along wall (m)",ylabel="Height above floor (m)",title=wall_id)
                for item in [opening for opening in fused if opening.wall_id == wall_id]:
                    height=item.height.value if item.height else 0
                    bottom=item.sill_height.value if item.sill_height else 0
                    axis.add_patch(Rectangle((item.position_along_wall.value,bottom),item.width.value,height,
                        fill=False,linewidth=2,label=f"{item.type.value}: {item.width.value:.2f} m"))
                axis.legend(); figure.savefig(wall_dir/f"{wall_id.replace(':','_')}.png",dpi=150,bbox_inches="tight"); figure.clear()
        for item in candidates:
            if item.rejection_reason:
                warnings.append(_warning(item.rejection_reason,
                    f"Candidate {item.candidate_id} rejected: {item.rejection_reason}.", item.room_id))
        diagnostics = OpeningDiagnostics(frames_processed=len(frames), candidates_total=len(candidates),
            candidates_accepted=len(accepted_candidates)-fusion_rejected,
            candidates_rejected=len(candidates)-len(accepted_candidates)+fusion_rejected,
            openings_fused=len(fused), semantic_device=self.semantic_detector.device,
            depth_source_counts=dict(depth_counts), warnings_by_code=dict(Counter(w.code for w in warnings)))
        return OpeningDetectionResult(geometry, candidates, diagnostics, warnings)

    def run(self, context: ScanContext) -> ScanContext:
        if context.result is None:
            raise NotImplementedError("Generic context opening detection is not implemented without calibrated frame evidence.")
        return context
