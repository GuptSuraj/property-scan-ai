"""Tier adapters for cached calibrated frames and opening post-processing."""

import json
from pathlib import Path
import numpy as np

from property_scanner.core.exceptions import ConfigurationError
from property_scanner.core.exceptions import PropertyScannerError
from property_scanner.openings.detector import OpeningDetector, OpeningDetectionResult
from property_scanner.openings.models import OpeningConfig, OpeningFrame, SEMANTIC_CACHE_NAME
from property_scanner.openings.semantic import SegFormerSurfaceDetector, SemanticSurfaceDetector
from property_scanner.openings.occupancy import export_wall_occupancy_plots
from property_scanner.reconstruction.lidar.diagnostics import write_json
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.schemas.result import PropertyScanResult, ResultWarning
from property_scanner.schemas.serialization import load_result, save_result


def _pose_lookup(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text())
    return {int(item["frame_id"]): np.asarray(item["matrix"], dtype=float) for item in payload["poses"]}


def _intrinsics_lookup(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {item["filename"]: item["intrinsics"] for item in payload["frames"]}


def load_cached_frames(output: Path, result: PropertyScanResult) -> list[OpeningFrame]:
    output, tier = Path(output), result.capture.tier.value
    frames = []
    if tier == "lidar":
        index = output/"lidar/opening_frames/frames.json"
        if index.is_file():
            for item in json.loads(index.read_text())["frames"]:
                for key in ("image_path", "depth_path"):
                    path = Path(item[key])
                    item[key] = output/path if not path.is_absolute() else path
                frames.append(OpeningFrame.model_validate(item))
        return frames
    if tier == "video":
        root = output/"video"
        poses = _pose_lookup(root/"canonical_poses.json")
        intrinsics = _intrinsics_lookup(root/"sfm/reconstruction.json")
        for image in sorted((root/"keyframes").glob("*.jpg")):
            frame_id = int(image.stem); depth = root/"depth"/f"{image.stem}.npy"
            if frame_id in poses and image.name in intrinsics and depth.is_file():
                frames.append(OpeningFrame(frame_id=f"video:{frame_id}", image_path=image,
                    depth_path=depth, intrinsics=intrinsics[image.name],
                    camera_to_property=poses[frame_id].tolist(), room_id="room_01",
                    depth_source="estimated"))
        return frames
    for room_file in sorted((output/"photo").glob("*/room.json")):
        room_id, root = room_file.parent.name, room_file.parent
        pose_path, reconstruction = root/"canonical_poses.json", root/"sfm/reconstruction.json"
        if not pose_path.is_file() or not reconstruction.is_file():
            continue
        poses, intrinsics = _pose_lookup(pose_path), _intrinsics_lookup(reconstruction)
        room_model = next((room for room in result.property.rooms if room.room_id == room_id), None)
        room_transform = np.eye(4)
        if room_model and room_model.metadata.get("coordinate_scope") == "property_shared":
            matrix = np.asarray(room_model.metadata["room_to_property_transform"], dtype=float)
            room_transform[:2, :2], room_transform[:2, 3] = matrix[:2, :2], matrix[:2, 2]
        for image in sorted((root/"selected_images").glob("*.jpg")):
            frame_id = int(image.stem); depth = root/"depth"/f"{image.stem}.npy"
            if frame_id in poses and image.name in intrinsics and depth.is_file():
                frames.append(OpeningFrame(frame_id=f"photo:{room_id}:{frame_id}", image_path=image,
                    depth_path=depth, intrinsics=intrinsics[image.name],
                    camera_to_property=(room_transform@poses[frame_id]).tolist(), room_id=room_id,
                    depth_source="estimated"))
    return frames


def load_structural_points(output: Path, result: PropertyScanResult, maximum=600_000) -> np.ndarray | None:
    try:
        import open3d as o3d
        tier, clouds = result.capture.tier.value, []
        if tier == "lidar":
            candidates = [output/"lidar/corrected_fused.ply", output/"lidar/raw_fused.ply"]
            path = next((item for item in candidates if item.is_file()), None)
            if path: clouds.append((path, np.eye(4)))
        elif tier == "video":
            path = output/"video/fused_video.ply"
            if path.is_file(): clouds.append((path, np.eye(4)))
        else:
            for room in result.property.rooms:
                path = output/"photo"/room.room_id/"room_fused.ply"
                if not path.is_file(): continue
                transform = np.eye(4)
                if room.metadata.get("coordinate_scope") == "property_shared":
                    matrix = np.asarray(room.metadata["room_to_property_transform"], dtype=float)
                    transform[:2, :2], transform[:2, 3] = matrix[:2, :2], matrix[:2, 2]
                clouds.append((path, transform))
        values = []
        for path, transform in clouds:
            cloud = o3d.io.read_point_cloud(str(path)); cloud.transform(transform)
            values.append(np.asarray(cloud.points))
        if not values: return None
        points = np.vstack(values)
        if len(points) > maximum: points = points[::int(np.ceil(len(points)/maximum))]
        return points
    except (ImportError, RuntimeError, ValueError):
        return None


def process_cached_openings(output: Path, result: PropertyScanResult, model_dir: Path,
                            config: OpeningConfig | None = None,
                            semantic_detector: SemanticSurfaceDetector | None = None) -> OpeningDetectionResult | None:
    config, output = config or OpeningConfig(), Path(output)
    result.warnings = [warning for warning in result.warnings
        if not (warning.code.startswith("OPENING_") or warning.code == "POSSIBLE_PHANTOM_OPENING")]
    if not config.enabled or not result.property.rooms:
        return None
    frames = load_cached_frames(output, result)
    if not frames:
        result.warnings.append(ResultWarning(code="OPENING_DEPTH_UNAVAILABLE",
            message="No cached calibrated RGB-depth keyframes are available for opening detection."))
        return None
    if semantic_detector is None and not (Path(model_dir)/SEMANTIC_CACHE_NAME).is_dir():
        result.warnings.append(ResultWarning(code="OPENING_MODEL_UNAVAILABLE",
            message="Opening detection skipped; run python scripts/download_models.py --openings."))
        return None
    semantic_detector = semantic_detector or SegFormerSurfaceDetector(model_dir, config)
    structural_points = load_structural_points(output, result)
    detected = OpeningDetector(config, semantic_detector).detect(result.property, frames,
        diagnostics_dir=output/"diagnostics/openings",
        structural_points=structural_points)
    result.property = detected.property; result.warnings.extend(detected.warnings)
    if result.processing_info and "opening_detection" not in result.processing_info.modules_used:
        result.processing_info.modules_used.append("opening_detection")
        result.processing_info.model_versions[config.semantic_model] = config.semantic_revision
        result.processing_info.metadata["openings"] = detected.diagnostics.model_dump(mode="json")
    opening_dir = output/"openings"
    write_json(opening_dir/"openings.json", [item.model_dump(mode="json") for item in result.property.openings])
    write_json(opening_dir/"detection_summary.json", detected.diagnostics.model_dump(mode="json"))
    write_json(opening_dir/"candidates.json", [item.model_dump(mode="json") for item in detected.candidates])
    write_json(opening_dir/"wall_occupancy.json", {
        "geometry_fallback_enabled": config.enable_geometry_fallback,
        "structural_point_count": len(structural_points) if structural_points is not None else 0,
        "geometry_candidates": [item.model_dump(mode="json") for item in detected.candidates if item.frame_id == "geometry"],
        "note": "Wall-local low occupancy is supplemental evidence; semantic-free results remain unknown/open_passage."})
    if structural_points is not None:
        export_wall_occupancy_plots(output/"diagnostics/openings/wall_occupancy",
            result.property, structural_points, config)
    if result.capture.tier.value == "photo":
        for room in result.property.rooms:
            wall_ids = {wall.wall_id for wall in room.walls}
            room_openings = [opening for opening in result.property.openings if opening.wall_id in wall_ids]
            if room.polygon is None or not room.walls:
                continue
            drawing = FloorPlanRenderer().render_room(room, openings=room_openings)
            try: drawing.save(output/"photo"/room.room_id)
            finally: drawing.close()
    renderable = result.property.rooms and all(room.polygon is not None and room.walls for room in result.property.rooms)
    if renderable and (len(result.property.rooms) == 1 or result.property.metadata.get("coordinate_scope") == "property_shared"):
        drawing = FloorPlanRenderer().render_property(result.property)
        try: drawing.save(output)
        finally: drawing.close()
        diagnostic = FloorPlanRenderer(RenderingConfig(show_opening_labels=True)).render_property(result.property)
        try: diagnostic.save_png(output/"diagnostics/openings/fused_openings.png")
        finally: diagnostic.close()
    return detected


def rerun_cached_openings(output: Path, model_dir: Path, config: OpeningConfig | None = None,
                          semantic_detector: SemanticSurfaceDetector | None = None):
    result = load_result(Path(output)/"result.json")
    detected = process_cached_openings(output, result, model_dir, config, semantic_detector)
    save_result(result, Path(output)/"result.json")
    return result, detected


def try_process_cached_openings(output: Path, result: PropertyScanResult, model_dir: Path,
                                config: OpeningConfig | None = None,
                                semantic_detector: SemanticSurfaceDetector | None = None):
    """Best-effort downstream stage: reconstruction artifacts survive model/detection failures."""
    try:
        return process_cached_openings(output, result, model_dir, config, semantic_detector)
    except (PropertyScannerError, ConfigurationError, OSError, RuntimeError, ValueError) as exc:
        result.warnings.append(ResultWarning(code="OPENING_DETECTION_FAILED", message=str(exc)))
        return None
