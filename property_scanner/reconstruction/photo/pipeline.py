"""Sequential room photo reconstruction with shared metric geometry components."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import platform
from time import perf_counter
from typing import Callable

import numpy as np
import open3d as o3d
from PIL import Image

from property_scanner.core.exceptions import PropertyScannerError, ProcessingError
from property_scanner.geometry.engine import GeometryEngine
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.schemas.capture import CaptureMetadata
from property_scanner.schemas.common import PreparationResult
from property_scanner.schemas.geometry import PropertyGeometry, Room
from property_scanner.schemas.measurements import AreaMeasurement
from property_scanner.schemas.result import PropertyScanResult, ProcessingInfo, ProcessingIssue, ResultWarning
from property_scanner.schemas.serialization import save_result
from property_scanner.reconstruction.lidar.diagnostics import write_json, save_poses, trajectory_plot
from property_scanner.reconstruction.lidar.fusion import fuse
from property_scanner.reconstruction.lidar.registration import optimize
from property_scanner.reconstruction.photo.images import discover_rooms, prepare_room_images
from property_scanner.reconstruction.photo.models import PhotoConfig, PreparedRoomImages, RoomSource
from property_scanner.reconstruction.video.depth import DepthAnythingMetric, MetricDepthEstimator
from property_scanner.reconstruction.video.models import ScaleReport
from property_scanner.reconstruction.video.orientation import canonicalize
from property_scanner.reconstruction.video.pipeline import metric_keyframe
from property_scanner.reconstruction.video.scale import estimate_scale
from property_scanner.reconstruction.video.sfm import ColmapReconstructor, SfMReconstructor
from property_scanner.stitching.diagnostics import export_stitching_diagnostics, room_scale_quality
from property_scanner.stitching.engine import MultiRoomStitcher
from property_scanner.stitching.evidence import collect_cached_photo_evidence

logger = logging.getLogger(__name__)
VERSION = "photo-metric-1.0.0"


def _warning(code: str, message: str, room_id: str) -> ResultWarning:
    return ResultWarning(code=code, message=message, room_id=room_id)


def _process_room(
    room: RoomSource,
    room_output: Path,
    diagnostics: Path,
    config: PhotoConfig,
    reconstructor: SfMReconstructor,
    estimator: MetricDepthEstimator,
) -> tuple[Room, list[ResultWarning], dict]:
    warnings: list[ResultWarning] = []
    stage = "IMAGE_VALIDATION_FAILED"
    prepared: PreparedRoomImages | None = None
    try:
        prepared = prepare_room_images(room, room_output, config)
        logger.info("%s photos validated and selected", len(prepared.selected))
        messages = {
            "LOW_PHOTO_COUNT": "Only two usable photos remain; reconstruction and scale may be fragile.",
            "EXCESSIVE_BLUR": "One or more blurred photos were excluded.",
            "LOW_LIGHT": "One or more low-light photos were excluded.",
            "REDUNDANT_PHOTOS": "One or more near-duplicate photos were excluded.",
        }
        warnings.extend(_warning(code, messages[code], room.room_id) for code in prepared.warning_codes)

        stage = "INSUFFICIENT_PHOTO_OVERLAP"
        room_config = config.model_copy(update={"single_camera": prepared.same_camera_supported})
        sfm = reconstructor.reconstruct(prepared.selected_directory, room_output / "sfm", room_config)
        logger.info("%s/%s cameras registered", len(sfm.frames), len(prepared.selected))
        if len(sfm.frames) < config.min_registered_frames:
            raise ProcessingError("COLMAP registered too few room photos")
        if len(sfm.frames) == 2:
            warnings.append(_warning("LOW_VIEW_COUNT", "Only two cameras registered; reconstruction is fragile.", room.room_id))
        if len(sfm.frames) < len(prepared.selected):
            warnings.append(_warning("LOW_IMAGE_OVERLAP",
                f"COLMAP registered {len(sfm.frames)}/{len(prepared.selected)} selected photos.", room.room_id))
        write_json(room_output / "feature_matching.json", {
            key: sfm.statistics.get(key) for key in ("features_per_image", "matched_image_pairs", "inlier_matches")})
        relative = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray([point.xyz for point in sfm.points], dtype=float).reshape(-1, 3)))
        o3d.io.write_point_cloud(str(room_output / "sfm/sparse_relative.ply"), relative)

        stage = "DEPTH_ESTIMATION_FAILED"
        depth_directory = room_output / "depth"
        depth_directory.mkdir(exist_ok=True)
        depth_paths = {}
        for frame in sfm.frames:
            try:
                with Image.open(prepared.selected_directory / frame.filename) as image:
                    depth = np.asarray(estimator.predict(image.convert("RGB")), dtype=np.float32)
                if depth.shape != (frame.intrinsics.height, frame.intrinsics.width):
                    raise ProcessingError("Metric depth shape differs from registered photo")
                path = depth_directory / f"{Path(frame.filename).stem}.npy"
                np.save(path, depth)
                depth_paths[frame.filename] = path
            except (PropertyScannerError, OSError, RuntimeError, ValueError) as exc:
                warnings.append(_warning("DEPTH_IMAGE_SKIPPED", f"{frame.filename}: {exc}", room.room_id))

        stage = "METRIC_SCALE_UNRESOLVED"
        scale: ScaleReport = estimate_scale(sfm, depth_paths, room_config)
        write_json(room_output / "scale_estimation.json", scale.model_dump())
        if not scale.metric_scale_resolved:
            raise ProcessingError(scale.failure_reason or "Reliable metric scale could not be established")
        warnings.append(_warning("LEARNED_METRIC_SCALE",
            "Metric scale comes from learned monocular depth; consistency is not benchmark accuracy.", room.room_id))
        logger.info("Metric scale recovered")
        relative.scale(scale.global_scale, center=(0, 0, 0))
        o3d.io.write_point_cloud(str(room_output / "sfm/sparse_metric.ply"), relative)

        stage = "POINT_CLOUD_FUSION_FAILED"
        keyframes = []
        for frame in sfm.frames:
            if frame.filename not in scale.frames_used:
                continue
            with Image.open(prepared.selected_directory / frame.filename) as image:
                keyframes.append(metric_keyframe(frame, image,
                    np.load(depth_paths[frame.filename], allow_pickle=False), scale.global_scale,
                    room_config, int(Path(frame.filename).stem)))
        if len(keyframes) < config.min_registered_frames:
            raise ProcessingError("Too few scale-consistent photo clouds for room fusion")
        initial = [frame.pose.copy() for frame in keyframes]
        save_poses(room_output / "initial_poses.json", keyframes, initial,
                   coordinate_system="sfm_metric_arbitrary_orientation")
        poses, registrations = initial, []
        if config.enable_icp_refinement:
            registration_config = config.registration
            if not config.enable_loop_closure:
                registration_config = registration_config.model_copy(
                    update={"loop_min_frame_separation": len(keyframes) + 1})
            poses, registrations, graph, fallback = optimize(keyframes, registration_config)
            o3d.io.write_pose_graph(str(diagnostics / "pose_graph.json"), graph)
            if fallback:
                warnings.append(_warning("ICP_REFINEMENT_REJECTED",
                    f"{fallback} rejected registrations retained weak COLMAP priors.", room.room_id))
        write_json(room_output / "registration.json", [record.model_dump() for record in registrations])
        save_poses(room_output / "optimized_poses.json", keyframes, poses,
                   coordinate_system="sfm_metric_arbitrary_orientation")
        trajectory_plot(diagnostics / "trajectory.png", keyframes, poses)
        fuse(keyframes, poses, config.registration, room_output / "room_metric_sfm.ply")

        stage = "ORIENTATION_UNRESOLVED"
        cloud = o3d.io.read_point_cloud(str(room_output / "room_metric_sfm.ply"))
        if config.registration.remove_frame_outliers and len(cloud.points) > config.registration.statistical_nb_neighbors:
            cloud, _ = cloud.remove_statistical_outlier(
                config.registration.statistical_nb_neighbors, config.registration.statistical_std_ratio)
        cloud, transform = canonicalize(cloud, poses, room_config)
        write_json(room_output / "canonical_transform.json", {
            "sfm_metric_to_room_local_z_up": transform.tolist(),
            "north_or_property_orientation_known": False,
        })
        if not o3d.io.write_point_cloud(str(room_output / "room_fused.ply"), cloud):
            raise ProcessingError("Cannot save canonical room point cloud")
        save_poses(room_output / "canonical_poses.json", keyframes, [transform @ pose for pose in poses])
        logger.info("Room point cloud generated")

        stage = "GEOMETRY_EXTRACTION_FAILED"
        geometry = GeometryEngine(config.registration.geometry).process_point_cloud(
            room_output / "room_fused.ply", diagnostics_dir=diagnostics / "geometry")
        write_json(room_output / "geometry.json", geometry.model_dump(mode="json"))
        write_json(room_output / "geometry_diagnostics.json", geometry.diagnostics.model_dump())
        warnings.extend(warning.model_copy(update={"room_id": room.room_id}) for warning in geometry.warnings)
        if geometry.ceiling_height is None:
            warnings.append(_warning("CEILING_NOT_DETECTED", "Ceiling height is unavailable for this room.", room.room_id))
        if geometry.room_polygon is None:
            raise ProcessingError("Geometry engine did not produce a closed room polygon")
        room_model = geometry.to_room(room.room_id)
        room_model.name = room.label
        room_model.metadata.update({"coordinate_scope": "room_local_unstitched",
                                    "source_folder": room.directory.name})
        write_json(room_output / "room.json", room_model.model_dump(mode="json"))
        logger.info("Geometry extracted")

        stage = "FLOORPLAN_RENDERING_FAILED"
        drawing = FloorPlanRenderer().render_room(geometry, room_id=room.room_id, name=room.label)
        try:
            drawing.save(room_output)
            warnings.extend(warning.model_copy(update={"room_id": room.room_id}) for warning in drawing.warnings)
        finally:
            drawing.close()
        logger.info("Floor plan generated")
        report = {"status": "success", "images": {
            "total": prepared.images_total, "valid": prepared.images_valid,
            "selected": len(prepared.selected), "rejected": prepared.images_rejected},
            "registered_images": len(sfm.frames), "scale": scale.model_dump(),
            "geometry_diagnostics": geometry.diagnostics.model_dump()}
        write_json(diagnostics / "room_status.json", report)
        return room_model, warnings, report
    except (PropertyScannerError, OSError, RuntimeError, ValueError) as exc:
        report = {"status": "failed", "error_code": stage, "message": str(exc),
                  "images_total": len(room.images),
                  "images_selected": len(prepared.selected) if prepared else 0}
        write_json(diagnostics / "room_status.json", report)
        raise ProcessingError(f"{stage}: {exc}") from exc


def process_photo(
    prepared: PreparationResult,
    config: PhotoConfig,
    model_dir: Path,
    *,
    reconstructor_factory: Callable[[], SfMReconstructor] | None = None,
    depth_estimator: MetricDepthEstimator | None = None,
) -> PropertyScanResult:
    """Process room folders sequentially and retain successful partial results."""
    started, timer = datetime.now(timezone.utc), perf_counter()
    output = prepared.output_dir
    photo_root, diagnostics_root = output / "photo", output / "diagnostics/photo"
    photo_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    rooms = discover_rooms(prepared.capture.source_path)
    logger.info("Property contains %s room folders", len(rooms))
    write_json(output / "processing_config.json", {
        "pipeline_version": VERSION, "python_version": platform.python_version(),
        "open3d_version": o3d.__version__, "config": config.model_dump(mode="json"),
        "coordinate_scope": "each room is local and unstitched",
        "rooms": [{"room_id": room.room_id, "folder": room.directory.name} for room in rooms],
    })
    estimator = depth_estimator
    successful: list[Room] = []
    warnings: list[ResultWarning] = []
    errors: list[ProcessingIssue] = []
    reports = {}
    modules = ["photo_discovery", "image_quality"]
    for room in rooms:
        logger.info("Processing room: %s", room.room_id)
        room_output, room_diagnostics = photo_root / room.room_id, diagnostics_root / room.room_id
        room_output.mkdir(parents=True, exist_ok=True)
        room_diagnostics.mkdir(parents=True, exist_ok=True)
        try:
            if estimator is None:
                estimator = DepthAnythingMetric(model_dir, config)
            reconstructor = reconstructor_factory() if reconstructor_factory else ColmapReconstructor()
            room_model, room_warnings, report = _process_room(
                room, room_output, room_diagnostics, config, reconstructor, estimator)
            successful.append(room_model)
            warnings.extend(room_warnings)
            reports[room.room_id] = report
        except (PropertyScannerError, OSError, RuntimeError, ValueError) as exc:
            text = str(exc)
            code = text.split(":", 1)[0] if ":" in text else "ROOM_RECONSTRUCTION_FAILED"
            if code not in {"IMAGE_VALIDATION_FAILED", "INSUFFICIENT_PHOTO_OVERLAP", "DEPTH_ESTIMATION_FAILED",
                            "METRIC_SCALE_UNRESOLVED", "POINT_CLOUD_FUSION_FAILED", "ORIENTATION_UNRESOLVED",
                            "GEOMETRY_EXTRACTION_FAILED", "FLOORPLAN_RENDERING_FAILED"}:
                code = "ROOM_RECONSTRUCTION_FAILED"
            errors.append(ProcessingIssue(code=code, message=f"Room {room.room_id}: {text}",
                                          module=f"photo:{room.room_id}"))
            reports[room.room_id] = {"status": "failed", "error_code": code, "message": text}
            logger.warning("Room %s failed: %s", room.room_id, text)

    if successful:
        modules.extend(["colmap_sfm", "metric_depth", "robust_metric_scale",
                        "shared_metric_fusion", "geometry", "floorplan_renderer"])
        if config.enable_icp_refinement:
            modules.append("shared_icp_pose_graph")
    areas = [room.floor.area.value for room in successful if room.floor and room.floor.area]
    total_area = AreaMeasurement(value=sum(areas), method="sum of independent unstitched room polygons") if areas else None
    summary = {"rooms_discovered": len(rooms), "rooms_successful": len(successful),
               "rooms_failed": len(rooms)-len(successful), "rooms": reports,
               "coordinate_scope": "room_local_unstitched", "stitching_performed": False}
    write_json(output / "photo_summary.json", summary)
    result = PropertyScanResult(
        capture=CaptureMetadata(capture_id=prepared.capture.capture_id, tier="photo", source_type="directory",
            source_reference=prepared.capture.source_path.name, processing_timestamp=started,
            metadata={"room_folder_count": len(rooms)}),
        property=PropertyGeometry(property_id=f"property:{prepared.capture.capture_id}", rooms=successful,
            total_floor_area=total_area, metadata={"coordinate_scope": "room_local_unstitched",
                                                   "stitching_performed": False}),
        warnings=warnings,
        processing_info=ProcessingInfo(pipeline_version=VERSION, started_at=started,
            completed_at=datetime.now(timezone.utc), processing_seconds=perf_counter()-timer,
            modules_used=modules, errors=errors,
            model_versions={config.depth_model: config.depth_revision} if estimator is not None else {},
            metadata=summary),
        metadata={"room_coordinates_are_local": True, "multi_room_stitching_implemented": False},
    )
    if config.enable_property_stitching and len(successful) > 1:
        logger.info("Evaluating cross-room stitching evidence")
        candidates = collect_cached_photo_evidence(output, [room.room_id for room in successful], config.stitching)
        stitched = MultiRoomStitcher(config.stitching).stitch(result.property.property_id, successful, candidates)
        export_stitching_diagnostics(output / "stitching", successful, candidates, stitched)
        scale_report, inconsistent_scales = room_scale_quality(
            output, [room.room_id for room in successful], config.stitching)
        write_json(output / "stitching/room_scale_consistency.json", scale_report)
        if inconsistent_scales:
            stitched.warnings.append(ResultWarning(code="ROOM_SCALE_INCONSISTENCY",
                message=f"Weak independent metric-scale consistency for: {', '.join(inconsistent_scales)}."))
        result.warnings.extend(stitched.warnings)
        result.processing_info.modules_used.append("multi_room_stitching")
        result.processing_info.metadata["stitching"] = stitched.diagnostics.model_dump(mode="json")
        result.metadata["multi_room_stitching_implemented"] = True
        if not any(item.relative_transform is not None for item in candidates):
            result.warnings.append(ResultWarning(code="NO_CROSS_ROOM_MATCHES",
                message="No geometrically verified metric cross-room correspondence was found."))
        if stitched.diagnostics.valid_layout:
            result.property = stitched.property
            result.metadata["room_coordinates_are_local"] = False
            drawing = FloorPlanRenderer().render_property(result.property, title="Property floor plan")
            try:
                drawing.save(output)
                result.warnings.extend(drawing.warnings)
            finally:
                drawing.close()
            summary.update({"coordinate_scope": "property_shared", "stitching_performed": True,
                            "stitching_valid": True})
        else:
            summary.update({"stitching_performed": True, "stitching_valid": False})
        write_json(output / "photo_summary.json", summary)
    save_result(result, output / "result.json")
    return result
