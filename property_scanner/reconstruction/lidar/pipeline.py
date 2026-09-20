"""Canonical RGB-D reconstruction orchestrated with the existing geometry and renderer."""
from datetime import datetime, timezone
import logging
from pathlib import Path
import platform
import shutil
from time import perf_counter
import numpy as np
import open3d as o3d
from property_scanner.core.exceptions import PropertyScannerError, ProcessingError
from property_scanner.geometry.engine import GeometryEngine
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.schemas.capture import CaptureMetadata
from property_scanner.schemas.common import PreparationResult
from property_scanner.schemas.geometry import PropertyGeometry
from property_scanner.schemas.result import PropertyScanResult, ProcessingInfo, ProcessingIssue, ResultWarning
from property_scanner.schemas.serialization import save_result
from property_scanner.reconstruction.lidar.adapter import CanonicalRGBDAdapter, LidarCaptureAdapter
from property_scanner.reconstruction.lidar.models import LidarConfig
from property_scanner.reconstruction.lidar.fusion import build_keyframes, fuse
from property_scanner.reconstruction.lidar.registration import optimize
from property_scanner.reconstruction.lidar.diagnostics import write_json, save_poses, comparison_metrics, trajectory_plot, floorplan_comparison

logger = logging.getLogger(__name__)
PIPELINE_VERSION = "lidar-rgbd-1.0.0"


def process_lidar(prepared: PreparationResult, config: LidarConfig, adapter: LidarCaptureAdapter | None = None,
                  model_dir: Path = Path("models")) -> PropertyScanResult:
    try:
        return _process(prepared, config, adapter, model_dir)
    except (OSError, RuntimeError) as exc:
        raise ProcessingError(f"LiDAR processing failed; existing artifacts retained: {exc}") from exc


def _process(prepared: PreparationResult, config: LidarConfig, adapter: LidarCaptureAdapter | None,
             model_dir: Path) -> PropertyScanResult:
    started, timer = datetime.now(timezone.utc), perf_counter()
    output = prepared.output_dir
    lidar, ablation, diagnostics = output/"lidar", output/"ablation", output/"diagnostics"/"lidar"
    for folder in (lidar, ablation, diagnostics):
        folder.mkdir(parents=True, exist_ok=True)
    logger.info("Validating RGB-D capture")
    adapter = adapter or CanonicalRGBDAdapter(prepared.capture.source_path)
    adapter.validate()
    manifest = adapter.manifest
    write_json(output/"processing_config.json", {"pipeline_version": PIPELINE_VERSION, "python_version": platform.python_version(),
        "open3d_version": o3d.__version__, "numpy_version": np.__version__, "config": config.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json"), "intrinsics": adapter.intrinsics.model_dump()})
    logger.info("%s frames discovered; building bounded keyframes", manifest.frame_count)
    opening_cache = lidar/"opening_frames"
    frames = build_keyframes(adapter, config, opening_cache)
    logger.info("%s keyframes selected", len(frames))
    warnings = list(adapter.warnings)
    errors = []
    modules = ["canonical_rgbd", "raw_fusion"]
    raw_poses = [f.pose.copy() for f in frames]
    save_poses(lidar/"raw_poses.json", frames, raw_poses)
    logger.info("Creating raw reconstruction")
    raw = fuse(frames, raw_poses, config, lidar/"raw_fused.ply")
    records, optimized_poses, fallback, graph = [], None, 0, None
    if config.drift_correction == "on":
        modules.extend(["icp", "pose_graph", "corrected_fusion"])
        logger.info("Running neighboring ICP, loop validation, and pose-graph optimization")
        optimized_poses, records, graph, fallback = optimize(frames, config)
        accepted = sum(r.kind == "loop" and r.accepted for r in records)
        logger.info("%s loop closures accepted", accepted)
        if not accepted:
            warnings.append(ResultWarning(code="NO_LOOP_CLOSURE", message="No loop closures accepted; only neighbor constraints and explicit device priors were optimized."))
        if fallback:
            warnings.append(ResultWarning(code="REJECTED_NEIGHBOR_ICP", message=f"{fallback} rejected neighbor registrations replaced by weak device-pose priors, not their ICP results."))
        save_poses(lidar/"optimized_poses.json", frames, optimized_poses)
        o3d.io.write_pose_graph(str(diagnostics/"pose_graph.json"), graph)
        logger.info("Creating corrected reconstruction")
        fuse(frames, optimized_poses, config, lidar/"corrected_fused.ply")
    opening_poses = optimized_poses if optimized_poses is not None else raw_poses
    write_json(opening_cache/"frames.json", {"frames": [{
        "frame_id": f"lidar:{frame.frame_id}",
        "image_path": f"lidar/opening_frames/{frame.frame_id:06d}.jpg",
        "depth_path": f"lidar/opening_frames/{frame.frame_id:06d}.npy",
        "intrinsics": adapter.intrinsics.model_dump(), "camera_to_property": pose.tolist(),
        "room_id": "room_01", "depth_source": "sensor", "metadata": {}
        } for frame, pose in zip(frames, opening_poses, strict=True)]})
    write_json(diagnostics/"registrations.json", [r.model_dump() for r in records])
    write_json(diagnostics/"frame_counts.json", adapter.counts.model_dump())
    modes = {"off": raw}
    if optimized_poses is not None:
        modes["on"] = lidar/"corrected_fused.ply"
    geometries = {}
    modules.append("geometry")
    for mode, cloud in modes.items():
        logger.info("Running geometry engine: drift %s", mode)
        try:
            geometry = GeometryEngine(config.geometry).process_point_cloud(cloud, diagnostics_dir=output/"diagnostics"/f"geometry_drift_{mode}")
            geometries[mode] = geometry
            warnings.extend(geometry.warnings)
            if geometry.room_polygon is not None:
                logger.info("Rendering floor plan: drift %s", mode)
                if "floorplan_renderer" not in modules:
                    modules.append("floorplan_renderer")
                drawing = FloorPlanRenderer().render_room(geometry)
                try:
                    drawing.save_png(ablation/f"floorplan_drift_{mode}.png")
                    drawing.save_svg(ablation/f"floorplan_drift_{mode}.svg")
                    warnings.extend(drawing.warnings)
                finally:
                    drawing.close()
            else:
                errors.append(ProcessingIssue(code="POLYGON_UNAVAILABLE", message=f"Drift {mode}: reconstruction retained; no closed room polygon.", module="geometry"))
        except PropertyScannerError as exc:
            errors.append(ProcessingIssue(code="GEOMETRY_OR_RENDERING_FAILED", message=f"Drift {mode}: {exc}", module="geometry_rendering"))
    selected = geometries.get(config.drift_correction)
    drift = {"enabled": config.drift_correction == "on", "keyframes": len(frames),
             "neighbor_edges": sum(r.kind == "neighbor" and r.accepted for r in records), "device_prior_edges": fallback,
             "loop_candidates": sum(r.kind == "loop" for r in records),
             "loop_edges_accepted": sum(r.kind == "loop" and r.accepted for r in records),
             "loop_edges_rejected": sum(r.kind == "loop" and not r.accepted for r in records),
             "loop_edges_retained_after_optimization": sum(e.uncertain for e in graph.edges) if graph is not None else 0}
    comparison = comparison_metrics(frames, optimized_poses, records, config) if optimized_poses is not None else {
        "interpretation": "Internal reconstruction metrics, NOT benchmark accuracy. Drift correction OFF: ICP and optimization not run."}
    comparison.update(drift_correction=drift, frame_counts=adapter.counts.model_dump(),
                      geometry_diagnostics={mode: g.diagnostics.model_dump() for mode, g in geometries.items()},
                      processing_errors=[e.model_dump() for e in errors])
    write_json(ablation/"drift_comparison.json", comparison)
    trajectory_plot(ablation/"trajectory_comparison.png", frames, optimized_poses)
    if optimized_poses is not None:
        floorplan_comparison(ablation/"drift_floorplan_comparison.png", ablation/"floorplan_drift_off.png", ablation/"floorplan_drift_on.png")
    for name in ("raw_fused.ply", "raw_poses.json", "corrected_fused.ply", "optimized_poses.json"):
        if (lidar/name).exists():
            shutil.copy2(lidar/name, ablation/name)
    for suffix in ("png", "svg"):
        source = ablation/f"floorplan_drift_{config.drift_correction}.{suffix}"
        if source.exists():
            shutil.copy2(source, output/f"floorplan.{suffix}")
    result = PropertyScanResult(
        capture=CaptureMetadata(capture_id=prepared.capture.capture_id, tier="lidar", source_type="directory",
            source_reference=f"capture:{manifest.capture_id}", device_model=manifest.device_model,
            capture_timestamp=manifest.timestamp, processing_timestamp=started,
            metadata={"source_capture_id": str(manifest.capture_id), "source_app": manifest.source_app}),
        property=PropertyGeometry(property_id=f"property:{manifest.capture_id}",
            rooms=[selected.to_room("room_01")] if selected else [], total_floor_area=selected.floor_area if selected else None),
        warnings=warnings, processing_info=ProcessingInfo(pipeline_version=PIPELINE_VERSION, started_at=started,
            completed_at=datetime.now(timezone.utc), processing_seconds=perf_counter()-timer,
            modules_used=modules,
            errors=errors, metadata={"drift_correction": drift, "frame_counts": adapter.counts.model_dump(),
                                   "python_version": platform.python_version(), "open3d_version": o3d.__version__}),
        metadata={"synthetic": manifest.metadata.get("synthetic", False), "source_to_floor_transform": selected.source_to_floor_transform if selected else None},
    )
    from property_scanner.openings.cached import try_process_cached_openings
    try_process_cached_openings(output, result, model_dir, config.opening)
    save_result(result, output/"result.json")
    return result
