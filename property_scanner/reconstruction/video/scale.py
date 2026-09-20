"""Robust cross-frame calibration of arbitrary SfM units against predicted metric depth."""
from pathlib import Path
import numpy as np
from property_scanner.reconstruction.video.models import SparseReconstruction, ScaleReport, VideoConfig


def estimate_scale(sfm: SparseReconstruction, depths: dict[str, Path], config: VideoConfig) -> ScaleReport:
    points = {p.point_id: p for p in sfm.points}
    report = ScaleReport()
    ratios_by_frame = {}
    for frame in sfm.frames:
        if frame.filename not in depths:
            continue
        depth = np.load(depths[frame.filename], allow_pickle=False)
        pose = np.array(frame.camera_to_world)
        ratios = []
        k = frame.intrinsics
        for observation in frame.observations:
            report.correspondences_total += 1
            point = points.get(observation.point_id)
            if point is None or point.reprojection_error > config.scale_max_reprojection_error:
                continue
            camera = pose[:3, :3].T @ (np.array(point.xyz)-pose[:3, 3])
            if not np.isfinite(camera).all() or camera[2] <= 0:
                continue
            u, v = k.fx*camera[0]/camera[2]+k.cx, k.fy*camera[1]/camera[2]+k.cy
            border = config.scale_image_border
            if not (border <= u < k.width-border and border <= v < k.height-border):
                continue
            if np.hypot(u-observation.x, v-observation.y) > config.scale_max_reprojection_error:
                continue
            pixel_x, pixel_y = int(round(u)), int(round(v))
            if not (0 <= pixel_x < depth.shape[1] and 0 <= pixel_y < depth.shape[0]):
                continue
            predicted = float(depth[pixel_y, pixel_x])
            if not np.isfinite(predicted) or not config.depth_min_m <= predicted <= config.depth_max_m:
                continue
            ratios.append(predicted/camera[2])
        if len(ratios) >= config.scale_min_frame_correspondences:
            array = np.array(ratios)
            median = float(np.median(array))
            mad = float(np.median(abs(array-median)))
            array = array[abs(array-median) <= max(config.scale_relative_floor*median, config.scale_outlier_threshold*1.4826*mad)]
            if len(array) >= config.scale_min_frame_correspondences:
                ratios_by_frame[frame.filename] = array
                report.frame_scales[frame.filename] = float(np.median(array))
    if len(ratios_by_frame) < config.scale_min_frames:
        report.failure_reason = "Insufficient frames with reliable depth/SfM correspondences"
        return report
    center = float(np.median(list(report.frame_scales.values())))
    report.frame_scale_median = center
    accepted = {name: ratios for name, ratios in ratios_by_frame.items()
                if abs(report.frame_scales[name]/center-1) <= config.scale_frame_consistency_threshold}
    report.rejected_frames = sorted(set(ratios_by_frame)-set(accepted))
    if len(accepted) < config.scale_min_frames:
        report.failure_reason = "Frame scale estimates are inconsistent"
        return report
    ratios = np.concatenate(list(accepted.values()))
    # Equal frame weighting avoids one feature-rich wall determining the whole scale.
    scale = float(np.median([np.median(r) for r in accepted.values()]))
    mad = float(np.median(abs(ratios-scale)))
    bound = max(config.scale_relative_floor*scale, config.scale_outlier_threshold*1.4826*mad)
    kept = ratios[abs(ratios-scale) <= bound]
    report.correspondences_used = len(kept)
    report.scale_mad = mad
    if len(kept) < config.scale_min_correspondences or mad/scale > config.scale_max_relative_mad:
        report.failure_reason = "Insufficient consistent scale support or excessive relative dispersion"
        return report
    report.metric_scale_resolved = True
    report.global_scale = scale
    report.frames_used = sorted(accepted)
    report.scale_confidence_quality = float(1/(1+mad/scale))
    return report


def scaled_pose(pose: list[list[float]], scale: float) -> np.ndarray:
    result = np.array(pose, dtype=float).copy()
    result[:3, 3] *= scale
    return result
