"""Cross-room SIFT evidence and metric RGB-D correspondence alignment."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import cv2
import numpy as np

from property_scanner.stitching.models import RoomConnectionEvidence, StitchingConfig


def candidate_room_pairs(room_ids: list[str], maximum: int) -> list[tuple[str, str]]:
    """Deterministic small-property strategy, bounded for future larger captures."""
    return list(combinations(sorted(room_ids), 2))[:maximum]


def _room_frames(room_dir: Path) -> dict[str, dict]:
    reconstruction = json.loads((room_dir / "sfm/reconstruction.json").read_text())
    pose_data = json.loads((room_dir / "canonical_poses.json").read_text())
    poses = {int(item["frame_id"]): np.asarray(item["matrix"], dtype=float) for item in pose_data["poses"]}
    result = {}
    for frame in reconstruction["frames"]:
        name = frame["filename"]
        frame_id = int(Path(name).stem)
        if frame_id not in poses:
            continue
        result[name] = {"image": room_dir / "selected_images" / name,
            "depth": room_dir / "depth" / f"{Path(name).stem}.npy",
            "intrinsics": frame["intrinsics"], "pose": poses[frame_id]}
    return result


def _features(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode stitching image: {path}")
    detector = cv2.SIFT_create(nfeatures=5000)
    points, descriptors = detector.detectAndCompute(image, None)
    return image.shape, points, descriptors


def _backproject(point, depth: np.ndarray, intrinsics: dict) -> np.ndarray | None:
    u, v = point
    col, row = int(round(u)), int(round(v))
    if row < 0 or col < 0 or row >= depth.shape[0] or col >= depth.shape[1]:
        return None
    z = float(depth[row, col])
    if not np.isfinite(z) or z <= 0:
        return None
    return np.array([(u-intrinsics["cx"])/intrinsics["fx"]*z,
                     (v-intrinsics["cy"])/intrinsics["fy"]*z, z, 1.0])


def _fit_se2(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    covariance = (source-source_center).T @ (target-target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    result = np.eye(3)
    result[:2, :2], result[:2, 2] = rotation, translation
    return result


def robust_se2(source_b: np.ndarray, target_a: np.ndarray, config: StitchingConfig,
               seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC fixed-scale transform from room B XY points into room A."""
    count = len(source_b)
    if count < 3:
        return None, np.zeros(count, dtype=bool)
    rng, best = np.random.default_rng(seed), np.zeros(count, dtype=bool)
    for _ in range(config.correspondence_ransac_iterations):
        sample = rng.choice(count, 2, replace=False)
        if np.linalg.norm(source_b[sample[0]]-source_b[sample[1]]) < 0.05:
            continue
        transform = _fit_se2(source_b[sample], target_a[sample])
        predicted = (transform[:2, :2] @ source_b.T).T + transform[:2, 2]
        inliers = np.linalg.norm(predicted-target_a, axis=1) <= config.correspondence_ransac_threshold_m
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < config.cross_room_min_inliers:
        return None, best
    transform = _fit_se2(source_b[best], target_a[best])
    predicted = (transform[:2, :2] @ source_b.T).T + transform[:2, 2]
    best = np.linalg.norm(predicted-target_a, axis=1) <= config.correspondence_ransac_threshold_m
    return transform, best


def _refine_clouds(room_a_dir: Path, room_b_dir: Path, initial: np.ndarray,
                   config: StitchingConfig):
    """Conservatively refine an already supported visual transform with local ICP."""
    if not config.enable_point_cloud_refinement:
        return initial, None, None
    try:
        import open3d as o3d
        target = o3d.io.read_point_cloud(str(room_a_dir / "room_fused.ply")).voxel_down_sample(0.08)
        source = o3d.io.read_point_cloud(str(room_b_dir / "room_fused.ply")).voxel_down_sample(0.08)
        if len(target.points) < 100 or len(source.points) < 100:
            return initial, None, None
        radius = config.registration_max_distance * 2
        for cloud in (source, target):
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        transform = np.eye(4)
        transform[:2, :2], transform[:2, 3] = initial[:2, :2], initial[:2, 2]
        fitted = o3d.pipelines.registration.registration_icp(source, target,
            config.registration_max_distance, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        rotation = fitted.transformation[:3, :3]
        if fitted.fitness < config.registration_min_fitness or fitted.inlier_rmse > config.registration_max_rmse:
            return initial, float(fitted.fitness), float(fitted.inlier_rmse)
        # Preserve the planar model; ICP may only refine XY and yaw.
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        refined = np.array([[np.cos(yaw), -np.sin(yaw), fitted.transformation[0, 3]],
                            [np.sin(yaw), np.cos(yaw), fitted.transformation[1, 3]], [0, 0, 1.]])
        return refined, float(fitted.fitness), float(fitted.inlier_rmse)
    except (ImportError, RuntimeError):
        return initial, None, None


def _match_frame_pair(frame_a: dict, frame_b: dict, config: StitchingConfig, seed: int):
    _, keys_a, desc_a = _features(frame_a["image"])
    _, keys_b, desc_b = _features(frame_b["image"])
    if desc_a is None or desc_b is None:
        return 0, np.zeros(0, dtype=bool), None, np.zeros(0, dtype=bool)
    nearest = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    matches = [first for first, second in nearest if first.distance < config.feature_ratio_test*second.distance]
    if len(matches) < 8:
        return len(matches), np.zeros(len(matches), dtype=bool), None, np.zeros(0, dtype=bool)
    points_a = np.float32([keys_a[item.queryIdx].pt for item in matches])
    points_b = np.float32([keys_b[item.trainIdx].pt for item in matches])
    _, mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC, 1.5, 0.995, 2000)
    geometric = np.zeros(len(matches), dtype=bool) if mask is None else mask.ravel().astype(bool)
    if geometric.sum() < config.cross_room_min_inliers:
        # Transition views can be dominated by one doorway/wall plane; a RANSAC
        # homography is still geometric verification, not a raw descriptor count.
        _, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 2.0,
                                     maxIters=2000, confidence=0.995)
        geometric = np.zeros(len(matches), dtype=bool) if mask is None else mask.ravel().astype(bool)
    if geometric.sum() < config.cross_room_min_inliers:
        return len(matches), geometric, None, np.zeros(0, dtype=bool)
    depth_a = np.load(frame_a["depth"], allow_pickle=False)
    depth_b = np.load(frame_b["depth"], allow_pickle=False)
    world_a, world_b = [], []
    for index in np.flatnonzero(geometric):
        a = _backproject(points_a[index], depth_a, frame_a["intrinsics"])
        b = _backproject(points_b[index], depth_b, frame_b["intrinsics"])
        if a is None or b is None:
            continue
        world_a.append((frame_a["pose"] @ a)[:3])
        world_b.append((frame_b["pose"] @ b)[:3])
    if len(world_a) < config.cross_room_min_inliers:
        return len(matches), geometric, None, np.zeros(len(world_a), dtype=bool)
    array_a, array_b = np.asarray(world_a), np.asarray(world_b)
    transform, metric_inliers = robust_se2(array_b[:, :2], array_a[:, :2], config, seed)
    if transform is not None:
        # Floor-normalized rooms should also agree vertically at matched scene points.
        metric_inliers &= np.abs(array_a[:, 2]-array_b[:, 2]) <= config.correspondence_ransac_threshold_m*2
        if metric_inliers.sum() < config.cross_room_min_inliers:
            transform = None
    return len(matches), geometric, transform, metric_inliers


def collect_cached_photo_evidence(output_dir: Path, room_ids: list[str],
                                  config: StitchingConfig) -> list[RoomConnectionEvidence]:
    """Use saved normalized images, metric depths and canonical poses; reconstruction is not rerun."""
    frames = {}
    for room_id in room_ids:
        try:
            frames[room_id] = _room_frames(Path(output_dir) / "photo" / room_id)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            frames[room_id] = {}
    evidence = []
    for pair_index, (room_a, room_b) in enumerate(candidate_room_pairs(room_ids, config.max_candidate_room_pairs)):
        best = None
        for name_a, frame_a in frames[room_a].items():
            if not frame_a["depth"].is_file():
                continue
            for name_b, frame_b in frames[room_b].items():
                if not frame_b["depth"].is_file():
                    continue
                try:
                    raw, visual_mask, transform, metric_mask = _match_frame_pair(
                        frame_a, frame_b, config, pair_index*1009+len(evidence))
                except (OSError, ValueError, cv2.error):
                    continue
                visual_inliers = int(visual_mask.sum())
                ratio = visual_inliers/max(raw, 1)
                metric_count = int(metric_mask.sum())
                visual_score = min(1.0, visual_inliers/max(config.cross_room_min_inliers*2, 1))*ratio
                geometry_score = metric_count/max(visual_inliers, 1)
                combined = 0.45*visual_score + 0.55*geometry_score
                record = RoomConnectionEvidence(connection_id=f"connection:{room_a}:{room_b}",
                    room_a_id=room_a, room_b_id=room_b, image_a=name_a, image_b=name_b,
                    raw_match_count=raw, inlier_match_count=visual_inliers, inlier_ratio=ratio,
                    relative_transform=transform.tolist() if transform is not None else None,
                    visual_score=visual_score, geometry_score=geometry_score,
                    combined_score=combined, accepted=transform is not None and
                        raw >= config.cross_room_min_matches and ratio >= config.cross_room_min_inlier_ratio and
                        combined >= config.connection_acceptance_threshold,
                    rejection_reason=None if transform is not None else "metric_3d_correspondences_unavailable",
                    metadata={"metric_correspondence_inliers": metric_count,
                              "transform_direction": "room_b_to_room_a"})
                if best is None or record.combined_score > best.combined_score:
                    best = record
        if best is None:
            best = RoomConnectionEvidence(connection_id=f"connection:{room_a}:{room_b}",
                room_a_id=room_a, room_b_id=room_b, rejection_reason="NO_CROSS_ROOM_MATCHES")
        if best.relative_transform is not None:
            refined, fitness, rmse = _refine_clouds(Path(output_dir)/"photo"/room_a,
                Path(output_dir)/"photo"/room_b, np.asarray(best.relative_transform), config)
            geometry_score = best.geometry_score
            if fitness is not None:
                geometry_score = max(geometry_score, fitness)
            combined = 0.45*best.visual_score + 0.55*geometry_score
            accepted = (best.raw_match_count >= config.cross_room_min_matches and
                best.inlier_match_count >= config.cross_room_min_inliers and
                best.inlier_ratio >= config.cross_room_min_inlier_ratio and
                combined >= config.connection_acceptance_threshold)
            best = best.model_copy(update={"relative_transform": refined.tolist(),
                "point_cloud_registration_fitness": fitness,
                "point_cloud_registration_rmse": rmse, "geometry_score": geometry_score,
                "combined_score": combined, "accepted": accepted,
                "rejection_reason": None if accepted else "weak_connection_evidence"})
        evidence.append(best)
    return evidence
