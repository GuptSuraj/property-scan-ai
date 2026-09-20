"""Bounded keyframe construction and incremental voxel fusion."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from property_scanner.core.exceptions import ProcessingError
from property_scanner.reconstruction.lidar.adapter import RGBDFrame, LidarCaptureAdapter
from property_scanner.reconstruction.lidar.models import CameraIntrinsics, LidarConfig
from property_scanner.reconstruction.lidar.poses import rotation_degrees


@dataclass
class Keyframe:
    frame_id: int
    cloud: o3d.geometry.PointCloud
    pose: np.ndarray


def frame_cloud(frame: RGBDFrame, intrinsics: CameraIntrinsics, config: LidarConfig) -> o3d.geometry.PointCloud:
    camera = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(frame.rgb)), o3d.geometry.Image(np.ascontiguousarray(frame.depth_m)),
        depth_scale=1.0, depth_trunc=config.depth_max_m, convert_rgb_to_intensity=False)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera).voxel_down_sample(config.frame_voxel_size)
    if len(cloud.points) > config.max_points_per_frame:
        step = int(np.ceil(len(cloud.points)/config.max_points_per_frame))
        cloud = cloud.uniform_down_sample(step)
    if config.remove_frame_outliers and len(cloud.points) > config.statistical_nb_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(config.statistical_nb_neighbors, config.statistical_std_ratio)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=config.normal_radius, max_nn=config.normal_max_nn))
    cloud.normalize_normals()
    return cloud


def build_keyframes(adapter: LidarCaptureAdapter, config: LidarConfig, cache_dir: Path | None = None) -> list[Keyframe]:
    frames = []
    sampled_ids = {frame.frame_id for frame in adapter.manifest.frames[::config.frame_stride]}
    for frame in adapter.load_frames(config):
        selected = frame.frame_id in sampled_ids and len(frames) < config.max_frames
        if selected and frames:
            delta = np.linalg.inv(frames[-1].pose) @ frame.pose
            selected = (np.linalg.norm(delta[:3, 3]) >= config.min_translation_between_keyframes
                        or rotation_degrees(delta[:3, :3]) >= config.min_rotation_between_keyframes)
        if not selected:
            adapter.counts.frames_not_selected += 1
            continue
        cloud = frame_cloud(frame, adapter.intrinsics, config)
        if len(cloud.points) < 20:
            adapter.counts.frames_not_selected += 1
            continue
        frames.append(Keyframe(frame.frame_id, cloud, frame.pose.copy()))
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.rgb).save(cache_dir/f"{frame.frame_id:06d}.jpg", quality=95)
            np.save(cache_dir/f"{frame.frame_id:06d}.npy", frame.depth_m.astype(np.float32))
    adapter.counts.frames_skipped = adapter.counts.frames_total - len(frames)
    if len(frames) < config.min_keyframes:
        raise ProcessingError(f"Only {len(frames)} usable keyframes; need {config.min_keyframes}")
    return frames


def fuse(frames: list[Keyframe], poses: list[np.ndarray], config: LidarConfig, path: Path) -> Path:
    combined = o3d.geometry.PointCloud()
    for frame, pose in zip(frames, poses, strict=True):
        transformed = o3d.geometry.PointCloud(frame.cloud)
        transformed.transform(pose)
        combined += transformed
        combined = combined.voxel_down_sample(config.fusion_voxel_size)
        if len(combined.points) > config.max_fused_points:
            raise ProcessingError("Fusion point budget exceeded; increase fusion_voxel_size")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), combined):
        raise ProcessingError(f"Cannot save fused cloud: {path.name}")
    return path
