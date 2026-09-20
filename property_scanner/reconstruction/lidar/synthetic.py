"""Small deterministic ray-cast RGB-D room fixture; not real-world accuracy evidence."""
from pathlib import Path
import json
from uuid import UUID
import numpy as np
from PIL import Image


def generate_capture(destination: Path, *, drift: bool = True, frames: int = 25) -> Path:
    """Render metric pinhole depth using true poses, then perturb only exported device poses."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/"manifest.json").exists():
        raise ValueError("Destination already contains a capture; choose a fresh directory")
    for name in ("rgb", "depth"):
        (destination/name).mkdir(exist_ok=True)
    width, height, focal = 128, 96, 66.
    intrinsics = dict(width=width, height=height, fx=focal, fy=focal, cx=(width-1)/2, cy=(height-1)/2)
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack([(xx-intrinsics["cx"])/focal, (yy-intrinsics["cy"])/focal, np.ones_like(xx)], axis=-1)
    references, poses, truth = [], [], []
    rng = np.random.default_rng(23)
    for index in range(frames):
        phase = index/(frames-1)
        angle = 2*np.pi*phase
        forward = np.array([np.cos(angle), np.sin(angle), 0.])
        right = np.cross(forward, [0., 0., 1.])
        rotation = np.column_stack([right, [0., 0., -1.], forward])
        position = np.array([2+0.15*np.cos(angle), 2.5+0.15*np.sin(angle), 1.4])
        directions = rays @ rotation.T
        distances = []
        for axis, maximum in enumerate((4., 5., 2.8)):
            for boundary in (0., maximum):
                with np.errstate(divide="ignore", invalid="ignore"):
                    distance = (boundary-position[axis])/directions[..., axis]
                distances.append(np.where(distance > 0, distance, np.inf))
        distance_array = np.stack(distances)
        depth = np.min(distance_array, axis=0)
        depth += rng.normal(0, 0.0005, depth.shape)
        raw = np.round(depth*1000).astype(np.uint16)
        face = np.argmin(distance_array, axis=0)
        palette = np.array([[140, 155, 160], [165, 175, 170], [145, 160, 150], [180, 170, 160], [130, 130, 125], [210, 210, 205]], dtype=np.uint8)
        rgb = palette[face]
        rgb_name, depth_name = f"rgb/{index:06}.png", f"depth/{index:06}.png"
        Image.fromarray(rgb).save(destination/rgb_name)
        Image.fromarray(raw).save(destination/depth_name)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, position
        truth.append({"frame_id": index, "matrix": pose.tolist()})
        if drift:
            yaw = np.deg2rad(0.4*phase)
            drift_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            pose[:3, :3] = drift_rotation @ rotation
            pose[:3, 3] += [0.035*phase, -0.015*phase, 0.004*phase]
        poses.append({"frame_id": index, "matrix": pose.tolist()})
        references.append(dict(frame_id=index, rgb_file=rgb_name, depth_file=depth_name))
    manifest = dict(format_version="1.0.0", capture_id=str(UUID("11111111-1111-4111-8111-111111111111")),
        source_app="synthetic_fixture", depth_unit="millimeter", depth_scale=1000, depth_truncation_m=8,
        pose_convention="camera_to_world", camera_convention="opencv_x_right_y_down_z_forward",
        coordinate_system="synthetic_z_up", source_to_canonical=np.eye(4).tolist(),
        registered_rgb_depth=True, synchronized_rgb_depth=True, frame_count=frames, frames=references,
        metadata={"synthetic": True, "injected_pose_drift": drift, "purpose": "Algorithm testing only; not a real scan or benchmark."})
    for name, data in (("manifest.json", manifest), ("intrinsics.json", intrinsics), ("poses.json", {"poses": poses}),
                       ("synthetic_ground_truth.json", {"synthetic": True, "poses": truth, "room_dimensions_m": [4, 5, 2.8]})):
        (destination/name).write_text(json.dumps(data, indent=2)+"\n")
    return destination
