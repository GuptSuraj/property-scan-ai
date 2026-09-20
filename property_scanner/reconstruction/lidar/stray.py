"""Convert a Stray Scanner RGB-D capture directory into canonical RGB-D format.

Stray Scanner records:
- camera_matrix.csv (3x3 intrinsic matrix for the full-resolution video)
- odometry.csv (per-frame timestamp, frame_id, x,y,z, qx,qy,qz,qw, fx,fy,cx,cy)
- depth/ (16-bit millimeter PNG depth maps, 256x192)
- rgb.mp4 (HEVC video, 1920x1440)
- confidence/ (8-bit confidence maps, 0-2)
"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil
from uuid import NAMESPACE_URL, uuid5

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from property_scanner.core.exceptions import InvalidInputError
from property_scanner.reconstruction.lidar.models import (
    CameraIntrinsics,
    CaptureManifest,
    FramePose,
    FrameReference,
    PoseFile,
)

logger = logging.getLogger(__name__)

# ARKit world coordinates have Y up. Convert to canonical Z up:
# X_canon = X_world, Y_canon = -Z_world, Z_canon = Y_world
_WORLD_Y_UP_TO_CANONICAL_Z_UP = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)


def is_stray_scanner_dir(path: Path) -> bool:
    """Check whether a directory contains a Stray Scanner export."""
    path = Path(path).resolve()
    return (
        path.is_dir()
        and (path / "camera_matrix.csv").is_file()
        and (path / "odometry.csv").is_file()
        and (path / "depth").is_dir()
        and (path / "rgb.mp4").is_file()
        and not (path / "manifest.json").is_file()
    )


def convert_stray_scanner(
    source: Path,
    destination: Path,
    *,
    max_frames: int = 150,
) -> Path:
    """Convert Stray Scanner raw capture into a validated canonical capture."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()

    if not is_stray_scanner_dir(source):
        raise InvalidInputError(f"{source} is not a valid Stray Scanner capture directory")

    if (destination / "manifest.json").is_file():
        return destination

    logger.info("Converting Stray Scanner capture %s -> %s", source.name, destination)

    # 1. Read camera_matrix.csv
    try:
        cam_matrix = np.loadtxt(source / "camera_matrix.csv", delimiter=",")
    except Exception as exc:
        raise InvalidInputError(f"Cannot read camera_matrix.csv: {exc}") from exc

    # 2. Read odometry.csv
    odo_rows = []
    with (source / "odometry.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            odo_rows.append({k.strip(): v.strip() for k, v in row.items()})

    if not odo_rows:
        raise InvalidInputError("odometry.csv is empty")

    # Map frame_id -> row
    odo_by_frame = {}
    for row in odo_rows:
        try:
            fid = int(row["frame"])
            odo_by_frame[fid] = row
        except (KeyError, ValueError):
            continue

    # 3. Available depth files
    depth_dir = source / "depth"
    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        raise InvalidInputError("No depth files found in depth/")

    # Probe depth dimensions
    with Image.open(depth_files[0]) as img:
        depth_w, depth_h = img.size

    # Valid frames present in both depth and odometry
    available_fids = [int(p.stem) for p in depth_files if int(p.stem) in odo_by_frame]
    if not available_fids:
        raise InvalidInputError("No matching frames between depth and odometry")

    # Sample candidate frames evenly up to max_frames
    if len(available_fids) > max_frames:
        step = max(1, len(available_fids) // max_frames)
        selected_fids = available_fids[::step][:max_frames]
    else:
        selected_fids = available_fids

    selected_set = set(selected_fids)

    # Staging directory
    staging = destination.with_name(f".{destination.name}.converting")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "rgb").mkdir(parents=True)
    (staging / "depth").mkdir(parents=True)

    try:
        # 4. Extract RGB frames at depth resolution using cv2
        cap = cv2.VideoCapture(str(source / "rgb.mp4"))
        if not cap.isOpened():
            raise InvalidInputError(f"Cannot open video {source / 'rgb.mp4'}")

        rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rgb_w <= 0 or rgb_h <= 0:
            rgb_w, rgb_h = 1920, 1440

        scale_x = depth_w / float(rgb_w)
        scale_y = depth_h / float(rgb_h)

        extracted_frames = set()
        current_idx = 0
        while cap.isOpened() and len(extracted_frames) < len(selected_set):
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in selected_set:
                resized = cv2.resize(frame, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
                out_path = staging / "rgb" / f"{current_idx:06d}.jpg"
                cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_frames.add(current_idx)
            current_idx += 1
        cap.release()

        # 5. Build frames, copy depth, build poses
        final_fids = sorted(extracted_frames)
        if len(final_fids) < 2:
            raise InvalidInputError("Too few valid frames extracted from Stray Scanner video")

        frame_refs: list[FrameReference] = []
        poses: list[FramePose] = []

        for out_idx, fid in enumerate(final_fids):
            # Copy depth frame
            src_depth = depth_dir / f"{fid:06d}.png"
            dst_depth = staging / "depth" / f"{out_idx:06d}.png"
            shutil.copy2(src_depth, dst_depth)

            # Move/rename rgb to match out_idx
            src_rgb = staging / "rgb" / f"{fid:06d}.jpg"
            dst_rgb = staging / "rgb" / f"{out_idx:06d}.jpg"
            if src_rgb != dst_rgb:
                src_rgb.rename(dst_rgb)

            # Build pose
            row = odo_by_frame[fid]
            tx, ty, tz = float(row["x"]), float(row["y"]), float(row["z"])
            qx, qy, qz, qw = float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

            matrix = np.eye(4)
            matrix[:3, :3] = rot
            matrix[:3, 3] = [tx, ty, tz]

            frame_refs.append(FrameReference(
                frame_id=out_idx,
                rgb_file=f"rgb/{out_idx:06d}.jpg",
                depth_file=f"depth/{out_idx:06d}.png",
            ))
            poses.append(FramePose(frame_id=out_idx, matrix=matrix.tolist()))

        # 6. Intrinsics scaled to depth resolution
        first_row = odo_by_frame[final_fids[0]]
        fx_raw = float(first_row.get("fx") or cam_matrix[0, 0])
        fy_raw = float(first_row.get("fy") or cam_matrix[1, 1])
        cx_raw = float(first_row.get("cx") or cam_matrix[0, 2])
        cy_raw = float(first_row.get("cy") or cam_matrix[1, 2])

        intrinsics = CameraIntrinsics(
            width=depth_w,
            height=depth_h,
            fx=fx_raw * scale_x,
            fy=fy_raw * scale_y,
            cx=cx_raw * scale_x,
            cy=cy_raw * scale_y,
        )

        (staging / "intrinsics.json").write_text(intrinsics.model_dump_json(indent=2))
        (staging / "poses.json").write_text(PoseFile(poses=poses).model_dump_json(indent=2))

        # 7. Manifest
        digest = sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
        manifest = CaptureManifest(
            format_version="1.0.0",
            capture_id=uuid5(NAMESPACE_URL, f"stray:{digest}"),
            source_app="StrayScanner",
            depth_unit="millimeter",
            depth_scale=1000,
            depth_truncation_m=8.0,
            pose_convention="camera_to_world",
            camera_convention="opencv_x_right_y_down_z_forward",
            coordinate_system="StrayScanner/ARKit world coordinates (Y up); converted to canonical Z up during loading",
            source_to_canonical=_WORLD_Y_UP_TO_CANONICAL_Z_UP.tolist(),
            registered_rgb_depth=True,
            synchronized_rgb_depth=True,
            frame_count=len(frame_refs),
            frames=frame_refs,
            metadata={"source_capture_name": source.name},
        )
        (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2))

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(destination, ignore_errors=True)
        staging.rename(destination)
        logger.info("Successfully converted %s (%d frames) -> %s", source.name, len(frame_refs), destination)
        return destination

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
