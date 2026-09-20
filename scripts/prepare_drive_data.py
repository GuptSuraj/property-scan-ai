#!/usr/bin/env python3
"""
Convert drive-download RGB-D captures to the canonical format expected by the
PropertyScanPipeline LiDAR adapter.

Each source folder must contain:
  - camera_matrix.csv   : 3×3 intrinsic matrix (fx 0 cx / 0 fy cy / 0 0 1)
  - odometry.csv        : columns: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
  - depth/NNNNNN.png    : uint16 depth frames (millimetres)
  - rgb.mp4             : RGB video; one frame extracted per depth frame by index
  - confidence/NNNNNN.png (optional)

Outputs written to <output_root>/<capture_name>/:
  manifest.json, intrinsics.json, poses.json
  rgb/NNNNNN.jpg  (extracted from video)

Usage:
  python scripts/prepare_drive_data.py \\
      --source  ./drive-download-20260919T042720Z-1-001 \\
      --output  ./outputs/canonical
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix (Hamilton convention: w last as stored in odometry.csv)
# ---------------------------------------------------------------------------
def quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Return a 3×3 rotation matrix from a unit quaternion (qx, qy, qz, qw)."""
    n = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if n < 1e-9:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=float)


def make_camera_to_world(x, y, z, qx, qy, qz, qw) -> list[list[float]]:
    """Build a 4×4 camera-to-world SE(3) matrix from translation + quaternion."""
    R = quat_to_rotation(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T.tolist()


# ---------------------------------------------------------------------------
# Read camera_matrix.csv  (3 rows, 3 comma-separated values per row)
# ---------------------------------------------------------------------------
def read_camera_matrix(path: Path) -> dict:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(v.strip()) for v in line.split(",")])
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError(f"Expected 3×3 camera matrix in {path}, got {len(rows)} rows")
    return {"fx": rows[0][0], "fy": rows[1][1], "cx": rows[0][2], "cy": rows[1][2]}


# ---------------------------------------------------------------------------
# Read odometry.csv
# header: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
# ---------------------------------------------------------------------------
def read_odometry(path: Path) -> list[dict]:
    records = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            records.append(row)
    return records


# ---------------------------------------------------------------------------
# Probe video dimensions using ffprobe
# ---------------------------------------------------------------------------
def probe_video(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width, height = result.stdout.strip().split(",")
    return int(width), int(height)


# ---------------------------------------------------------------------------
# Extract a single frame from the video by frame index using ffmpeg
# ---------------------------------------------------------------------------
def extract_frame(video_path: Path, frame_index: int, output_path: Path,
                  resize_w: int | None = None, resize_h: int | None = None) -> None:
    """Extract one frame by 0-based index using ffmpeg select filter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{frame_index})"
    if resize_w and resize_h:
        vf += f",scale={resize_w}:{resize_h}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-frames:v", "1",
        "-q:v", "2",
        "-loglevel", "error",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Get depth image dimensions
# ---------------------------------------------------------------------------
def depth_image_size(depth_path: Path) -> tuple[int, int]:
    with Image.open(depth_path) as img:
        return img.size  # (width, height)


# ---------------------------------------------------------------------------
# Core converter for one capture folder
# ---------------------------------------------------------------------------
def convert_capture(source: Path, output: Path, max_frames: int | None = None) -> None:
    log.info("Converting %s → %s", source.name, output)
    output.mkdir(parents=True, exist_ok=True)

    # --- intrinsics from camera_matrix.csv ---
    cam_csv = source / "camera_matrix.csv"
    if not cam_csv.exists():
        raise FileNotFoundError(f"Missing {cam_csv}")
    intr = read_camera_matrix(cam_csv)

    # --- depth frames ---
    depth_dir = source / "depth"
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth/ directory in {source}")
    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        raise FileNotFoundError(f"No depth PNGs in {depth_dir}")

    # Check depth resolution
    depth_w, depth_h = depth_image_size(depth_files[0])
    log.info("  Depth resolution: %d×%d  (%d frames)", depth_w, depth_h, len(depth_files))

    # --- odometry ---
    odo_csv = source / "odometry.csv"
    if not odo_csv.exists():
        raise FileNotFoundError(f"Missing {odo_csv}")
    odometry = read_odometry(odo_csv)
    # Build frame_id → odometry row map
    odo_by_frame = {}
    for row in odometry:
        frame_str = row.get("frame", "").strip()
        if frame_str:
            try:
                odo_by_frame[int(frame_str)] = row
            except ValueError:
                pass

    log.info("  Odometry rows: %d", len(odo_by_frame))

    # --- video ---
    video_path = source / "rgb.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing rgb.mp4 in {source}")
    rgb_w, rgb_h = probe_video(video_path)
    log.info("  RGB video: %d×%d", rgb_w, rgb_h)

    # --- RGB output directory ---
    rgb_out = output / "rgb"
    rgb_out.mkdir(parents=True, exist_ok=True)

    # --- Select frames: intersection of depth files and odometry ---
    frame_ids = []
    for df in depth_files:
        fid = int(df.stem)
        if fid in odo_by_frame:
            frame_ids.append(fid)
    if max_frames:
        step = max(1, len(frame_ids) // max_frames)
        frame_ids = frame_ids[::step][:max_frames]

    log.info("  Processing %d frames", len(frame_ids))

    poses = []
    frames_manifest = []

    for i, fid in enumerate(frame_ids):
        if i % 100 == 0:
            log.info("    Frame %d / %d", i, len(frame_ids))

        row = odo_by_frame[fid]

        # Extract RGB frame from video
        rgb_out_path = rgb_out / f"{fid:06d}.jpg"
        if not rgb_out_path.exists():
            extract_frame(video_path, fid, rgb_out_path)

        # Build pose matrix (camera-to-world)
        try:
            tx = float(row["x"]); ty = float(row["y"]); tz = float(row["z"])
            qx = float(row["qx"]); qy = float(row["qy"])
            qz = float(row["qz"]); qw = float(row["qw"])
        except (KeyError, ValueError) as exc:
            log.warning("  Skipping frame %d — bad odometry: %s", fid, exc)
            continue

        matrix = make_camera_to_world(tx, ty, tz, qx, qy, qz, qw)
        poses.append({"frame_id": fid, "matrix": matrix})
        frames_manifest.append({
            "frame_id": fid,
            "rgb_file": f"rgb/{fid:06d}.jpg",
            "depth_file": f"depth/{fid:06d}.png",
        })

    if not frames_manifest:
        raise RuntimeError("No valid frames found after filtering")

    log.info("  %d frames with valid poses", len(frames_manifest))

    # Symlink depth/ and confidence/ from source into canonical output so the
    # adapter can read them via the relative paths stored in the manifest.
    for subdir in ("depth", "confidence"):
        src_dir = source / subdir
        dst_link = output / subdir
        if src_dir.is_dir() and not dst_link.exists():
            dst_link.symlink_to(src_dir.resolve())
            log.info("  Linked %s/ from source", subdir)

    log.info("  %d frames with valid poses", len(frames_manifest))

    # --- Write intrinsics.json ---
    # Use per-frame intrinsics from first odometry row if available, else camera_matrix.csv
    first_row = odo_by_frame.get(frame_ids[0], {})
    fx = float(first_row.get("fx") or intr["fx"])
    fy = float(first_row.get("fy") or intr["fy"])
    cx = float(first_row.get("cx") or intr["cx"])
    cy = float(first_row.get("cy") or intr["cy"])

    intrinsics = {
        "width": depth_w,
        "height": depth_h,
        "fx": fx * (depth_w / rgb_w),   # scale to depth resolution
        "fy": fy * (depth_h / rgb_h),
        "cx": cx * (depth_w / rgb_w),
        "cy": cy * (depth_h / rgb_h),
    }

    # Validate principal point is inside image
    if intrinsics["cx"] >= intrinsics["width"]:
        intrinsics["cx"] = intrinsics["width"] / 2.0
    if intrinsics["cy"] >= intrinsics["height"]:
        intrinsics["cy"] = intrinsics["height"] / 2.0

    (output / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
    log.info("  Intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (%dx%d)",
             intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"],
             intrinsics["width"], intrinsics["height"])

    # --- Write poses.json ---
    (output / "poses.json").write_text(json.dumps({"poses": poses}, indent=2))

    # --- Write manifest.json ---
    capture_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(source.resolve())))
    manifest = {
        "format_version": "1.0.0",
        "capture_id": capture_id,
        "source_app": "drive-download-converter",
        "depth_unit": "millimeter",
        "depth_scale": 1000,           # uint16 mm → divide by 1000 → metres
        "depth_truncation_m": 5.0,
        "pose_convention": "camera_to_world",
        "camera_convention": "opencv_x_right_y_down_z_forward",
        "coordinate_system": "metric_z_up",
        "source_to_canonical": [       # 4×4 identity (already metric)
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "registered_rgb_depth": True,
        "synchronized_rgb_depth": True,
        "frame_count": len(frames_manifest),
        "frames": frames_manifest,
        "metadata": {"source_folder": source.name},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("  manifest.json written — %d frames, capture_id=%s", len(frames_manifest), capture_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert drive-download RGB-D captures to canonical format.")
    parser.add_argument("--source", required=True, help="Root folder containing capture sub-folders (1a8384c3f6, etc.)")
    parser.add_argument("--output", required=True, help="Root output directory for canonical captures")
    parser.add_argument("--capture", help="Convert only this capture folder (by name); defaults to all three")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit total frames per capture (useful for quick tests)")
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()

    if not source_root.is_dir():
        log.error("Source directory not found: %s", source_root)
        sys.exit(1)

    captures = (
        [source_root / args.capture] if args.capture
        else [d for d in sorted(source_root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    )

    if not captures:
        log.error("No capture sub-folders found in %s", source_root)
        sys.exit(1)

    for cap in captures:
        dest = output_root / cap.name
        try:
            convert_capture(cap, dest, max_frames=args.max_frames)
            log.info("✓  %s converted successfully → %s", cap.name, dest)
        except Exception as exc:
            log.error("✗  %s failed: %s", cap.name, exc)
            raise


if __name__ == "__main__":
    main()
