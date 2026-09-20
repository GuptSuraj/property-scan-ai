"""Convert a Record3D ``.r3d`` archive into the canonical RGB-D format.

Record3D stores registered RGB and LZFSE-compressed float32 depth frames plus
camera-to-world poses.  Conversion is deliberately isolated here so the
reconstruction pipeline remains capture-application independent.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
from uuid import NAMESPACE_URL, uuid5
import zipfile

import numpy as np
from PIL import Image

from property_scanner.core.exceptions import InvalidInputError
from property_scanner.reconstruction.lidar.models import (
    CameraIntrinsics,
    CaptureManifest,
    FramePose,
    FrameReference,
    PoseFile,
)

_RGB_PATTERN = re.compile(r"(?:^|/)rgbd/(\d+)\.jpg$")
_CAMERA_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
_WORLD_Y_UP_TO_CANONICAL_Z_UP = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)


def _decompress_lzfse(payload: bytes) -> bytes:
    try:
        import liblzfse  # type: ignore[import-not-found]
    except ImportError as exc:
        raise InvalidInputError(
            "Record3D conversion requires pyliblzfse. Install the LiDAR dependencies "
            "with: pip install -e '.[lidar]'"
        ) from exc
    try:
        return liblzfse.decompress(payload)
    except Exception as exc:
        raise InvalidInputError(f"Could not decompress a Record3D depth frame: {exc}") from exc


def _quaternion_pose(values: list[float]) -> np.ndarray:
    """Return Record3D camera-to-world pose from qx,qy,qz,qw,tx,ty,tz."""
    if len(values) != 7 or not np.isfinite(values).all():
        raise InvalidInputError("Record3D poses must contain seven finite values")
    x, y, z, w, tx, ty, tz = (float(value) for value in values)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 0.5 or norm > 1.5:
        raise InvalidInputError("Record3D pose contains an invalid quaternion")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [tx, ty, tz]
    # Record3D/ARKit camera coordinates are OpenGL-style. Canonical files
    # declare OpenCV camera axes, so convert the camera basis once here.
    return matrix @ _CAMERA_OPENGL_TO_OPENCV


def _read_metadata(handle: zipfile.ZipFile) -> dict:
    names = set(handle.namelist())
    candidates = [name for name in ("metadata", "metadata.json") if name in names]
    if not candidates:
        candidates = [name for name in names if name.endswith("/metadata") or name.endswith("/metadata.json")]
    if len(candidates) != 1:
        raise InvalidInputError("Record3D archive must contain exactly one metadata file")
    try:
        value = json.loads(handle.read(candidates[0]))
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
        raise InvalidInputError(f"Invalid Record3D metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidInputError("Record3D metadata must be a JSON object")
    return value


def _metadata_dimensions(metadata: dict) -> tuple[int, int, int, int]:
    try:
        width, height = int(metadata["w"]), int(metadata["h"])
        depth_width, depth_height = int(metadata["dw"]), int(metadata["dh"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError("Record3D metadata is missing valid w/h/dw/dh dimensions") from exc
    if min(width, height, depth_width, depth_height) <= 0:
        raise InvalidInputError("Record3D image dimensions must be positive")
    return width, height, depth_width, depth_height


def _intrinsics(metadata: dict, width: int, height: int) -> CameraIntrinsics:
    try:
        # Record3D publishes K flattened in column-major order.
        matrix = np.asarray(metadata["K"], dtype=float).reshape(3, 3).T
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError("Record3D metadata is missing a valid 3x3 K matrix") from exc
    if not np.isfinite(matrix).all():
        raise InvalidInputError("Record3D intrinsics contain non-finite values")
    try:
        return CameraIntrinsics(width=width, height=height, fx=matrix[0, 0], fy=matrix[1, 1],
                                cx=matrix[0, 2], cy=matrix[1, 2])
    except Exception as exc:
        raise InvalidInputError(f"Record3D intrinsics are invalid: {exc}") from exc


def _frame_names(handle: zipfile.ZipFile) -> dict[int, str]:
    frames: dict[int, str] = {}
    for name in handle.namelist():
        match = _RGB_PATTERN.search(name)
        if match:
            frames[int(match.group(1))] = name
    if not frames:
        raise InvalidInputError("Record3D archive contains no rgbd/<frame>.jpg images")
    return frames


def convert_record3d(
    archive: Path,
    destination: Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Path:
    """Convert an official Record3D archive to a validated canonical capture."""
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if archive.suffix.lower() != ".r3d" or not archive.is_file() or not zipfile.is_zipfile(archive):
        raise InvalidInputError("Expected a readable Record3D .r3d archive")
    if frame_stride < 1 or (max_frames is not None and max_frames < 1):
        raise InvalidInputError("Record3D frame_stride and max_frames must be positive")

    digest = sha256(archive.read_bytes()).hexdigest()
    if (destination / "manifest.json").is_file():
        return destination
    staging = destination.with_name(f".{destination.name}.converting")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "rgb").mkdir(parents=True)
    (staging / "depth").mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as handle:
            metadata = _read_metadata(handle)
            width, height, depth_width, depth_height = _metadata_dimensions(metadata)
            intrinsics = _intrinsics(metadata, width, height)
            rgb_names = _frame_names(handle)
            poses_data = metadata.get("poses")
            if not isinstance(poses_data, list):
                raise InvalidInputError("Record3D metadata is missing its poses array")
            selected = sorted(index for index in rgb_names if index % frame_stride == 0)
            if max_frames is not None:
                selected = selected[:max_frames]
            if not selected:
                raise InvalidInputError("No Record3D frames remain after sampling")

            frame_refs: list[FrameReference] = []
            poses: list[FramePose] = []
            for output_index, source_index in enumerate(selected):
                depth_name = re.sub(r"\.jpg$", ".depth", rgb_names[source_index])
                if depth_name not in handle.namelist():
                    raise InvalidInputError(f"Record3D frame {source_index} has no matching depth payload")
                if source_index >= len(poses_data):
                    raise InvalidInputError(f"Record3D frame {source_index} has no matching pose")

                with Image.open(BytesIO(handle.read(rgb_names[source_index]))) as image:
                    rgb = image.convert("RGB")
                    if rgb.size != (width, height):
                        raise InvalidInputError(f"Record3D RGB frame {source_index} dimensions differ from metadata")
                    rgb.save(staging / "rgb" / f"{output_index:06d}.jpg", quality=95)

                raw = _decompress_lzfse(handle.read(depth_name))
                depth = np.frombuffer(raw, dtype=np.float32)
                if depth.size != depth_width * depth_height:
                    raise InvalidInputError(f"Record3D depth frame {source_index} has an invalid size")
                depth = depth.reshape(depth_height, depth_width)
                if not np.isfinite(depth).all() or (depth < 0).any():
                    raise InvalidInputError(f"Record3D depth frame {source_index} contains invalid values")
                if depth.shape != (height, width):
                    with Image.fromarray(depth) as depth_image:
                        depth_image = depth_image.resize((width, height), resample=Image.Resampling.NEAREST)
                        depth = np.asarray(depth_image, dtype=np.float32)
                millimeters = np.where(depth > 0, np.clip(np.rint(depth * 1000), 1, 65535), 0).astype(np.uint16)
                Image.fromarray(millimeters).save(staging / "depth" / f"{output_index:06d}.png")

                pose = _quaternion_pose(poses_data[source_index])
                frame_refs.append(FrameReference(frame_id=output_index,
                    rgb_file=f"rgb/{output_index:06d}.jpg", depth_file=f"depth/{output_index:06d}.png"))
                poses.append(FramePose(frame_id=output_index, matrix=pose.tolist()))

        intrinsics_path = staging / "intrinsics.json"
        intrinsics_path.write_text(intrinsics.model_dump_json(indent=2))
        (staging / "poses.json").write_text(PoseFile(poses=poses).model_dump_json(indent=2))
        manifest = CaptureManifest(
            format_version="1.0.0",
            capture_id=uuid5(NAMESPACE_URL, f"record3d:{digest}"),
            source_app="Record3D",
            source_app_version=str(metadata.get("version")) if metadata.get("version") is not None else None,
            device_model=str(metadata.get("deviceName")) if metadata.get("deviceName") else None,
            depth_unit="millimeter",
            depth_scale=1000,
            depth_truncation_m=15,
            pose_convention="camera_to_world",
            camera_convention="opencv_x_right_y_down_z_forward",
            coordinate_system="Record3D/ARKit world coordinates (Y up); converted to canonical Z up during loading",
            source_to_canonical=_WORLD_Y_UP_TO_CANONICAL_Z_UP.tolist(),
            registered_rgb_depth=True,
            synchronized_rgb_depth=True,
            frame_count=len(frame_refs),
            frames=frame_refs,
            metadata={"source_archive_sha256": digest, "source_frame_ids": selected},
        )
        (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(destination, ignore_errors=True)
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
