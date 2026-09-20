"""Application-independent canonical RGB-D reader with explicit frame matching."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
import numpy as np
from PIL import Image
from pydantic import ValidationError
from property_scanner.core.exceptions import InvalidInputError
from property_scanner.reconstruction.lidar.models import CaptureManifest, CameraIntrinsics, PoseFile, LidarConfig, FrameCounts
from property_scanner.reconstruction.lidar.poses import rigid_transform, normalize_pose
from property_scanner.schemas.result import ResultWarning


@dataclass
class RGBDFrame:
    frame_id: int
    rgb: np.ndarray
    depth_m: np.ndarray
    pose: np.ndarray


class LidarCaptureAdapter(ABC):
    manifest: CaptureManifest
    intrinsics: CameraIntrinsics
    counts: FrameCounts
    warnings: list[ResultWarning]

    @abstractmethod
    def load_manifest(self) -> CaptureManifest: ...

    @abstractmethod
    def validate(self) -> None: ...

    @abstractmethod
    def load_frames(self, config: LidarConfig) -> Iterator[RGBDFrame]: ...


class CanonicalRGBDAdapter(LidarCaptureAdapter):
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.warnings: list[ResultWarning] = []
        self.counts = FrameCounts()
        self.poses: dict[int, np.ndarray] = {}

    def path(self, reference: str) -> Path:
        path = (self.root / reference).resolve()
        if not path.is_relative_to(self.root) or Path(reference).is_absolute():
            raise InvalidInputError("Capture references must stay inside the capture directory")
        return path

    def load_manifest(self) -> CaptureManifest:
        try:
            return CaptureManifest.model_validate_json(self.path("manifest.json").read_text())
        except (OSError, ValidationError) as exc:
            raise InvalidInputError(f"Invalid or missing canonical manifest: {exc}") from exc

    def validate(self) -> None:
        self.manifest = self.load_manifest()
        self.counts = FrameCounts(frames_total=self.manifest.frame_count)
        self.warnings = []
        try:
            self.intrinsics = CameraIntrinsics.model_validate_json(self.path(self.manifest.intrinsics_file).read_text())
            pose_file = PoseFile.model_validate_json(self.path(self.manifest.poses_file).read_text())
            transform = rigid_transform(self.manifest.source_to_canonical)
            if len({p.frame_id for p in pose_file.poses}) != len(pose_file.poses):
                raise InvalidInputError("Duplicate frame IDs in poses.json")
            self.poses = {}
            for pose in pose_file.poses:
                try:
                    self.poses[pose.frame_id] = normalize_pose(pose.matrix, self.manifest.pose_convention, transform)
                except InvalidInputError:
                    self.warnings.append(ResultWarning(code="INVALID_FRAME_POSE", message=f"Frame {pose.frame_id}: invalid rigid pose; frame will be skipped."))
            for frame in self.manifest.frames:
                self.path(frame.rgb_file)
                self.path(frame.depth_file)
        except (OSError, ValidationError, ValueError) as exc:
            raise InvalidInputError(f"Invalid capture calibration or poses: {exc}") from exc

    def load_frames(self, config: LidarConfig) -> Iterator[RGBDFrame]:
        for frame in self.manifest.frames:
            try:
                if frame.frame_id not in self.poses:
                    raise ValueError("missing or invalid pose")
                with Image.open(self.path(frame.rgb_file)) as image:
                    rgb = np.asarray(image.convert("RGB"))
                depth_path = self.path(frame.depth_file)
                if depth_path.suffix.lower() == ".npy":
                    depth = np.load(depth_path, allow_pickle=False)
                elif depth_path.suffix.lower() == ".png":
                    with Image.open(depth_path) as image:
                        depth = np.asarray(image)
                else:
                    raise ValueError("depth must be PNG or NPY")
                expected = (self.intrinsics.height, self.intrinsics.width)
                if rgb.shape[:2] != expected or depth.shape != expected:
                    raise InvalidInputError("RGB/depth/calibration dimensions differ; registration is required and is not implemented")
                if not np.issubdtype(depth.dtype, np.number) or not np.isfinite(depth).all() or (depth < 0).any():
                    raise ValueError("depth contains invalid, negative, NaN, or infinite values")
                depth = depth.astype(np.float32) / self.manifest.depth_scale
                valid = (depth >= config.depth_min_m) & (depth > 0) & (depth <= min(config.depth_max_m, self.manifest.depth_truncation_m))
                if np.count_nonzero(valid) < config.min_valid_depth_pixels:
                    raise ValueError("insufficient valid depth pixels")
                depth[~valid] = 0
                self.counts.frames_loaded += 1
                yield RGBDFrame(frame.frame_id, rgb, depth, self.poses[frame.frame_id])
            except (OSError, ValueError, EOFError) as exc:
                self.counts.frames_invalid += 1
                self.counts.frames_skipped += 1
                reason = (exc.strerror or "unreadable RGB/depth file") if isinstance(exc, OSError) else str(exc)
                self.warnings.append(ResultWarning(code="INVALID_RGBD_FRAME", message=f"Frame {frame.frame_id} skipped: {reason}"))
