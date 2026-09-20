"""Typed video configuration and acquisition-independent sparse reconstruction."""
from typing import Literal
from pydantic import Field, model_validator
from property_scanner.schemas.base import ContractModel
from property_scanner.reconstruction.lidar.models import CameraIntrinsics, LidarConfig

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_REVISION = "327deb803d09fac46b05f31a1ccc78a8470c7f6f"


class VideoConfig(ContractModel):
    extraction_fps: float = Field(default=3, gt=0, le=10)
    max_extracted_frames: int = Field(default=180, ge=3, le=1000)
    max_keyframes: int = Field(default=60, ge=3, le=150)
    min_keyframes: int = Field(default=6, ge=3)
    frame_max_width: int = Field(default=960, ge=128, le=1920)
    blur_threshold: float = Field(default=30, ge=0)
    brightness_threshold: float = Field(default=12, ge=0, le=255)
    duplicate_threshold: float = Field(default=2, ge=0, le=255)
    minimum_time_gap: float = Field(default=0.25, ge=0)
    colmap_camera_model: Literal["PINHOLE"] = "PINHOLE"
    intrinsics: CameraIntrinsics | None = None
    sequential_overlap: int = Field(default=8, ge=2, le=30)
    colmap_max_features: int = Field(default=4096, ge=100)
    num_threads: int = Field(default=4, ge=1, le=16)
    command_timeout: int = Field(default=1800, ge=10)
    min_registered_frames: int = Field(default=3, ge=2)
    depth_model: Literal[DEPTH_MODEL] = DEPTH_MODEL
    depth_revision: str = DEPTH_REVISION
    depth_input_size: int = Field(default=518, ge=140, le=700)
    depth_frame_stride: int = Field(default=1, ge=1)
    depth_min_m: float = Field(default=0.2, gt=0)
    depth_max_m: float = Field(default=15, gt=0)
    scale_min_correspondences: int = Field(default=100, ge=10)
    scale_min_frame_correspondences: int = Field(default=15, ge=3)
    scale_min_frames: int = Field(default=3, ge=2)
    scale_outlier_threshold: float = Field(default=3.5, gt=0)
    scale_relative_floor: float = Field(default=0.05, gt=0)
    scale_frame_consistency_threshold: float = Field(default=0.3, gt=0, lt=1)
    scale_max_relative_mad: float = Field(default=0.25, gt=0, lt=1)
    scale_max_reprojection_error: float = Field(default=2, gt=0)
    scale_image_border: int = Field(default=5, ge=0)
    point_cloud_pixel_stride: int = Field(default=3, ge=1, le=16)
    enable_icp_refinement: bool = True
    enable_loop_closure: bool = True
    orientation_min_up_coherence: float = Field(default=0.7, gt=0, le=1)
    registration: LidarConfig = Field(default_factory=LidarConfig)

    @model_validator(mode="after")
    def limits(self):
        if self.depth_max_m <= self.depth_min_m:
            raise ValueError("depth_max_m must exceed depth_min_m")
        if self.min_keyframes > self.max_keyframes or self.max_keyframes > self.max_extracted_frames:
            raise ValueError("Require min_keyframes <= max_keyframes <= max_extracted_frames")
        return self


class VideoMetadata(ContractModel):
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    codec: str
    rotation_degrees: float = 0


class SelectedFrame(ContractModel):
    frame_id: int
    timestamp: float
    filename: str
    blur_score: float
    brightness: float
    visual_change: float | None = None


class SparsePoint(ContractModel):
    point_id: int
    xyz: tuple[float, float, float]
    reprojection_error: float


class Observation(ContractModel):
    x: float
    y: float
    point_id: int


class RegisteredFrame(ContractModel):
    image_id: int
    filename: str
    intrinsics: CameraIntrinsics
    camera_to_world: list[list[float]]
    observations: list[Observation] = Field(default_factory=list)


class SparseReconstruction(ContractModel):
    frames: list[RegisteredFrame]
    points: list[SparsePoint]
    statistics: dict[str, int | float | str] = Field(default_factory=dict)


class ScaleReport(ContractModel):
    metric_scale_resolved: bool = False
    global_scale: float | None = None
    correspondences_total: int = 0
    correspondences_used: int = 0
    frames_used: list[str] = Field(default_factory=list)
    rejected_frames: list[str] = Field(default_factory=list)
    frame_scales: dict[str, float] = Field(default_factory=dict)
    frame_scale_median: float | None = None
    scale_mad: float | None = None
    scale_confidence_quality: float | None = None
    interpretation: str = "Internal scale consistency only; not benchmark accuracy or a measurement confidence interval."
    failure_reason: str | None = None
