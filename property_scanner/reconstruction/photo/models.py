"""Typed photo configuration, discovery records, and image diagnostics."""

from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from property_scanner.reconstruction.lidar.models import CameraIntrinsics, LidarConfig
from property_scanner.reconstruction.video.models import DEPTH_MODEL, DEPTH_REVISION
from property_scanner.schemas.base import ContractModel
from property_scanner.stitching.models import StitchingConfig


class PhotoConfig(ContractModel):
    min_images_per_room: int = Field(default=2, ge=2, le=8)
    max_images_per_room: int = Field(default=8, ge=2, le=8)
    min_image_dimension: int = Field(default=64, ge=16)
    normalized_max_width: int = Field(default=1600, ge=128, le=4096)
    blur_threshold: float = Field(default=20, ge=0)
    brightness_min: float = Field(default=8, ge=0, le=255)
    brightness_max: float = Field(default=247, ge=0, le=255)
    contrast_min: float = Field(default=5, ge=0)
    duplicate_threshold: float = Field(default=2, ge=0, le=255)
    colmap_camera_model: Literal["PINHOLE"] = "PINHOLE"
    colmap_matching_strategy: Literal["exhaustive"] = "exhaustive"
    intrinsics: CameraIntrinsics | None = None
    single_camera: bool = False
    colmap_max_features: int = Field(default=8192, ge=100)
    num_threads: int = Field(default=4, ge=1, le=16)
    command_timeout: int = Field(default=1800, ge=10)
    min_registered_frames: int = Field(default=2, ge=2, le=8)
    depth_model: Literal[DEPTH_MODEL] = DEPTH_MODEL
    depth_revision: str = DEPTH_REVISION
    depth_input_size: int = Field(default=518, ge=140, le=700)
    depth_min_m: float = Field(default=0.2, gt=0)
    depth_max_m: float = Field(default=15, gt=0)
    scale_min_correspondences: int = Field(default=50, ge=10)
    scale_min_frame_correspondences: int = Field(default=15, ge=3)
    scale_min_frames: int = Field(default=2, ge=2, le=8)
    scale_outlier_threshold: float = Field(default=3.5, gt=0)
    scale_relative_floor: float = Field(default=0.05, gt=0)
    scale_frame_consistency_threshold: float = Field(default=0.3, gt=0, lt=1)
    scale_max_relative_mad: float = Field(default=0.25, gt=0, lt=1)
    scale_max_reprojection_error: float = Field(default=2, gt=0)
    scale_image_border: int = Field(default=5, ge=0)
    point_cloud_pixel_stride: int = Field(default=3, ge=1, le=16)
    enable_icp_refinement: bool = False
    enable_loop_closure: bool = False
    enable_property_stitching: bool = True
    stitching: StitchingConfig = Field(default_factory=StitchingConfig)
    orientation_min_up_coherence: float = Field(default=0.6, gt=0, le=1)
    registration: LidarConfig = Field(default_factory=LidarConfig)

    @model_validator(mode="after")
    def consistent_limits(self):
        if self.min_images_per_room > self.max_images_per_room:
            raise ValueError("min_images_per_room must not exceed max_images_per_room")
        if self.depth_max_m <= self.depth_min_m:
            raise ValueError("depth_max_m must exceed depth_min_m")
        if self.scale_min_frames > self.max_images_per_room:
            raise ValueError("scale_min_frames exceeds the room image limit")
        return self


class RoomSource(ContractModel):
    room_id: str
    label: str
    directory: Path
    images: list[Path]


class ImageQualityRecord(ContractModel):
    source_filename: str
    selected_filename: str | None = None
    readable: bool
    width: int | None = None
    height: int | None = None
    blur_score: float | None = None
    brightness: float | None = None
    contrast: float | None = None
    feature_count: int | None = None
    quality_score: float | None = None
    selected: bool = False
    rejection_reason: str | None = None
    exif: dict[str, JsonValue] = Field(default_factory=dict)


class PreparedRoomImages(ContractModel):
    room: RoomSource
    selected_directory: Path
    selected: list[ImageQualityRecord]
    records: list[ImageQualityRecord]
    images_total: int
    images_valid: int
    images_rejected: int
    same_camera_supported: bool = False
    warning_codes: list[str] = Field(default_factory=list)
