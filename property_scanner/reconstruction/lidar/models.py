"""Versioned canonical export format and typed processing configuration."""
from typing import Literal, Self
from uuid import UUID
from pydantic import Field, AwareDatetime, JsonValue, model_validator
from property_scanner.schemas.base import ContractModel, NonEmptyText
from property_scanner.geometry.config import GeometryConfig
from property_scanner.openings.models import OpeningConfig


class FrameReference(ContractModel):
    frame_id: int = Field(ge=0)
    rgb_file: NonEmptyText
    depth_file: NonEmptyText


class CaptureManifest(ContractModel):
    format_version: Literal["1.0.0"]
    capture_id: UUID
    source_app: NonEmptyText | None = None
    source_app_version: NonEmptyText | None = None
    device_model: NonEmptyText | None = None
    timestamp: AwareDatetime | None = None
    poses_file: NonEmptyText = "poses.json"
    intrinsics_file: NonEmptyText = "intrinsics.json"
    depth_unit: Literal["meter", "millimeter"]
    depth_scale: float = Field(gt=0, description="Stored depth units per meter; raw / depth_scale gives meters.")
    depth_truncation_m: float = Field(default=8, gt=0)
    pose_convention: Literal["camera_to_world", "world_to_camera"]
    camera_convention: Literal["opencv_x_right_y_down_z_forward"]
    coordinate_system: NonEmptyText
    source_to_canonical: list[list[float]]
    registered_rgb_depth: Literal[True]
    synchronized_rgb_depth: Literal[True]
    frame_count: int = Field(ge=1)
    frames: list[FrameReference] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_manifest(self) -> Self:
        expected = 1000 if self.depth_unit == "millimeter" else 1
        if self.depth_scale != expected:
            raise ValueError(f"depth_scale must be {expected} for {self.depth_unit}; never guess units")
        if self.frame_count != len(self.frames) or len({f.frame_id for f in self.frames}) != len(self.frames):
            raise ValueError("frame_count must match unique explicit frame IDs")
        return self


class CameraIntrinsics(ContractModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float = Field(ge=0)
    cy: float = Field(ge=0)

    @model_validator(mode="after")
    def principal_point(self) -> Self:
        if self.cx >= self.width or self.cy >= self.height:
            raise ValueError("Principal point must lie inside the registered image")
        return self


class FramePose(ContractModel):
    frame_id: int = Field(ge=0)
    matrix: list[list[float]]


class PoseFile(ContractModel):
    poses: list[FramePose]


class LidarConfig(ContractModel):
    drift_correction: Literal["on", "off"] = "on"
    frame_stride: int = Field(default=1, ge=1)
    max_frames: int = Field(default=80, ge=2, le=500)
    min_translation_between_keyframes: float = Field(default=0.08, ge=0)
    min_rotation_between_keyframes: float = Field(default=8, ge=0, le=180)
    depth_min_m: float = Field(default=0.15, ge=0)
    depth_max_m: float = Field(default=8, gt=0)
    min_valid_depth_pixels: int = Field(default=100, ge=3)
    min_keyframes: int = Field(default=2, ge=2)
    frame_voxel_size: float = Field(default=0.05, gt=0)
    fusion_voxel_size: float = Field(default=0.03, gt=0)
    max_points_per_frame: int = Field(default=20000, ge=100)
    max_fused_points: int = Field(default=500000, ge=100)
    normal_radius: float = Field(default=0.2, gt=0)
    normal_max_nn: int = Field(default=30, ge=3)
    remove_frame_outliers: bool = False
    statistical_nb_neighbors: int = Field(default=20, ge=2)
    statistical_std_ratio: float = Field(default=2.5, gt=0)
    icp_max_correspondence_coarse: float = Field(default=0.25, gt=0)
    icp_max_correspondence_fine: float = Field(default=0.08, gt=0)
    icp_max_iterations: int = Field(default=40, ge=1)
    neighbor_min_fitness: float = Field(default=0.25, ge=0, le=1)
    neighbor_max_rmse: float = Field(default=0.06, gt=0)
    loop_min_frame_separation: int = Field(default=8, ge=2)
    loop_search_radius: float = Field(default=0.8, gt=0)
    loop_orientation_tolerance: float = Field(default=35, gt=0, le=180)
    max_loop_candidates_per_frame: int = Field(default=2, ge=1, le=10)
    loop_min_fitness: float = Field(default=0.5, ge=0, le=1)
    loop_min_overlap: float = Field(default=0.5, ge=0, le=1)
    loop_max_rmse: float = Field(default=0.05, gt=0)
    max_transform_translation: float = Field(default=0.5, gt=0)
    max_transform_rotation: float = Field(default=15, gt=0, le=180)
    pose_graph_edge_prune_threshold: float = Field(default=0.25, ge=0, le=1)
    device_prior_information: float = Field(default=0.01, gt=0)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    opening: OpeningConfig = Field(default_factory=OpeningConfig)

    @model_validator(mode="after")
    def limits(self) -> Self:
        if self.depth_max_m <= self.depth_min_m:
            raise ValueError("depth_max_m must exceed depth_min_m")
        if self.min_keyframes > self.max_frames:
            raise ValueError("min_keyframes exceeds max_frames")
        if self.geometry.up_axis != "z":
            raise ValueError("LiDAR normalizes to canonical Z-up; geometry.up_axis must be z")
        return self


class FrameCounts(ContractModel):
    frames_total: int = 0
    frames_loaded: int = 0
    frames_skipped: int = 0
    frames_invalid: int = 0
    frames_not_selected: int = 0


class RegistrationRecord(ContractModel):
    source_frame: int
    target_frame: int
    kind: Literal["neighbor", "loop"]
    initial_transform: list[list[float]]
    optimized_transform: list[list[float]]
    fitness_before: float
    rmse_before: float | None
    fitness: float
    inlier_rmse: float | None
    reverse_fitness: float
    information_matrix: list[list[float]]
    accepted: bool
    rejection_reason: str | None = None
