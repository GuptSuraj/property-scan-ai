"""Typed configuration and evidence records for shared opening detection."""

from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from property_scanner.schemas.base import ContractModel

SEMANTIC_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
SEMANTIC_REVISION = "b9175de73a0a34f7843135853d27629aa6987b2f"
SEMANTIC_CACHE_NAME = "segformer_b0_ade20k"


class OpeningConfig(ContractModel):
    enabled: bool = True
    semantic_model: Literal[SEMANTIC_MODEL] = SEMANTIC_MODEL
    semantic_revision: str = SEMANTIC_REVISION
    semantic_input_size: int = Field(default=512, ge=128, le=1024)
    semantic_min_score: float = Field(default=0.55, ge=0, le=1)
    candidate_min_pixel_area: int = Field(default=150, ge=4)
    mask_morphology_size: int = Field(default=3, ge=1, le=9)
    depth_sample_stride: int = Field(default=4, ge=1, le=32)
    depth_min_m: float = Field(default=0.15, ge=0)
    depth_max_m: float = Field(default=15.0, gt=0)
    wall_association_max_distance: float = Field(default=0.18, gt=0)
    wall_endpoint_tolerance: float = Field(default=0.12, ge=0)
    robust_lower_percentile: float = Field(default=2.0, ge=0, le=25)
    robust_upper_percentile: float = Field(default=98.0, ge=75, le=100)
    minimum_3d_points: int = Field(default=25, ge=3)
    multi_view_merge_distance: float = Field(default=0.25, gt=0)
    minimum_supporting_views: int = Field(default=2, ge=1)
    single_view_min_quality: float = Field(default=0.82, ge=0, le=1)
    door_min_width: float = Field(default=0.45, gt=0)
    door_max_width: float = Field(default=2.5, gt=0)
    window_min_width: float = Field(default=0.3, gt=0)
    window_max_width: float = Field(default=5.0, gt=0)
    opening_min_height: float = Field(default=0.4, gt=0)
    max_measurement_dispersion: float = Field(default=0.25, gt=0)
    floor_contact_tolerance: float = Field(default=0.22, ge=0)
    connection_boundary_tolerance: float = Field(default=0.2, ge=0)
    occupancy_cell_size: float = Field(default=0.04, gt=0, le=0.2)
    occupancy_min_points_per_cell: int = Field(default=1, ge=1)
    occupancy_min_boundary_support: float = Field(default=0.45, ge=0, le=1)
    enable_geometry_fallback: bool = True
    cache_semantic_masks: bool = True

    @model_validator(mode="after")
    def limits(self):
        if self.depth_max_m <= self.depth_min_m:
            raise ValueError("depth_max_m must exceed depth_min_m")
        if self.robust_upper_percentile <= self.robust_lower_percentile:
            raise ValueError("robust percentiles must be ordered")
        if self.door_max_width <= self.door_min_width or self.window_max_width <= self.window_min_width:
            raise ValueError("opening maximum width must exceed minimum width")
        return self


class OpeningFrame(ContractModel):
    frame_id: str
    image_path: Path
    depth_path: Path
    intrinsics: dict[str, float | int]
    camera_to_property: list[list[float]]
    room_id: str | None = None
    depth_source: Literal["sensor", "estimated"]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class OpeningCandidate(ContractModel):
    candidate_id: str
    frame_id: str
    semantic_type: Literal["door", "window", "unknown", "open_passage"]
    bounding_box: tuple[int, int, int, int] | None = None
    pixel_area: int = Field(default=0, ge=0)
    semantic_score: float = Field(default=0, ge=0, le=1)
    wall_id: str | None = None
    room_id: str | None = None
    u_min: float | None = None
    u_max: float | None = None
    v_min: float | None = None
    v_max: float | None = None
    sample_count: int = Field(default=0, ge=0)
    wall_distance_median: float | None = None
    geometry_score: float = Field(default=0, ge=0, le=1)
    quality: float = Field(default=0, ge=0, le=1)
    accepted: bool = False
    rejection_reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class OpeningDiagnostics(ContractModel):
    frames_processed: int = 0
    candidates_total: int = 0
    candidates_accepted: int = 0
    candidates_rejected: int = 0
    openings_fused: int = 0
    semantic_device: str | None = None
    depth_source_counts: dict[str, int] = Field(default_factory=dict)
    warnings_by_code: dict[str, int] = Field(default_factory=dict)

