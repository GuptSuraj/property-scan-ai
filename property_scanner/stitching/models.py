"""Typed inputs and diagnostics for evidence-based rigid room stitching."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from property_scanner.schemas.base import ContractModel


class StitchingConfig(ContractModel):
    cross_room_min_matches: int = Field(default=30, ge=4)
    cross_room_min_inliers: int = Field(default=12, ge=3)
    cross_room_min_inlier_ratio: float = Field(default=0.25, ge=0, le=1)
    feature_ratio_test: float = Field(default=0.75, gt=0, lt=1)
    correspondence_ransac_threshold_m: float = Field(default=0.18, gt=0)
    correspondence_ransac_iterations: int = Field(default=400, ge=10)
    registration_max_distance: float = Field(default=0.15, gt=0)
    registration_min_fitness: float = Field(default=0.25, ge=0, le=1)
    registration_max_rmse: float = Field(default=0.12, gt=0)
    enable_point_cloud_refinement: bool = True
    connection_acceptance_threshold: float = Field(default=0.35, ge=0, le=1)
    enable_global_optimization: bool = True
    optimization_loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "huber"
    optimization_max_iterations: int = Field(default=300, ge=10)
    max_overlap_area_m2: float = Field(default=0.08, ge=0)
    max_overlap_ratio: float = Field(default=0.02, ge=0, le=1)
    boundary_distance_tolerance: float = Field(default=0.35, ge=0)
    min_room_scale_quality: float = Field(default=0.4, ge=0, le=1)
    max_room_scale_relative_mad: float = Field(default=0.25, gt=0)
    cycle_error_threshold: float = Field(default=0.35, gt=0)
    enable_manhattan_regularization: bool = False
    manhattan_angle_tolerance: float = Field(default=5.0, ge=0, le=20)
    max_candidate_room_pairs: int = Field(default=100, ge=1)
    diagnostics: bool = True


class RoomToPropertyTransform(ContractModel):
    room_id: str
    translation_x: float
    translation_y: float
    rotation_yaw: float = Field(description="Counterclockwise radians")
    quality: float = Field(default=1.0, ge=0, le=1)
    source_connection_ids: list[str] = Field(default_factory=list)


class RoomConnectionEvidence(ContractModel):
    """A validated constraint. relative_transform maps room B into room A."""

    connection_id: str
    room_a_id: str
    room_b_id: str
    image_a: str | None = None
    image_b: str | None = None
    raw_match_count: int = Field(default=0, ge=0)
    inlier_match_count: int = Field(default=0, ge=0)
    inlier_ratio: float = Field(default=0, ge=0, le=1)
    relative_transform: list[list[float]] | None = None
    point_cloud_registration_fitness: float | None = Field(default=None, ge=0, le=1)
    point_cloud_registration_rmse: float | None = Field(default=None, ge=0)
    visual_score: float = Field(default=0, ge=0, le=1)
    geometry_score: float = Field(default=0, ge=0, le=1)
    combined_score: float = Field(default=0, ge=0, le=1)
    accepted: bool = False
    rejection_reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_constraint(self):
        if self.room_a_id == self.room_b_id:
            raise ValueError("Connection evidence must join two different rooms")
        if self.relative_transform is not None:
            if len(self.relative_transform) != 3 or any(len(row) != 3 for row in self.relative_transform):
                raise ValueError("relative_transform must be a 3x3 SE(2) matrix")
        return self


class StitchingDiagnostics(ContractModel):
    rooms_total: int
    rooms_positioned: int
    rooms_unpositioned: int
    candidate_edges: int
    accepted_edges: int
    rejected_edges: int
    root_room_id: str | None = None
    graph_connected: bool = False
    connected_components: list[list[str]] = Field(default_factory=list)
    cycle_residual: float | None = None
    overlap_area_total: float = 0
    maximum_pair_overlap: float = 0
    maximum_connected_boundary_distance: float | None = None
    layout_constraint_residual: float | None = None
    room_area_sum_m2: float | None = None
    polygon_union_area_m2: float | None = None
    footprint_component_count: int = 0
    valid_layout: bool = False
