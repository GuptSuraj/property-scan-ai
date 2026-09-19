"""Metric thresholds for single-room structural geometry."""
from typing import Literal, Self
from pydantic import Field, model_validator
from property_scanner.schemas.base import ContractModel


class GeometryConfig(ContractModel):
    up_axis: Literal["x", "y", "z", "-x", "-y", "-z"] = "z"
    voxel_size: float = Field(default=0.03, gt=0)
    statistical_nb_neighbors: int = Field(default=20, ge=2)
    statistical_std_ratio: float = Field(default=2.5, gt=0)
    remove_outliers: bool = True
    normal_radius: float = Field(default=0.15, gt=0)
    normal_max_neighbors: int = Field(default=30, ge=3)
    min_points: int = Field(default=200, ge=3)
    max_downsampled_points: int = Field(default=300000, ge=200)
    ransac_distance_threshold: float = Field(default=0.025, gt=0)
    ransac_n: int = Field(default=3, ge=3)
    ransac_iterations: int = Field(default=1000, ge=1)
    random_seed: int = Field(default=7, ge=0)
    max_planes: int = Field(default=24, ge=3, le=100)
    horizontal_angle_tolerance: float = Field(default=12, gt=0, lt=45)
    vertical_angle_tolerance: float = Field(default=12, gt=0, lt=45)
    ceiling_parallel_tolerance: float = Field(default=5, gt=0, lt=45)
    min_plane_inliers: int = Field(default=100, ge=3)
    min_horizontal_area: float = Field(default=1, gt=0)
    min_floor_coverage: float = Field(default=0.2, gt=0, le=1)
    min_ceiling_coverage: float = Field(default=0.4, gt=0, le=1)
    floor_elevation_tolerance: float = Field(default=0.25, gt=0)
    ceiling_top_tolerance: float = Field(default=0.3, gt=0)
    min_ceiling_height: float = Field(default=2, gt=0)
    max_ceiling_height: float = Field(default=5, gt=0)
    min_wall_length: float = Field(default=0.7, gt=0)
    min_wall_height: float = Field(default=1.8, gt=0)
    wall_base_tolerance: float = Field(default=0.35, gt=0)
    wall_merge_angle_tolerance: float = Field(default=5, gt=0, lt=45)
    wall_merge_distance_tolerance: float = Field(default=0.06, gt=0)
    wall_merge_gap_tolerance: float = Field(default=0.08, ge=0)
    corner_merge_tolerance: float = Field(default=0.04, gt=0)
    intersection_extension: float = Field(default=0.3, gt=0)
    min_intersection_angle: float = Field(default=15, gt=0, lt=90)
    min_polygon_area: float = Field(default=1, gt=0)
    extent_quantile: float = Field(default=0.005, ge=0, lt=0.1)

    @model_validator(mode="after")
    def consistent_limits(self) -> Self:
        if self.max_ceiling_height <= self.min_ceiling_height:
            raise ValueError("max_ceiling_height must exceed min_ceiling_height")
        if self.ransac_n > self.min_plane_inliers:
            raise ValueError("ransac_n must not exceed min_plane_inliers")
        return self
