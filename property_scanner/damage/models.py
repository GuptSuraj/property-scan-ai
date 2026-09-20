"""Typed configuration and intermediate records for shared damage analysis."""

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from property_scanner.schemas.base import ContractModel
from property_scanner.schemas.damage import DamageType

DAMAGE_MODEL_FILE = "yoloe-11s-seg-pf.pt"
DAMAGE_PROMPTS = (
    "wall crack", "water stain", "water damage", "mold", "hole in wall",
    "peeling paint", "damaged wall",
)


class DamageConfig(ContractModel):
    enabled: bool = True
    model_file: str = DAMAGE_MODEL_FILE
    prompts: tuple[str, ...] = DAMAGE_PROMPTS
    semantic_min_score: float = Field(default=0.35, ge=0, le=1)
    minimum_pixel_area: int = Field(default=80, ge=4)
    depth_sample_stride: int = Field(default=4, ge=1, le=32)
    minimum_3d_points: int = Field(default=20, ge=3)
    depth_min_m: float = Field(default=0.15, ge=0)
    depth_max_m: float = Field(default=15.0, gt=0)
    surface_max_distance_m: float = Field(default=0.18, gt=0)
    robust_lower_percentile: float = Field(default=2.0, ge=0, le=25)
    robust_upper_percentile: float = Field(default=98.0, ge=75, le=100)
    multi_view_merge_distance_m: float = Field(default=0.3, gt=0)
    minimum_supporting_views: int = Field(default=2, ge=1)


class DamageFrame(ContractModel):
    frame_id: str
    image_path: Path
    depth_path: Path
    intrinsics: dict[str, float | int]
    camera_to_property: list[list[float]]
    room_id: str | None = None
    depth_source: Literal["sensor", "estimated"]


class DamagePrediction:
    """Image-aligned segmentation instance returned by a local vision backend."""

    def __init__(self, damage_type: DamageType, mask: np.ndarray, score: float) -> None:
        self.damage_type = damage_type
        self.mask = np.asarray(mask, dtype=bool)
        self.score = float(score)


class DamageCandidate:
    def __init__(self, candidate_id: str, frame_id: str, damage_type: DamageType,
                 score: float, room_id: str | None, surface_id: str,
                 coordinates: np.ndarray, residual: float) -> None:
        self.candidate_id = candidate_id
        self.frame_id = frame_id
        self.damage_type = damage_type
        self.score = score
        self.room_id = room_id
        self.surface_id = surface_id
        self.coordinates = coordinates
        self.residual = residual
