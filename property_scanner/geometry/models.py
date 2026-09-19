"""Internal results contain only measured geometry and explicitly heuristic quality."""
from typing import Literal
from pydantic import Field
from property_scanner.schemas.base import ContractModel, ConfidenceScore, NonNegativeNumber
from property_scanner.schemas.primitives import Point2D, Point3D, Polygon2D
from property_scanner.schemas.measurements import LengthMeasurement, AreaMeasurement
from property_scanner.schemas.geometry import Room, Wall, FloorSurface, CeilingSurface
from property_scanner.schemas.result import ResultWarning


class DetectedPlane(ContractModel):
    plane_id: str
    equation: tuple[float, float, float, float]
    normal: Point3D
    centroid: Point3D
    inlier_count: int = Field(ge=0)
    inlier_ratio: ConfidenceScore
    area_estimate: NonNegativeNumber
    orientation: Literal["horizontal", "vertical", "other"]
    residual_rmse: NonNegativeNumber
    width: NonNegativeNumber
    height_coverage: NonNegativeNumber
    orientation_degrees: float
    quality_score: ConfidenceScore


class WallSegment2D(ContractModel):
    wall_id: str
    source_plane_id: str
    start: Point2D
    end: Point2D
    length: LengthMeasurement
    quality_score: ConfidenceScore


class RoomCorner(ContractModel):
    point: Point2D
    wall_ids: list[str]
    quality_score: ConfidenceScore


class GeometryDiagnostics(ContractModel):
    original_points: int
    downsampled_points: int
    filtered_points: int
    floor_inlier_ratio: ConfidenceScore
    ceiling_inlier_ratio: ConfidenceScore | None = None
    wall_inlier_ratio: ConfidenceScore
    explained_point_ratio: ConfidenceScore
    detected_wall_count: int
    rejected_wall_candidates: int
    merged_wall_candidates: int
    polygon_closure_error: NonNegativeNumber | None = None
    plane_residual_rmse: dict[str, float]


class RoomGeometryResult(ContractModel):
    floor_plane: DetectedPlane
    ceiling_plane: DetectedPlane | None = None
    ceiling_height: LengthMeasurement | None = None
    wall_planes: list[DetectedPlane]
    wall_segments_2d: list[WallSegment2D]
    corners: list[RoomCorner]
    room_polygon: Polygon2D | None = None
    floor_area: AreaMeasurement | None = None
    diagnostics: GeometryDiagnostics
    warnings: list[ResultWarning] = Field(default_factory=list)
    source_to_floor_transform: list[list[float]]

    def to_room(self, room_id: str) -> Room:
        """Convert known geometry only; do not invent openings, damage, or capture."""
        return Room(
            room_id=room_id, polygon=self.room_polygon,
            floor=FloorSurface(surface_id=f"{room_id}:floor", polygon=self.room_polygon, area=self.floor_area),
            ceiling=(CeilingSurface(surface_id=f"{room_id}:ceiling", height=self.ceiling_height)
                     if self.ceiling_plane is not None else None),
            walls=[Wall(wall_id=f"{room_id}:{s.wall_id}", room_id=room_id,
                        start_point=s.start, end_point=s.end, length=s.length,
                        metadata={"source_plane_id": s.source_plane_id, "geometry_quality_score": s.quality_score})
                   for s in self.wall_segments_2d],
        )
