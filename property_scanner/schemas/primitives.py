"""Simple metric geometry containers; no reconstruction or topology algorithms."""

from typing import Self

from pydantic import Field, model_validator

from property_scanner.schemas.base import ContractModel, FiniteNumber


class Point2D(ContractModel):
    """XY position in meters in the coordinate frame documented by the owner."""

    x: FiniteNumber
    y: FiniteNumber


class Point3D(Point2D):
    """XYZ position in meters; property coordinates use Z up."""

    z: FiniteNumber


class Polygon2D(ContractModel):
    """Ordered vertices; implicit closure, with an optional repeated final vertex.

    At least three distinct points are required. Winding, intersections, holes,
    collinearity, and planarity are not evaluated in this schema phase.
    """

    points: list[Point2D] = Field(min_length=3, description="Ordered metric vertices; no pixels.")

    @model_validator(mode="after")
    def distinct_vertices(self) -> Self:
        if len({(point.x, point.y) for point in self.points}) < 3:
            raise ValueError("A polygon requires at least 3 distinct points")
        return self


class BoundingBox2D(ContractModel):
    """Axis-aligned metric box in the owner's coordinate frame."""

    min_point: Point2D
    max_point: Point2D

    @model_validator(mode="after")
    def ordered_bounds(self) -> Self:
        if self.min_point.x > self.max_point.x or self.min_point.y > self.max_point.y:
            raise ValueError("Bounding box minimum must not exceed maximum")
        return self


class BoundingBox3D(ContractModel):
    """Axis-aligned metric box in the property frame."""

    min_point: Point3D
    max_point: Point3D

    @model_validator(mode="after")
    def ordered_bounds(self) -> Self:
        if any(getattr(self.min_point, axis) > getattr(self.max_point, axis) for axis in ("x", "y", "z")):
            raise ValueError("Bounding box minimum must not exceed maximum")
        return self
