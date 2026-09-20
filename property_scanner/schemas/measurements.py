"""SI measurements with optional, evidence-supplied uncertainty."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from property_scanner.schemas.base import (
    ConfidenceScore, ContractModel, NonEmptyText, NonNegativeNumber,
)


class MeasurementUnit(StrEnum):
    METER = "m"
    SQUARE_METER = "m2"
    DEGREE = "degree"
    ITEM = "item"


class ConfidenceMethod(StrEnum):
    BENCHMARK_CALIBRATED = "benchmark_calibrated"
    MULTI_VIEW_DISPERSION = "multi_view_dispersion"
    GEOMETRY_RESIDUAL = "geometry_residual"
    TIER_PRIOR_UNCALIBRATED = "tier_prior_uncalibrated"
    UNAVAILABLE = "unavailable"


class MeasuredValue(ContractModel):
    """Known nonnegative value; absence of the whole model means unknown.

    Bounds, interval coverage, and confidence score are independently optional.
    No uncertainty is estimated or filled in by this schema.
    """

    value: NonNegativeNumber = Field(description="Observed or computed value in the declared unit.")
    unit: MeasurementUnit
    lower_bound: NonNegativeNumber | None = Field(default=None, description="Lower bound in the same unit.")
    upper_bound: NonNegativeNumber | None = Field(default=None, description="Upper bound in the same unit.")
    confidence_level: ConfidenceScore | None = Field(default=None, description="Coverage probability of an interval, if known.")
    confidence_score: ConfidenceScore | None = Field(default=None, description="Evidence quality score; not interval coverage.")
    method: NonEmptyText | None = Field(default=None, description="Provenance or calibration method; no default method is assumed.")
    confidence_method: ConfidenceMethod = Field(default=ConfidenceMethod.UNAVAILABLE,
        description="Explicit uncertainty source; uncalibrated evidence is never labelled benchmark-calibrated.")

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.lower_bound is not None and self.lower_bound > self.value:
            raise ValueError("lower_bound must be <= value")
        if self.upper_bound is not None and self.value > self.upper_bound:
            raise ValueError("value must be <= upper_bound")
        return self


class LengthMeasurement(MeasuredValue):
    """Distance in meters; cannot accidentally accept an area or imperial unit."""

    unit: Literal[MeasurementUnit.METER] = MeasurementUnit.METER


class AreaMeasurement(MeasuredValue):
    """Area in square meters."""

    unit: Literal[MeasurementUnit.SQUARE_METER] = MeasurementUnit.SQUARE_METER


class AngleMeasurement(MeasuredValue):
    """Counterclockwise orientation in degrees from property +X, in [0, 360]."""

    unit: Literal[MeasurementUnit.DEGREE] = MeasurementUnit.DEGREE
    value: NonNegativeNumber = Field(le=360)
    lower_bound: NonNegativeNumber | None = Field(default=None, le=360)
    upper_bound: NonNegativeNumber | None = Field(default=None, le=360)


class QuantityMeasurement(MeasuredValue):
    """Scope quantity with its unit stored once; pricing is outside the contract."""

    unit: Literal[MeasurementUnit.METER, MeasurementUnit.SQUARE_METER, MeasurementUnit.ITEM]
