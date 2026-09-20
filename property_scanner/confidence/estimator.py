"""Evidence-based uncertainty with optional benchmark calibration profiles."""
from pathlib import Path
from pydantic import Field
from property_scanner.pipeline.context import ScanContext
from property_scanner.schemas.base import ContractModel
from property_scanner.schemas.measurements import ConfidenceMethod, MeasuredValue
from property_scanner.schemas.result import PropertyScanResult


class ConfidenceConfig(ContractModel):
    confidence_level: float = Field(default=.95, gt=0, lt=1)
    profile_directory: Path = Path("confidence_profiles")
    allow_tier_prior_uncalibrated: bool = False
    tier_relative_error: dict[str, float] = Field(default_factory=lambda: {"photo": .10, "video": .08, "lidar": .04})


class ConfidenceProfile(ContractModel):
    tier: str
    confidence_level: float = Field(default=.95, gt=0, lt=1)
    absolute_error_by_type: dict[str, float]
    benchmark_rows: int = Field(ge=1)


def _interval(value: MeasuredValue, half_width, method, level):
    value.lower_bound = max(0.0, value.value-float(half_width)); value.upper_bound = value.value+float(half_width)
    value.confidence_level = level; value.confidence_method = method


class ConfidenceEstimator:
    def __init__(self, config=None): self.config = config or ConfidenceConfig()

    def _profile(self, tier):
        path = self.config.profile_directory/f"{tier}.json"
        return ConfidenceProfile.model_validate_json(path.read_text()) if path.is_file() else None

    def apply(self, result: PropertyScanResult) -> PropertyScanResult:
        profile = self._profile(result.capture.tier.value); measures = []
        for room in result.property.rooms:
            for wall in room.walls:
                if wall.length: measures.append(("wall_length", wall.length, wall.metadata))
            if room.floor and room.floor.area: measures.append(("floor_area", room.floor.area, room.floor.metadata))
            if room.ceiling and room.ceiling.height: measures.append(("ceiling_height", room.ceiling.height, room.ceiling.metadata))
        for opening in result.property.openings:
            for kind, value in (("opening_width", opening.width), ("opening_height", opening.height), ("sill_height", opening.sill_height)):
                if value: measures.append((kind, value, opening.metadata))
        for damage in result.damages:
            if damage.metric_area: measures.append(("damage_area", damage.metric_area, damage.metadata))
            if damage.metric_length: measures.append(("damage_length", damage.metric_length, damage.metadata))
        for kind, value, metadata in measures:
            value.confidence_method = ConfidenceMethod.UNAVAILABLE
            if profile and kind in profile.absolute_error_by_type:
                _interval(value, profile.absolute_error_by_type[kind], ConfidenceMethod.BENCHMARK_CALIBRATED, profile.confidence_level); continue
            key = "area_mad_m2" if "area" in kind else "width_mad_m" if kind == "opening_width" else "height_mad_m"
            mad = metadata.get(key)
            if isinstance(mad, (int, float)) and mad > 0:
                _interval(value, 1.96*1.4826*mad, ConfidenceMethod.MULTI_VIEW_DISPERSION, self.config.confidence_level); continue
            residual = metadata.get("residual_m") or metadata.get("surface_residual_m")
            if isinstance(residual, (int, float)) and residual > 0:
                _interval(value, 1.96*residual, ConfidenceMethod.GEOMETRY_RESIDUAL, self.config.confidence_level); continue
            if self.config.allow_tier_prior_uncalibrated:
                _interval(value, value.value*self.config.tier_relative_error[result.capture.tier.value],
                    ConfidenceMethod.TIER_PRIOR_UNCALIBRATED, self.config.confidence_level)
        if result.processing_info:
            if "confidence_estimation" not in result.processing_info.modules_used: result.processing_info.modules_used.append("confidence_estimation")
            result.processing_info.metadata["confidence"] = {"profile_loaded": profile is not None,
                "methods": sorted({value.confidence_method.value for _, value, _ in measures})}
        return result

    def run(self, context: ScanContext) -> ScanContext:
        if context.result is None: raise NotImplementedError("Confidence estimation is not implemented without a unified result")
        context.result = self.apply(context.result); return context
