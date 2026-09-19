"""Separate working preparation from deliberately unavailable processing."""

import logging

from config.settings import Settings
from property_scanner.confidence.estimator import ConfidenceEstimator
from property_scanner.core.paths import create_capture_output
from property_scanner.damage.detector import DamageDetector
from property_scanner.geometry.engine import GeometryEngine
from property_scanner.inputs.base import BaseInputAdapter
from property_scanner.measurements.calculator import MeasurementCalculator
from property_scanner.openings.detector import OpeningDetector
from property_scanner.pipeline.context import ProcessingStage, ScanContext
from property_scanner.reconstruction.base import SceneReconstructor
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.schemas.common import PreparationResult
from property_scanner.schemas.result import PropertyScanResult
from property_scanner.stitching.engine import StitchingEngine

logger = logging.getLogger(__name__)


class PropertyScanPipeline:
    """One orchestrator for all tiers; only prepare() works in this phase."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stages: tuple[ProcessingStage, ...] = (
            SceneReconstructor(),
            GeometryEngine(),
            StitchingEngine(),
            OpeningDetector(),
            DamageDetector(),
            MeasurementCalculator(),
            ConfidenceEstimator(),
            FloorPlanRenderer(),
        )
        logger.info("Initialized property scan pipeline")

    def prepare(self, adapter: BaseInputAdapter) -> PreparationResult:
        """Validate a capture and allocate its directory without processing it."""
        logger.info("Loading %s capture", adapter.tier.value)
        capture = adapter.prepare()
        logger.info("Input validation complete")
        output_dir = create_capture_output(self.settings.output_dir, capture.capture_id)
        logger.info("Output directory created: %s", output_dir)
        return PreparationResult(capture=capture, output_dir=output_dir)

    def process(self, prepared: PreparationResult) -> PropertyScanResult:
        """Reserved downstream flow; currently raises at reconstruction.

        The CLI intentionally calls only prepare(). No success-shaped scan result
        can be returned by the unimplemented stages.
        """
        context = ScanContext(capture=prepared.capture, output_dir=prepared.output_dir)
        for stage in self.stages:
            context = stage.run(context)
        if context.result is None:
            raise NotImplementedError("Unified result assembly is not implemented yet.")
        return context.result
