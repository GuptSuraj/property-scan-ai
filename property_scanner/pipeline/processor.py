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
from property_scanner.reconstruction.lidar.models import LidarConfig
from property_scanner.stitching.engine import StitchingEngine
from property_scanner.reconstruction.video.models import VideoConfig
from property_scanner.reconstruction.photo.models import PhotoConfig

logger = logging.getLogger(__name__)


class PropertyScanPipeline:
    """Shared entry point for photo rooms, metric video, and canonical LiDAR."""

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

    def process(self, prepared: PreparationResult, *, lidar_config: LidarConfig | None = None,
                video_config: VideoConfig | None = None, photo_config: PhotoConfig | None = None,
                skip_damage: bool = False) -> PropertyScanResult:
        """Dispatch implemented reconstruction modes into shared downstream stages."""
        if prepared.capture.tier == "lidar" and ((prepared.capture.source_path / "manifest.json").is_file() or lidar_config is not None):
            try:
                from property_scanner.reconstruction.lidar.pipeline import process_lidar
            except ImportError as exc:
                from property_scanner.core.exceptions import ConfigurationError
                raise ConfigurationError("Install LiDAR dependencies: pip install -e '.[lidar]'") from exc
            result = process_lidar(prepared, lidar_config or LidarConfig(), model_dir=self.settings.model_dir)
            from property_scanner.pipeline.finalize import finalize_result
            return finalize_result(prepared.output_dir, result, self.settings.model_dir, skip_damage=skip_damage)
        if prepared.capture.tier == "video" and video_config is not None:
            try:
                from property_scanner.reconstruction.video.pipeline import process_video
            except ImportError as exc:
                from property_scanner.core.exceptions import ConfigurationError
                raise ConfigurationError("Install video dependencies: pip install -e '.[video]'") from exc
            result = process_video(prepared, video_config, self.settings.model_dir)
            from property_scanner.pipeline.finalize import finalize_result
            return finalize_result(prepared.output_dir, result, self.settings.model_dir, skip_damage=skip_damage)
        if prepared.capture.tier == "photo" and photo_config is not None:
            try:
                from property_scanner.reconstruction.photo.pipeline import process_photo
            except ImportError as exc:
                from property_scanner.core.exceptions import ConfigurationError
                raise ConfigurationError("Install photo dependencies: pip install -e '.[photo]'") from exc
            result = process_photo(prepared, photo_config, self.settings.model_dir)
            from property_scanner.pipeline.finalize import finalize_result
            return finalize_result(prepared.output_dir, result, self.settings.model_dir, skip_damage=skip_damage)
        context = ScanContext(capture=prepared.capture, output_dir=prepared.output_dir)
        for stage in self.stages:
            context = stage.run(context)
        if context.result is None:
            raise NotImplementedError("Unified result assembly is not implemented yet.")
        return context.result
