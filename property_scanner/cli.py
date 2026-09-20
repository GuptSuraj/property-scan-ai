"""Capture preparation plus local video and canonical LiDAR reconstruction."""

import argparse
from pathlib import Path
from pydantic import ValidationError
from collections.abc import Sequence

from config.settings import load_settings
from property_scanner.core.exceptions import PropertyScannerError
from property_scanner.core.logging import configure_logging
from property_scanner.inputs import select_adapter
from property_scanner.pipeline.processor import PropertyScanPipeline
from property_scanner.schemas.common import InputTier


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 for successful preparation, 2 for expected user errors."""
    parser = argparse.ArgumentParser(
        description="Prepare photo inputs or reconstruct videos and canonical LiDAR RGB-D captures."
    )
    parser.add_argument("--tier", required=True, choices=[tier.value for tier in InputTier])
    parser.add_argument("--drift-correction", choices=["on", "off"], default=None, help="LiDAR correction mode; canonical captures default to on")
    parser.add_argument("--lidar-config", type=Path, help="Optional LidarConfig JSON")
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Input path, relative to the current working directory or absolute.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Validate input paths and allocate output without reconstruction")
    parser.add_argument("--video-config", type=Path, help="Optional VideoConfig JSON")
    parser.add_argument("--max-keyframes", type=int, help="Video keyframe budget override")
    parser.add_argument("--extraction-fps", type=float, help="Video candidate-frame extraction rate override")
    parser.add_argument("--no-registration-refinement", action="store_true", help="Disable video ICP/pose-graph refinement")
    args = parser.parse_args(argv)
    video_options = (args.video_config is not None or args.max_keyframes is not None
                     or args.extraction_fps is not None or args.no_registration_refinement)
    if video_options and args.tier != "video":
        parser.error("Video options require --tier video")
    if args.tier != "lidar" and (args.drift_correction is not None or args.lidar_config is not None):
        parser.error("LiDAR options require --tier lidar")
    try:
        settings = load_settings()
        configure_logging(settings.log_level)
        adapter = select_adapter(args.tier, args.input)
        pipeline = PropertyScanPipeline(settings)
        result = pipeline.prepare(adapter)
        canonical = args.tier == "lidar" and ((result.capture.source_path / "manifest.json").is_file() or args.drift_correction is not None or args.lidar_config is not None)
        if args.tier == "video" and not args.prepare_only:
            from property_scanner.reconstruction.video.models import VideoConfig
            config = VideoConfig.model_validate_json(args.video_config.read_text()) if args.video_config else VideoConfig()
            updates = {}
            if args.max_keyframes is not None:
                updates["max_keyframes"] = args.max_keyframes
            if args.extraction_fps is not None:
                updates["extraction_fps"] = args.extraction_fps
            if args.no_registration_refinement:
                updates["enable_icp_refinement"] = False
            if updates:
                config = VideoConfig.model_validate({**config.model_dump(), **updates})
            scan = pipeline.process(result, video_config=config)
            print(f"Capture: {scan.capture.capture_id}\nOutput: {result.output_dir}\nJSON: {result.output_dir / 'result.json'}")
            print("Status: completed_with_processing_errors" if scan.processing_info.errors else "Status: processed")
            return 2 if scan.processing_info.errors else 0
        if canonical and not args.prepare_only:
            from property_scanner.reconstruction.lidar.models import LidarConfig
            config = LidarConfig.model_validate_json(args.lidar_config.read_text()) if args.lidar_config else LidarConfig()
            if args.drift_correction is not None:
                config = LidarConfig.model_validate({**config.model_dump(), "drift_correction": args.drift_correction})
            scan = pipeline.process(result, lidar_config=config)
            print(f"Capture: {scan.capture.capture_id}\nOutput: {result.output_dir}")
            print("Status: completed_with_processing_errors" if scan.processing_info.errors else "Status: processed")
            print(f"JSON: {result.output_dir / 'result.json'}")
            return 0
    except (PropertyScannerError, ValidationError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Capture: {result.capture.capture_id}")
    print(f"Status: {result.status}")
    print(f"Output: {result.output_dir}")
    print(result.message)
    return 0
