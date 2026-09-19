"""Argument parsing and friendly errors for the preparation-only CLI."""

import argparse
from pathlib import Path
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
        description="Prepare a property capture. Reconstruction and AI processing are not implemented."
    )
    parser.add_argument("--tier", required=True, choices=[tier.value for tier in InputTier])
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Input path, relative to the current working directory or absolute.",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        configure_logging(settings.log_level)
        adapter = select_adapter(args.tier, args.input)
        pipeline = PropertyScanPipeline(settings)
        result = pipeline.prepare(adapter)
    except PropertyScannerError as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Capture: {result.capture.capture_id}")
    print(f"Status: {result.status}")
    print(f"Output: {result.output_dir}")
    print(result.message)
    return 0
