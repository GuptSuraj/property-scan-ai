"""Rerun evidence matching and room stitching from cached photo artifacts."""

import argparse
from pathlib import Path

from property_scanner.stitching.cached import stitch_cached_output
from property_scanner.stitching.models import StitchingConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Existing outputs/<capture_id> directory")
    parser.add_argument("--config", type=Path, help="Optional StitchingConfig JSON")
    args = parser.parse_args()
    config = StitchingConfig.model_validate_json(args.config.read_text()) if args.config else StitchingConfig()
    result = stitch_cached_output(args.input, config)
    print(f"Rooms positioned: {result.diagnostics.rooms_positioned}/{result.diagnostics.rooms_total}")
    print(f"Layout valid: {result.diagnostics.valid_layout}")
    print(f"Diagnostics: {args.input / 'stitching'}")
    if result.diagnostics.valid_layout:
        print(f"PNG: {args.input / 'floorplan.png'}")
        print(f"SVG: {args.input / 'floorplan.svg'}")
    return 0 if result.diagnostics.valid_layout else 2


if __name__ == "__main__":
    raise SystemExit(main())
