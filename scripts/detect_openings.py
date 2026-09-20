"""Rerun opening detection from cached reconstruction artifacts."""

import argparse
from pathlib import Path

from config.settings import load_settings
from property_scanner.openings.cached import rerun_cached_openings
from property_scanner.openings.models import OpeningConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path, help="Existing outputs/<capture_id> directory")
    parser.add_argument("--config", type=Path, help="Optional OpeningConfig JSON")
    args = parser.parse_args()
    config = OpeningConfig.model_validate_json(args.config.read_text()) if args.config else OpeningConfig()
    result, detected = rerun_cached_openings(args.capture, load_settings().model_dir, config)
    if detected is None:
        print("Opening detection did not run; inspect result warnings.")
        return 2
    print(f"Openings detected: {len(result.property.openings)}")
    print(f"JSON: {args.capture / 'result.json'}")
    print(f"PNG: {args.capture / 'floorplan.png'}")
    print(f"SVG: {args.capture / 'floorplan.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
