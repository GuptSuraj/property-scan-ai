"""Development CLI for a reconstructed metric single-room point cloud."""
import argparse
from pathlib import Path
from uuid import uuid4
from pydantic import ValidationError
from property_scanner.core.exceptions import PropertyScannerError
from property_scanner.core.logging import configure_logging
from property_scanner.geometry.config import GeometryConfig
from property_scanner.geometry.engine import GeometryEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="GeometryConfig JSON file")
    parser.add_argument("--up-axis", choices=["x", "y", "z", "-x", "-y", "-z"])
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    configure_logging()
    try:
        config = GeometryConfig.model_validate_json(args.config.read_text()) if args.config else GeometryConfig()
        if args.up_axis:
            config = GeometryConfig.model_validate({**config.model_dump(), "up_axis": args.up_axis})
        directory = args.output_dir / str(uuid4()) / "diagnostics" / "geometry" if args.diagnostics else None
        result = GeometryEngine(config).process_point_cloud(args.input, diagnostics_dir=directory)
    except (PropertyScannerError, ValidationError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print("Geometry processing complete")
    print(f"Walls detected: {len(result.wall_planes)}")
    print("Wall lengths: " + ", ".join(f"{s.length.value:.3f} m" for s in result.wall_segments_2d))
    print(f"Floor area: {result.floor_area.value:.3f} m²" if result.floor_area else "Floor area: unavailable")
    print(f"Ceiling height: {result.ceiling_height.value:.3f} m" if result.ceiling_height else "Ceiling height: unavailable")
    for warning in result.warnings:
        print(f"Warning [{warning.code}]: {warning.message}")
    if directory:
        print(f"Geometry diagnostics: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
