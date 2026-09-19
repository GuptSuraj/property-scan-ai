"""Render supplied typed JSON; no point-cloud processing or reconstruction."""
import argparse
from pathlib import Path
from uuid import uuid4
from pydantic import ValidationError
from property_scanner.core.exceptions import PropertyScannerError
from property_scanner.geometry.models import RoomGeometryResult
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.schemas.geometry import PropertyGeometry
from property_scanner.schemas.result import PropertyScanResult


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-kind", choices=["geometry", "property", "result"], default="geometry")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, help="RenderingConfig JSON file")
    parser.add_argument("--name", help="Optional name for single-room geometry")
    args = parser.parse_args()
    try:
        config = RenderingConfig.model_validate_json(args.config.read_text()) if args.config else RenderingConfig()
        renderer = FloorPlanRenderer(config)
        payload = args.input.read_text(encoding="utf-8")
        capture_id = str(uuid4())
        if args.input_kind == "geometry":
            drawing = renderer.render_room(RoomGeometryResult.model_validate_json(payload), name=args.name)
        elif args.input_kind == "property":
            drawing = renderer.render_property(PropertyGeometry.model_validate_json(payload))
        else:
            result = PropertyScanResult.model_validate_json(payload)
            capture_id = str(result.capture.capture_id)
            drawing = renderer.render_property(result.property)
        try:
            files = drawing.save(args.output_dir or Path("outputs")/capture_id)
            for warning in drawing.warnings:
                print(f"Warning [{warning.code}]: {warning.message}")
        finally:
            drawing.close()
    except (PropertyScannerError, ValidationError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Floor plan generated.\nPNG: {files.png}\nSVG: {files.svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
