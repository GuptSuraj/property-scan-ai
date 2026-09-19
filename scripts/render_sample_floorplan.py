"""Generate an explicitly synthetic architectural drawing for visual inspection."""
import argparse
from pathlib import Path
from property_scanner.core.exceptions import PropertyScannerError
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.schemas.geometry import Room, Wall, FloorSurface, CeilingSurface, Opening
from property_scanner.schemas.primitives import Polygon2D


def sample_room(shape: str = "rectangle", *, ceiling: bool = True) -> tuple[Room, list[Opening]]:
    """Known synthetic measurements only; no measured property or detection claims."""
    if shape == "rectangle":
        vertices = [(0, 0), (4, 0), (4, 5), (0, 5)]
        lengths, area = [4, 5, 4, 5], 20
    else:
        vertices = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 5), (0, 5)]
        lengths, area = [5, 2, 2, 3, 3, 5], 19
    polygon = Polygon2D(points=[{"x": x, "y": y} for x, y in vertices])
    walls = [Wall(wall_id=f"wall_{i+1}", room_id="room_01", start_point=polygon.points[i],
                  end_point=polygon.points[(i+1) % len(vertices)], length={"value": length})
             for i, length in enumerate(lengths)]
    openings = [
        Opening(opening_id="door_01", type="door", wall_id="wall_1", room_ids=["room_01"],
                width={"value": 0.9}, position_along_wall={"value": 1.2}),
        Opening(opening_id="window_01", type="window", wall_id=f"wall_{len(walls)}", room_ids=["room_01"],
                width={"value": 1.4}, position_along_wall={"value": 1.6}),
    ]
    return Room(room_id="room_01", name="Sample room", polygon=polygon, walls=walls,
                floor=FloorSurface(surface_id="floor_01", area={"value": area}),
                ceiling=CeilingSurface(surface_id="ceiling_01", height={"value": 2.8}) if ceiling else None,
                metadata={"synthetic": True}), openings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["rectangle", "l"], default="rectangle")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sample_floorplan"))
    args = parser.parse_args()
    room, openings = sample_room(args.shape)
    try:
        drawing = FloorPlanRenderer().render_room(room, openings=openings, title="Synthetic sample · not a real scan")
        try:
            files = drawing.save(args.output_dir)
            for warning in drawing.warnings:
                print(f"Warning [{warning.code}]: {warning.message}")
        finally:
            drawing.close()
    except PropertyScannerError as exc:
        parser.exit(2, f"Error: {exc}\n")
    print("Sample floor plan generated successfully.")
    print(f"PNG: {files.png}")
    print(f"SVG: {files.svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
