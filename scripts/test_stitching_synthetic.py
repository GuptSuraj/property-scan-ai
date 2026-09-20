"""Generate a deterministic three-room property plan through the stitcher."""

import argparse
from pathlib import Path
import numpy as np
from uuid import UUID

from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.stitching.diagnostics import export_stitching_diagnostics
from property_scanner.stitching.engine import MultiRoomStitcher
from property_scanner.stitching.models import RoomConnectionEvidence
from property_scanner.stitching.transforms import se2
from property_scanner.schemas.geometry import CeilingSurface, FloorSurface, Room, Wall
from property_scanner.schemas.measurements import AreaMeasurement, LengthMeasurement
from property_scanner.schemas.primitives import Point2D, Polygon2D
from property_scanner.schemas.capture import CaptureMetadata
from property_scanner.schemas.result import ProcessingInfo, PropertyScanResult
from property_scanner.schemas.serialization import save_result


def room(room_id: str, points) -> Room:
    vertices = [Point2D(x=float(x), y=float(y)) for x, y in points]
    polygon = Polygon2D(points=vertices)
    walls = []
    for index, start in enumerate(vertices):
        end = vertices[(index+1) % len(vertices)]
        walls.append(Wall(wall_id=f"{room_id}:wall:{index}", room_id=room_id,
            start_point=start, end_point=end,
            length=LengthMeasurement(value=float(np.hypot(end.x-start.x, end.y-start.y)))))
    area = 0.5*abs(sum(a.x*b.y-b.x*a.y for a, b in zip(vertices, vertices[1:]+vertices[:1])))
    return Room(room_id=room_id, name=room_id.title(), polygon=polygon, walls=walls,
        floor=FloorSurface(surface_id=f"{room_id}:floor", polygon=polygon,
            area=AreaMeasurement(value=area)),
        ceiling=CeilingSurface(surface_id=f"{room_id}:ceiling", polygon=polygon,
            height=LengthMeasurement(value=2.8)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sample_stitching"))
    args = parser.parse_args()
    rooms = [room("living", ((0, 0), (5, 0), (5, 4), (0, 4))),
             room("hall", ((0, 0), (2, 0), (2, 4), (0, 4))),
             room("kitchen", ((0, 0), (4, 0), (4, 3), (0, 3)))]
    specs = [("living", "hall", se2(5, 0, 0)), ("hall", "kitchen", se2(2, 0, 0))]
    evidence = [RoomConnectionEvidence(connection_id=f"edge:{a}:{b}", room_a_id=a, room_b_id=b,
        relative_transform=t.tolist(), raw_match_count=100, inlier_match_count=80, inlier_ratio=.8,
        visual_score=.9, geometry_score=.9, combined_score=.9, accepted=True) for a, b, t in specs]
    result = MultiRoomStitcher().stitch("synthetic-property", rooms, evidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_stitching_diagnostics(args.output_dir / "stitching", rooms, evidence, result)
    drawing = FloorPlanRenderer().render_property(result.property, title="Synthetic stitched property")
    try:
        drawing.save(args.output_dir)
    finally:
        drawing.close()
    (args.output_dir / "property_geometry.json").write_text(result.property.model_dump_json(indent=2)+"\n")
    scan = PropertyScanResult(capture=CaptureMetadata(
        capture_id=UUID("00000000-0000-4000-8000-000000000008"), tier="photo",
        source_type="directory", source_reference="synthetic_stitching_fixture"),
        property=result.property, warnings=result.warnings,
        processing_info=ProcessingInfo(pipeline_version="synthetic-stitching-1.0",
            modules_used=["multi_room_stitching", "floorplan_renderer"],
            metadata={"stitching": result.diagnostics.model_dump(mode="json")}),
        metadata={"synthetic": True, "benchmark_accuracy": False})
    save_result(scan, args.output_dir / "result.json")
    print(f"Rooms positioned: {result.diagnostics.rooms_positioned}")
    print(f"Total floor area: {result.property.total_floor_area.value:.2f} m²")
    print(f"PNG: {args.output_dir / 'floorplan.png'}")
    print(f"SVG: {args.output_dir / 'floorplan.svg'}")
    print(f"JSON: {args.output_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
