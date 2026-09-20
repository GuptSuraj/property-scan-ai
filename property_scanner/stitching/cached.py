"""Rerun stitching from Prompt 7 artifacts without COLMAP or depth inference."""

import json
from pathlib import Path

from property_scanner.reconstruction.lidar.diagnostics import write_json
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.schemas.geometry import PropertyGeometry, Room
from property_scanner.schemas.measurements import AreaMeasurement
from property_scanner.schemas.serialization import load_result, save_result
from property_scanner.stitching.diagnostics import export_stitching_diagnostics, room_scale_quality
from property_scanner.schemas.result import ResultWarning
from property_scanner.stitching.engine import MultiRoomStitcher, StitchingResult
from property_scanner.stitching.evidence import collect_cached_photo_evidence
from property_scanner.stitching.models import RoomConnectionEvidence, StitchingConfig


def load_local_rooms(output_dir: Path) -> list[Room]:
    rooms = []
    for path in sorted((Path(output_dir) / "photo").glob("*/room.json")):
        rooms.append(Room.model_validate_json(path.read_text()))
    if not rooms:
        raise ValueError("No cached photo/<room_id>/room.json artifacts were found")
    return rooms


def stitch_cached_output(output_dir: Path, config: StitchingConfig | None = None,
                         evidence: list[RoomConnectionEvidence] | None = None) -> StitchingResult:
    output_dir, config = Path(output_dir), config or StitchingConfig()
    scan = load_result(output_dir / "result.json")
    rooms = load_local_rooms(output_dir)
    candidates = evidence if evidence is not None else collect_cached_photo_evidence(
        output_dir, [room.room_id for room in rooms], config)
    stitched = MultiRoomStitcher(config).stitch(scan.property.property_id, rooms, candidates)
    export_stitching_diagnostics(output_dir / "stitching", rooms, candidates, stitched)
    scale_report, inconsistent = room_scale_quality(output_dir, [room.room_id for room in rooms], config)
    write_json(output_dir / "stitching/room_scale_consistency.json", scale_report)
    if inconsistent:
        stitched.warnings.append(ResultWarning(code="ROOM_SCALE_INCONSISTENCY",
            message=f"Weak independent metric-scale consistency for: {', '.join(inconsistent)}."))
    if not any(item.relative_transform is not None for item in candidates):
        stitched.warnings.append(ResultWarning(code="NO_CROSS_ROOM_MATCHES",
            message="No geometrically verified metric cross-room correspondence was found."))
    stitching_codes = {"NO_CROSS_ROOM_MATCHES", "WEAK_ROOM_CONNECTION", "AMBIGUOUS_ROOM_CONNECTION",
        "ROOM_REGISTRATION_FAILED", "DISCONNECTED_ROOM_GRAPH", "PROPERTY_LAYOUT_CONFLICT",
        "EXCESSIVE_ROOM_OVERLAP", "HIGH_CYCLE_CLOSURE_ERROR", "ROOM_SCALE_INCONSISTENCY",
        "PROPERTY_FOOTPRINT_DISCONNECTED"}
    scan.warnings = [warning for warning in scan.warnings if warning.code not in stitching_codes]
    scan.warnings.extend(stitched.warnings)
    if "multi_room_stitching" not in scan.processing_info.modules_used:
        scan.processing_info.modules_used.append("multi_room_stitching")
    scan.processing_info.metadata["stitching"] = stitched.diagnostics.model_dump(mode="json")
    scan.metadata.update({"room_coordinates_are_local": not stitched.diagnostics.valid_layout,
                          "multi_room_stitching_implemented": True})
    if stitched.diagnostics.valid_layout:
        scan.property = stitched.property
        drawing = FloorPlanRenderer().render_property(scan.property, title="Property floor plan")
        try:
            drawing.save(output_dir)
            scan.warnings.extend(drawing.warnings)
        finally:
            drawing.close()
    else:
        areas = [room.floor.area.value for room in rooms if room.floor and room.floor.area]
        scan.property = PropertyGeometry(property_id=scan.property.property_id, rooms=rooms,
            total_floor_area=AreaMeasurement(value=sum(areas),
                method="sum of independent unstitched room polygons") if areas else None,
            metadata={"coordinate_scope": "room_local_unstitched", "stitching_performed": True,
                      "layout_valid": False})
        for stale in (output_dir / "floorplan.png", output_dir / "floorplan.svg"):
            stale.unlink(missing_ok=True)
    save_result(scan, output_dir / "result.json")
    return stitched
