"""Shared downstream finalization for every reconstructed input tier."""
import csv
import json
from pathlib import Path
from property_scanner.confidence.estimator import ConfidenceConfig, ConfidenceEstimator
from property_scanner.damage.cached import try_process_cached_damage
from property_scanner.damage.models import DamageConfig
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.schemas.serialization import save_result
from property_scanner.schemas.result import ResultWarning


def _measurements(result):
    tier = result.capture.tier.value
    if result.property.total_floor_area:
        yield "property:total_floor_area", tier, "", "", "total_floor_area", result.property.total_floor_area
    if result.property.bounding_dimensions:
        for kind, value in (("property_length", result.property.bounding_dimensions.length),
                            ("property_width", result.property.bounding_dimensions.width),
                            ("property_height", result.property.bounding_dimensions.height)):
            if value: yield f"property:{kind}", tier, "", "", kind, value
    for room in result.property.rooms:
        for wall in room.walls:
            for kind, value in (("wall_length", wall.length), ("wall_height", wall.height),
                                ("wall_thickness", wall.thickness), ("wall_orientation", wall.orientation)):
                if value: yield f"{wall.wall_id}:{kind}", tier, room.room_id, wall.wall_id, kind, value
        if room.floor and room.floor.area: yield room.floor.surface_id, tier, room.room_id, room.floor.surface_id, "floor_area", room.floor.area
        if room.ceiling:
            if room.ceiling.height:
                yield f"{room.ceiling.surface_id}:height", tier, room.room_id, room.ceiling.surface_id, "ceiling_height", room.ceiling.height
            if room.ceiling.area:
                yield f"{room.ceiling.surface_id}:area", tier, room.room_id, room.ceiling.surface_id, "ceiling_area", room.ceiling.area
    for opening in result.property.openings:
        for kind, value in (("opening_width", opening.width), ("opening_height", opening.height),
                            ("sill_height", opening.sill_height), ("opening_position", opening.position_along_wall)):
            if value: yield f"{opening.opening_id}:{kind}", tier, opening.room_ids[0] if opening.room_ids else "", opening.wall_id or "", kind, value
    for damage in result.damages:
        for kind, value in (("damage_area", damage.metric_area), ("damage_length", damage.metric_length)):
            if value: yield f"{damage.damage_id}:{kind}", tier, damage.room_id or "", damage.surface_id or "", kind, value
    for item in result.scope_line_items:
        if item.quantity:
            yield f"{item.line_item_id}:quantity", tier, item.room_id or "", item.surface_id or "", "scope_quantity", item.quantity


def write_measurements_csv(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["measurement_id", "tier", "room_id", "surface_id", "measurement_type", "value", "unit",
              "lower_bound", "upper_bound", "confidence_level", "confidence_method"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for identifier, tier, room, surface, kind, value in _measurements(result):
            writer.writerow(dict(measurement_id=identifier, tier=tier, room_id=room, surface_id=surface,
                measurement_type=kind, value=value.value, unit=value.unit.value, lower_bound="" if value.lower_bound is None else value.lower_bound,
                upper_bound="" if value.upper_bound is None else value.upper_bound, confidence_level="" if value.confidence_level is None else value.confidence_level,
                confidence_method=value.confidence_method.value))


def finalize_result(output: Path, result, model_dir: Path, *, skip_damage=False,
                    damage_config: DamageConfig | None=None, confidence_config: ConfidenceConfig | None=None):
    output = Path(output)
    if not skip_damage: try_process_cached_damage(output, result, model_dir, damage_config)
    confidence_config = confidence_config or ConfidenceConfig(allow_tier_prior_uncalibrated=True)
    ConfidenceEstimator(confidence_config).apply(result)
    methods = result.processing_info.metadata.get("confidence", {}).get("methods", []) if result.processing_info else []
    if "tier_prior_uncalibrated" in methods and not any(w.code == "UNCALIBRATED_CONFIDENCE_INTERVALS" for w in result.warnings):
        result.warnings.append(ResultWarning(code="UNCALIBRATED_CONFIDENCE_INTERVALS",
            message="Some intervals use conservative tier priors because benchmark calibration or direct dispersion evidence is unavailable."))
    write_measurements_csv(output/"measurements.csv", result)
    renderable = result.property.rooms and all(room.polygon and room.walls for room in result.property.rooms)
    shared = len(result.property.rooms) == 1 or result.property.metadata.get("coordinate_scope") == "property_shared"
    if renderable and shared:
        drawing = FloorPlanRenderer().render_property(result.property, damages=result.damages)
        try: drawing.save(output)
        finally: drawing.close()
    artifact = output/"artifacts"/result.capture.tier.value; artifact.mkdir(parents=True, exist_ok=True)
    (artifact/"manifest.json").write_text(json.dumps({"tier_artifacts": f"../../{result.capture.tier.value}",
        "note": "Existing tier artifact paths are retained for backward compatibility."}, indent=2)+"\n")
    save_result(result, output/"result.json")
    return result
