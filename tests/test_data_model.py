"""Contract validation, graph consistency, interoperability, and honest integration."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

from jsonschema import Draft202012Validator
import pytest
from pydantic import ValidationError

from config.settings import Settings
from property_scanner.inputs import select_adapter
from property_scanner.pipeline.context import ScanContext
from property_scanner.pipeline.processor import PropertyScanPipeline
from property_scanner.schemas import PropertyScanResult
from property_scanner.schemas.capture import CaptureMetadata
from property_scanner.schemas.damage import ConcealedDamageFlag, DamageRegion, ScopeLineItem
from property_scanner.schemas.geometry import CeilingSurface, Opening, PropertyGeometry, RoomConnection, Wall
from property_scanner.schemas.measurements import AreaMeasurement, LengthMeasurement, MeasuredValue
from property_scanner.schemas.primitives import BoundingBox2D, Point3D, Polygon2D
from property_scanner.schemas.result import ProcessingInfo, ResultWarning
from property_scanner.schemas.serialization import (
    export_json_schema, load_result, result_to_json, save_result, validate_result_json,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_property_result.json"


@pytest.fixture
def sample() -> PropertyScanResult:
    return load_result(FIXTURE)


def test_complete_synthetic_fixture(sample: PropertyScanResult) -> None:
    assert sample.metadata["synthetic"] is True
    assert len(sample.property.rooms) == 2
    assert all(len(room.walls) == 4 for room in sample.property.rooms)
    assert sample.property.rooms[0].ceiling.height.value == 2.7
    assert sample.property.rooms[0].floor.area.value == 12
    assert len(sample.property.openings) == 1
    assert sample.property.openings[0].room_ids == ["room_01", "room_02"]
    assert len(sample.damages) == len(sample.scope_line_items) == 1
    assert sample.processing_info is None


@pytest.mark.parametrize("fields", [
    {"lower_bound": 5}, {"upper_bound": 3},
    {"confidence_level": -0.1}, {"confidence_level": 1.1},
    {"confidence_score": -0.1}, {"confidence_score": 1.1},
    {"value": -1}, {"value": float("nan")}, {"value": float("inf")},
    {"value": True}, {"lower_bound": -1}, {"upper_bound": float("inf")},
])
def test_reject_invalid_measurements(fields: dict) -> None:
    with pytest.raises(ValidationError):
        MeasuredValue.model_validate({"value": 4, "unit": "m", **fields})


def test_bounds_are_inclusive_and_not_fabricated() -> None:
    value = LengthMeasurement(value=4, lower_bound=4, upper_bound=4, confidence_level=0, confidence_score=1)
    assert value.value == 4
    unknown_uncertainty = LengthMeasurement(value=4)
    assert unknown_uncertainty.lower_bound is None
    assert unknown_uncertainty.confidence_level is None
    assert unknown_uncertainty.method is None


@pytest.mark.parametrize("model,payload", [
    (Wall, {"wall_id": "w", "room_id": "r", "length": {"value": -1}}),
    (Opening, {"opening_id": "o", "width": {"value": -1}}),
    (CeilingSurface, {"surface_id": "c", "height": {"value": -1}}),
    (AreaMeasurement, {"value": -1}),
    (Wall, {"wall_id": "w", "room_id": "r", "length": {"value": 2, "unit": "m2"}}),
    (LengthMeasurement, {"value": 2, "unit": "ft"}),
    (AreaMeasurement, {"value": 2, "unit": "m"}),
    (ScopeLineItem, {"line_item_id": "s", "action": "repair", "description": "test", "quantity": {"value": 2, "unit": "degree"}}),
])
def test_physical_field_constraints(model: type, payload: dict) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("points", [[], [{"x": 0, "y": 0}], [{"x": 0, "y": 0}, {"x": 1, "y": 1}], [{"x": 0, "y": 0}] * 3])
def test_polygons_need_three_distinct_points(points: list) -> None:
    with pytest.raises(ValidationError):
        Polygon2D(points=points)


def test_geometry_numbers_and_box_bounds() -> None:
    with pytest.raises(ValidationError):
        Point3D(x=0, y=float("nan"), z=1)
    with pytest.raises(ValidationError):
        BoundingBox2D(min_point={"x": 2, "y": 0}, max_point={"x": 1, "y": 1})


def test_connection_rejects_self_loop() -> None:
    with pytest.raises(ValidationError, match="itself"):
        RoomConnection(connection_id="c", room_a_id="r", room_b_id="r")


@pytest.mark.parametrize("field,value", [("rule_id", ""), ("rule_id", "  "), ("rule_description", "  ")])
def test_concealed_flags_require_rule_and_explanation(field: str, value: str) -> None:
    fields = {"flag_id": "f", "rule_id": "rule", "rule_description": "Reason"}
    fields[field] = value
    with pytest.raises(ValidationError):
        ConcealedDamageFlag.model_validate(fields)


@pytest.mark.parametrize("missing", ["rule_id", "rule_description"])
def test_concealed_rule_fields_cannot_be_omitted(missing: str) -> None:
    fields = {"flag_id": "f", "rule_id": "rule", "rule_description": "Reason"}
    del fields[missing]
    with pytest.raises(ValidationError):
        ConcealedDamageFlag.model_validate(fields)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_entity_confidence_range(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Opening(opening_id="o", confidence=confidence)


@pytest.mark.parametrize("tier", ["photo", "video", "lidar"])
def test_partial_result_and_all_capture_tiers(tier: str) -> None:
    result = PropertyScanResult(
        capture=CaptureMetadata(capture_id="00000000-0000-4000-8000-000000000001", tier=tier),
        property=PropertyGeometry(property_id="p", rooms=[{
            "room_id": "r", "room_type": "custom_workshop",
            "ceiling": {"surface_id": "c", "height": None},
        }]),
    )
    assert result.property.rooms[0].ceiling.height is None
    assert result.property.total_floor_area is None
    assert result.capture.capture_timestamp is None
    assert result.processing_info is None
    assert '"height": null' in result_to_json(result)
    assert validate_result_json(result_to_json(result)) == result


def test_damage_can_have_area_length_or_neither() -> None:
    assert DamageRegion(damage_id="d", damage_type="custom_damage").metric_area is None
    assert DamageRegion(damage_id="d", metric_length={"value": 1}).metric_area is None
    assert DamageRegion(damage_id="d", metric_area={"value": 1}).metric_length is None


@pytest.mark.parametrize("collection", ["damages", "concealed_damage_flags", "scope_line_items"])
def test_duplicate_root_ids(sample: PropertyScanResult, collection: str) -> None:
    payload = sample.model_dump(mode="json")
    payload[collection].append(deepcopy(payload[collection][0]))
    with pytest.raises(ValidationError, match="Duplicate"):
        PropertyScanResult.model_validate(payload)


@pytest.mark.parametrize("collection", ["rooms", "openings", "room_connections"])
def test_duplicate_property_ids(sample: PropertyScanResult, collection: str) -> None:
    payload = sample.model_dump(mode="json")
    payload["property"][collection].append(deepcopy(payload["property"][collection][0]))
    with pytest.raises(ValidationError, match="Duplicate"):
        PropertyScanResult.model_validate(payload)


def test_duplicate_surface_ids_across_rooms(sample: PropertyScanResult) -> None:
    payload = sample.model_dump(mode="json")
    payload["property"]["rooms"][1]["floor"]["surface_id"] = "wall_1_1"
    with pytest.raises(ValidationError, match="Duplicate surface"):
        PropertyScanResult.model_validate(payload)


@pytest.mark.parametrize("path,value", [
    (("damages", 0, "room_id"), "missing"),
    (("damages", 0, "surface_id"), "missing"),
    (("damages", 0, "room_id"), "room_02"),
    (("scope_line_items", 0, "damage_id"), "missing"),
    (("scope_line_items", 0, "surface_id"), "wall_1_2"),
    (("property", "openings", 0, "room_ids"), ["missing"]),
    (("property", "openings", 0, "wall_id"), "missing"),
    (("property", "room_connections", 0, "opening_id"), "missing"),
    (("property", "room_connections", 0, "room_b_id"), "missing"),
    (("property", "rooms", 0, "opening_ids"), ["missing"]),
    (("property", "rooms", 0, "damage_ids"), ["missing"]),
    (("property", "rooms", 0, "walls", 0, "room_id"), "room_02"),
])
def test_invalid_references(sample: PropertyScanResult, path: tuple, value: object) -> None:
    payload = sample.model_dump(mode="json")
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        PropertyScanResult.model_validate(payload)


@pytest.mark.parametrize("identifier", ["", "  ", "bad id", "../wall"])
def test_reference_structure(identifier: str) -> None:
    with pytest.raises(ValidationError):
        ResultWarning(code="WARNING", message="Details", room_id=identifier)


def test_processing_timestamps() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        ProcessingInfo(started_at="2026-01-02T00:00:00Z", completed_at="2026-01-01T00:00:00Z")
    with pytest.raises(ValidationError):
        ProcessingInfo(started_at="2026-01-01T00:00:00")
    assert ProcessingInfo().processing_seconds is None


def test_unknown_version_and_extra_fields(sample: PropertyScanResult) -> None:
    payload = sample.model_dump(mode="json")
    payload["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError):
        PropertyScanResult.model_validate(payload)
    with pytest.raises(ValidationError):
        LengthMeasurement(value=1, centimetres=100)


def test_json_round_trip_and_determinism(sample: PropertyScanResult, tmp_path: Path) -> None:
    sample.metadata["label"] = "Synthetic café"
    sample.capture.capture_timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pretty = result_to_json(sample)
    assert result_to_json(sample) == pretty
    assert "café" in pretty
    assert validate_result_json(pretty) == sample
    assert validate_result_json(result_to_json(sample, pretty=False)) == sample
    path = save_result(sample, tmp_path / "nested" / "result.json")
    assert load_result(path) == sample
    assert path.read_text(encoding="utf-8") == pretty


def test_save_revalidates_mutations_before_writing(sample: PropertyScanResult, tmp_path: Path) -> None:
    sample.property.rooms[0].walls[0].length.value = -1
    with pytest.raises(ValidationError):
        save_result(sample, tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_result_json("not JSON")


def test_official_json_schema(sample: PropertyScanResult, tmp_path: Path) -> None:
    path = export_json_schema(tmp_path / "schema.json")
    schema = json.loads(path.read_text())
    assert schema == PropertyScanResult.model_json_schema(mode="validation")
    assert schema == json.loads((ROOT / "schemas" / "property_scan_result.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(json.loads(FIXTURE.read_text()))
    validator.validate(json.loads(result_to_json(sample)))
    invalid = sample.model_dump(mode="json")
    invalid["property"]["openings"][0]["width"]["value"] = -1
    assert list(validator.iter_errors(invalid))


def test_schema_export_script(tmp_path: Path) -> None:
    target = tmp_path / "exported.json"
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_schema.py"), "--output", str(target)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(target.read_text())["title"] == "PropertyScanResult"


@pytest.mark.parametrize("tier", ["photo", "video", "lidar"])
def test_shared_pipeline_still_does_not_process(tmp_path: Path, tier: str) -> None:
    from tests.test_project_startup import make_source

    source = make_source(tmp_path, tier)
    settings = Settings(_env_file=None, output_dir=tmp_path / "outputs")
    pipeline = PropertyScanPipeline(settings)
    prepared = pipeline.prepare(select_adapter(tier, source))
    assert get_type_hints(pipeline.process)["return"] is PropertyScanResult
    assert get_type_hints(ScanContext)["result"] == PropertyScanResult | None
    with pytest.raises(NotImplementedError, match="Scene reconstruction"):
        pipeline.process(prepared)
    assert list(prepared.output_dir.iterdir()) == []


def test_unimplemented_assembly_never_returns_fake_result(tmp_path: Path) -> None:
    from tests.test_project_startup import make_source

    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path / "outputs"))
    prepared = pipeline.prepare(select_adapter("video", make_source(tmp_path, "video")))
    pipeline.stages = ()
    with pytest.raises(NotImplementedError, match="result assembly"):
        pipeline.process(prepared)
