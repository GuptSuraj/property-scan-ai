"""Deterministic graph, rigid-transform, overlap and rendering tests."""

from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import Polygon

from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.schemas.geometry import CeilingSurface, FloorSurface, Room, Wall
from property_scanner.schemas.measurements import AreaMeasurement, LengthMeasurement
from property_scanner.schemas.primitives import Point2D, Polygon2D
from property_scanner.stitching.engine import MultiRoomStitcher
from property_scanner.stitching.models import RoomConnectionEvidence, StitchingConfig
from property_scanner.stitching.transforms import se2
from property_scanner.stitching.evidence import candidate_room_pairs, robust_se2
from property_scanner.stitching.evidence import collect_cached_photo_evidence
import json
import cv2


def room(room_id: str, points=((0, 0), (4, 0), (4, 3), (0, 3))) -> Room:
    polygon = Polygon(points)
    vertices = [Point2D(x=float(x), y=float(y)) for x, y in points]
    walls = []
    for index, start in enumerate(vertices):
        end = vertices[(index+1) % len(vertices)]
        length = float(np.hypot(end.x-start.x, end.y-start.y))
        walls.append(Wall(wall_id=f"{room_id}:wall:{index}", room_id=room_id,
            start_point=start, end_point=end, length=LengthMeasurement(value=length)))
    shape = Polygon2D(points=vertices)
    return Room(room_id=room_id, name=room_id, polygon=shape, walls=walls,
        floor=FloorSurface(surface_id=f"{room_id}:floor", polygon=shape,
                           area=AreaMeasurement(value=float(polygon.area))),
        ceiling=CeilingSurface(surface_id=f"{room_id}:ceiling", polygon=shape,
                               height=LengthMeasurement(value=2.8)))


def edge(a: str, b: str, transform: np.ndarray, quality=0.9, suffix="") -> RoomConnectionEvidence:
    return RoomConnectionEvidence(connection_id=f"edge:{a}:{b}{suffix}", room_a_id=a, room_b_id=b,
        relative_transform=transform.tolist(), raw_match_count=80, inlier_match_count=60,
        inlier_ratio=.75, visual_score=.8, geometry_score=.9, combined_score=quality, accepted=True)


def relative(result, a, b):
    ta, tb = result.transforms[a], result.transforms[b]
    return np.linalg.inv(se2(ta.translation_x, ta.translation_y, ta.rotation_yaw)) @ \
        se2(tb.translation_x, tb.translation_y, tb.rotation_yaw)


def test_three_room_linear_layout_preserves_measurements(tmp_path: Path):
    rooms = [room(name) for name in "ABC"]
    before = {item.room_id: (item.floor.area.value, [wall.length.value for wall in item.walls]) for item in rooms}
    result = MultiRoomStitcher().stitch("property", rooms,
        [edge("A", "B", se2(4, 0, 0)), edge("B", "C", se2(4, 0, 0))])
    assert result.diagnostics.valid_layout
    assert result.diagnostics.rooms_positioned == 3
    assert relative(result, "A", "B") == pytest.approx(se2(4, 0, 0), abs=1e-6)
    for placed in result.property.rooms:
        assert (placed.floor.area.value, [wall.length.value for wall in placed.walls]) == before[placed.room_id]
    assert result.property.total_floor_area.value == pytest.approx(36)
    assert result.property.footprint_polygon is not None
    drawing = FloorPlanRenderer().render_property(result.property)
    try:
        files = drawing.save(tmp_path)
        assert files.png.stat().st_size and files.svg.stat().st_size
    finally:
        drawing.close()


def test_four_room_loop_global_optimization_and_cycle_consistency():
    rooms = [room(name, ((0, 0), (2, 0), (2, 2), (0, 2))) for name in "ABCD"]
    evidence = [edge("A", "B", se2(2, 0, 0)), edge("B", "C", se2(0, 2, 0)),
                edge("C", "D", se2(-2, 0, 0)), edge("D", "A", se2(0, -2, 0))]
    result = MultiRoomStitcher().stitch("p", rooms, evidence)
    assert result.diagnostics.valid_layout
    assert result.diagnostics.cycle_residual == pytest.approx(0, abs=1e-8)
    assert result.diagnostics.layout_constraint_residual == pytest.approx(0, abs=1e-8)


def test_low_confidence_conflicting_cycle_edge_is_rejected():
    rooms = [room(name) for name in "ABC"]
    evidence = [edge("A", "B", se2(4, 0, 0), .95), edge("B", "C", se2(4, 0, 0), .9),
                edge("A", "C", se2(2, 8, 0), .4, ":bad")]
    result = MultiRoomStitcher().stitch("p", rooms, evidence)
    assert any(item.connection_id.endswith(":bad") and item.rejection_reason == "cycle_inconsistent"
               for item in result.rejected_evidence)
    assert any(warning.code == "AMBIGUOUS_ROOM_CONNECTION" for warning in result.warnings)


def test_substantial_overlap_is_invalid_not_rescaled():
    rooms = [room("A"), room("B")]
    result = MultiRoomStitcher().stitch("p", rooms, [edge("A", "B", se2(1, 0, 0))])
    assert not result.diagnostics.valid_layout
    assert result.diagnostics.maximum_pair_overlap == pytest.approx(9)
    assert {warning.code for warning in result.warnings} >= {"EXCESSIVE_ROOM_OVERLAP", "PROPERTY_LAYOUT_CONFLICT"}
    assert all(placed.floor.area.value == 12 for placed in result.property.rooms)


def test_disconnected_room_has_no_fabricated_transform():
    rooms = [room(name) for name in "ABC"]
    result = MultiRoomStitcher().stitch("p", rooms, [edge("A", "B", se2(4, 0, 0))])
    assert result.diagnostics.rooms_positioned == 2
    assert "C" not in result.transforms
    assert result.property.metadata["unpositioned_room_ids"] == ["C"]
    assert any(warning.code == "DISCONNECTED_ROOM_GRAPH" for warning in result.warnings)


def test_irregular_room_shape_and_area_are_preserved():
    points = ((0, 0), (4, 0), (4, 1), (2, 1), (2, 3), (0, 3))
    rooms = [room("L", points), room("B", ((0, 0), (2, 0), (2, 2), (0, 2)))]
    result = MultiRoomStitcher().stitch("p", rooms, [edge("L", "B", se2(4, 0, 0))])
    placed = next(item for item in result.property.rooms if item.room_id == "L")
    assert len(placed.polygon.points) == 6
    assert Polygon([(p.x, p.y) for p in placed.polygon.points]).area == pytest.approx(8)


def test_transform_accuracy_with_yaw():
    rooms = [room("A"), room("B")]
    expected = se2(5.2, -1.4, np.deg2rad(18))
    # Permit overlap here so the test isolates transform estimation semantics.
    config = StitchingConfig(max_overlap_area_m2=100, max_overlap_ratio=1)
    result = MultiRoomStitcher(config).stitch("p", rooms, [edge("A", "B", expected)])
    assert relative(result, "A", "B") == pytest.approx(expected, abs=1e-6)


def test_candidate_generation_is_deterministic_and_bounded():
    assert candidate_room_pairs(["C", "A", "B"], 2) == [("A", "B"), ("A", "C")]


def test_metric_correspondence_ransac_rejects_outliers_without_scale():
    rng = np.random.default_rng(7)
    source = rng.uniform(-2, 2, (80, 2))
    truth = se2(3.2, -1.1, np.deg2rad(22))
    target = (truth[:2, :2] @ source.T).T + truth[:2, 2] + rng.normal(0, .01, source.shape)
    target[:18] = rng.uniform(-8, 8, (18, 2))
    fitted, inliers = robust_se2(source, target, StitchingConfig(cross_room_min_inliers=10), seed=4)
    assert inliers.sum() >= 60
    assert fitted == pytest.approx(truth, abs=.03)


def test_cached_cross_room_visual_metric_matching(tmp_path: Path):
    rng = np.random.default_rng(12)
    gray = rng.integers(0, 256, (192, 256), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    intrinsics = {"width": 256, "height": 192, "fx": 180.0, "fy": 180.0,
                  "cx": 127.5, "cy": 95.5}
    for room_id, x in (("A", 0.0), ("B", -4.0)):
        base = tmp_path / "photo" / room_id
        (base / "selected_images").mkdir(parents=True)
        (base / "depth").mkdir()
        (base / "sfm").mkdir()
        assert cv2.imwrite(str(base / "selected_images/000000.jpg"), image)
        np.save(base / "depth/000000.npy", np.full((192, 256), 2.0, dtype=np.float32))
        reconstruction = {"frames": [{"image_id": 1, "filename": "000000.jpg",
            "intrinsics": intrinsics, "camera_to_world": np.eye(4).tolist(), "observations": []}],
            "points": [], "statistics": {}}
        (base / "sfm/reconstruction.json").write_text(json.dumps(reconstruction))
        pose = np.eye(4); pose[0, 3] = x
        (base / "canonical_poses.json").write_text(json.dumps({"poses": [{"frame_id": 0,
            "matrix": pose.tolist()}]}))
    config = StitchingConfig(cross_room_min_matches=12, cross_room_min_inliers=8,
        connection_acceptance_threshold=.2, enable_point_cloud_refinement=False)
    evidence = collect_cached_photo_evidence(tmp_path, ["A", "B"], config)[0]
    assert evidence.accepted
    assert evidence.raw_match_count >= 12 and evidence.inlier_match_count >= 8
    assert np.asarray(evidence.relative_transform) == pytest.approx(se2(4, 0, 0), abs=.03)
