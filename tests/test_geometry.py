"""Tests exercise real geometry on explicitly synthetic, seeded point clouds."""
from pathlib import Path
import subprocess
import sys
import pytest
o3d = pytest.importorskip("open3d", reason="Install .[geometry] to run point-cloud tests")
import numpy as np
from shapely.geometry import Polygon
from property_scanner.geometry.config import GeometryConfig
from property_scanner.geometry.engine import GeometryEngine
from property_scanner.geometry.exceptions import PointCloudLoadError, InsufficientGeometryError, FloorNotDetectedError
from property_scanner.geometry.preprocessing import load_cloud, preprocess, axis_rotation
from property_scanner.geometry.planes import fit_plane
from property_scanner.geometry.walls import merge_walls
from property_scanner.geometry.models import RoomGeometryResult
from tests.synthetic_geometry import room_points, write_cloud


@pytest.fixture(scope="module")
def rectangle(tmp_path_factory):
    root = tmp_path_factory.mktemp("geometry")
    path = write_cloud(root / "room.ply", room_points())
    diagnostics = root / "diagnostics"
    result = GeometryEngine().process_point_cloud(path, diagnostics_dir=diagnostics)
    return result, diagnostics


def test_rectangle_measurements(rectangle):
    result, _ = rectangle
    assert len(result.wall_planes) == 4
    assert len(result.corners) == 4
    assert result.floor_area.value == pytest.approx(20, abs=0.2)
    assert result.ceiling_height.value == pytest.approx(2.8, abs=0.03)
    assert sorted(s.length.value for s in result.wall_segments_2d) == pytest.approx([4, 4, 5, 5], abs=0.05)
    polygon = Polygon([(p.x, p.y) for p in result.room_polygon.points])
    assert polygon.is_valid and polygon.exterior.is_ccw
    assert result.diagnostics.explained_point_ratio > 0.95
    assert result.diagnostics.polygon_closure_error < 0.15
    assert result.ceiling_height.lower_bound is None
    assert result.ceiling_height.confidence_score is None
    assert result.to_room("room_01").ceiling.height == result.ceiling_height
    assert RoomGeometryResult.model_validate_json(result.model_dump_json()) == result


def test_debug_exports(rectangle):
    result, directory = rectangle
    assert (directory / "geometry.png").read_bytes().startswith(b"\x89PNG")
    for name in ("cleaned", "floor_inliers", "ceiling_inliers", "wall_01_inliers"):
        cloud = o3d.io.read_point_cloud(str(directory / f"{name}.ply"))
        assert len(cloud.points) > 0
    assert RoomGeometryResult.model_validate_json((directory / "geometry.json").read_text()) == result


@pytest.mark.parametrize("extension", ["ply", "pcd"])
def test_loading_and_preprocessing(tmp_path, extension):
    points = room_points()
    path = write_cloud(tmp_path / f"room.{extension}", np.concatenate([points, points]))
    cloud = load_cloud(path, GeometryConfig())
    cleaned, original, downsampled = preprocess(cloud, GeometryConfig())
    assert original == len(points)*2
    assert len(cleaned.points) <= downsampled < original
    assert cleaned.has_normals()
    assert np.allclose(np.linalg.norm(np.asarray(cleaned.normals), axis=1), 1)


@pytest.mark.parametrize("options,expected_walls,area", [
    ({"outliers": 700}, 4, 20), ({"ceiling": False}, 4, 20), ({"shape": "l"}, 6, 19),
])
def test_room_variations(tmp_path, options, expected_walls, area):
    path = write_cloud(tmp_path / "room.pcd", room_points(**options))
    result = GeometryEngine().process_point_cloud(path)
    assert len(result.wall_planes) == expected_walls
    assert result.floor_area is not None
    assert result.floor_area.value == pytest.approx(area, abs=0.3)
    if options.get("ceiling") is False:
        assert result.ceiling_plane is None and result.ceiling_height is None
        assert "CEILING_UNAVAILABLE" in [w.code for w in result.warnings]
    else:
        assert result.ceiling_height.value == pytest.approx(2.8, abs=0.04)
    if options.get("shape") == "l":
        assert len(result.room_polygon.points) == 6


@pytest.mark.parametrize("axis", ["x", "y", "-z"])
def test_configured_up_axis(tmp_path, axis):
    source = room_points() @ axis_rotation(axis)
    result = GeometryEngine(GeometryConfig(up_axis=axis)).process_point_cloud(write_cloud(tmp_path / "room.ply", source))
    assert result.floor_area.value == pytest.approx(20, abs=0.2)
    assert result.ceiling_height.value == pytest.approx(2.8, abs=0.04)


def test_missing_invalid_and_insufficient_clouds(tmp_path):
    engine = GeometryEngine()
    with pytest.raises(PointCloudLoadError):
        engine.process_point_cloud(tmp_path / "missing.ply")
    path = tmp_path / "bad.xyz"
    path.write_text("not a point cloud")
    with pytest.raises(PointCloudLoadError):
        engine.process_point_cloud(path)
    path = tmp_path / "bad.ply"
    path.write_text("invalid")
    with pytest.raises(PointCloudLoadError):
        engine.process_point_cloud(path)
    with pytest.raises(InsufficientGeometryError):
        engine.process_point_cloud(write_cloud(tmp_path / "tiny.ply", np.zeros((10, 3))))
    rng = np.random.default_rng(0)
    with pytest.raises(FloorNotDetectedError):
        engine.process_point_cloud(write_cloud(tmp_path / "random.ply", rng.uniform(0, 5, (300, 3))))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_rejected(tmp_path, value):
    points = room_points()
    points[0, 0] = value
    with pytest.raises(PointCloudLoadError, match="NaN or infinite"):
        GeometryEngine().process_point_cloud(write_cloud(tmp_path / "bad.ply", points))


def test_duplicate_wall_merging():
    rng = np.random.default_rng(1)
    points = np.column_stack([rng.normal(0, 0.003, 1000), rng.uniform(0, 4, 1000), rng.uniform(0, 2.8, 1000)])
    candidates = [fit_plane(points, np.arange(0, 500)), fit_plane(points, np.arange(500, 1000))]
    merged, count = merge_walls(candidates, points, GeometryConfig())
    assert count == 1 and len(merged) == 1
    assert len(merged[0].indices) == 1000
    points[500:, 0] += 0.3
    separate = [fit_plane(points, np.arange(0, 500)), fit_plane(points, np.arange(500, 1000))]
    assert len(merge_walls(separate, points, GeometryConfig())[0]) == 2


def test_missing_wall_preserves_evidence(tmp_path):
    points = room_points()
    # Remove one entire wall while keeping floor and ceiling observations.
    keep = ~((abs(points[:, 0]) < 0.025) & (points[:, 2] > 0.04) & (points[:, 2] < 2.76))
    result = GeometryEngine().process_point_cloud(write_cloud(tmp_path / "open.ply", points[keep]))
    assert len(result.wall_planes) == 3
    assert result.room_polygon is None and result.floor_area is None
    assert "POLYGON_UNAVAILABLE" in [w.code for w in result.warnings]


def test_tilted_floor_uses_plane_distance(tmp_path):
    points = room_points()
    angle = np.deg2rad(7)
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    points = points @ rotation.T + np.array([3, -2, 5])
    assert np.ptp(points[:, 2]) > 3.2
    result = GeometryEngine().process_point_cloud(write_cloud(tmp_path / "tilted.ply", points))
    assert result.ceiling_height.value == pytest.approx(2.8, abs=0.04)
    assert result.floor_area.value == pytest.approx(20, abs=0.2)
    assert abs(result.floor_plane.centroid.z) < 0.01


def test_furniture_is_not_a_ceiling(tmp_path):
    points = room_points(ceiling=False)
    rng = np.random.default_rng(55)
    xy = rng.uniform([0.4, 0.4], [3.6, 4.6], (1200, 2))
    table = np.column_stack([xy, rng.normal(0.9, 0.003, len(xy))])
    shelf = np.column_stack([xy, rng.normal(2.2, 0.003, len(xy))])
    result = GeometryEngine().process_point_cloud(write_cloud(tmp_path / "furniture.ply", np.concatenate([points, table, shelf])))
    assert result.ceiling_height is None
    assert result.floor_area.value == pytest.approx(20, abs=0.3)


def test_repeatability(tmp_path):
    path = write_cloud(tmp_path / "room.ply", room_points())
    engine = GeometryEngine()
    first, second = engine.process_point_cloud(path), engine.process_point_cloud(path)
    assert first.floor_area.value == pytest.approx(second.floor_area.value, abs=0.01)
    assert first.ceiling_height.value == pytest.approx(second.ceiling_height.value, abs=0.01)


def test_disjoint_coplanar_walls_are_not_merged():
    rng = np.random.default_rng(72)
    first = np.column_stack([np.zeros(500), rng.uniform(0, 2, 500), rng.uniform(0, 2.8, 500)])
    second = np.column_stack([np.zeros(500), rng.uniform(3, 5, 500), rng.uniform(0, 2.8, 500)])
    points = np.concatenate([first, second])
    planes = [fit_plane(points, np.arange(500)), fit_plane(points, np.arange(500, 1000))]
    assert len(merge_walls(planes, points, GeometryConfig())[0]) == 2


def test_geometry_cli(tmp_path):
    path = write_cloud(tmp_path / "room.ply", room_points())
    script = Path(__file__).resolve().parents[1] / "scripts" / "test_geometry.py"
    completed = subprocess.run([sys.executable, str(script), "--input", str(path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "Walls detected: 4" in completed.stdout
    failed = subprocess.run([sys.executable, str(script), "--input", str(tmp_path / "missing.ply")], capture_output=True, text=True)
    assert failed.returncode == 2
    assert "Traceback" not in failed.stderr
