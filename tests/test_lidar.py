"""Canonical RGB-D and drift tests use synthetic data only, not benchmark claims."""
import json
from pathlib import Path
import subprocess
import sys
import shutil
import zipfile
import pytest
o3d = pytest.importorskip("open3d", reason="Install .[lidar] for RGB-D tests")
import numpy as np
from PIL import Image
from pydantic import ValidationError
from config.settings import Settings
from property_scanner.core.exceptions import InvalidInputError, ProcessingError
from property_scanner.inputs import select_adapter
from property_scanner.pipeline.processor import PropertyScanPipeline
from property_scanner.reconstruction.lidar.adapter import CanonicalRGBDAdapter, RGBDFrame
from property_scanner.reconstruction.lidar.models import CaptureManifest, CameraIntrinsics, LidarConfig
from property_scanner.reconstruction.lidar.poses import rigid_transform, normalize_pose
from property_scanner.reconstruction.lidar.fusion import Keyframe, frame_cloud, build_keyframes, fuse
from property_scanner.reconstruction.lidar.registration import register_pair, loop_candidates, optimize
from property_scanner.reconstruction.lidar.synthetic import generate_capture
from property_scanner.reconstruction.lidar.record3d import convert_record3d
from property_scanner.schemas.serialization import load_result


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    return generate_capture(tmp_path_factory.mktemp("lidar")/"capture")


@pytest.fixture
def mutable_capture(capture, tmp_path):
    target = tmp_path/"capture"
    shutil.copytree(capture, target)
    return target


@pytest.fixture(scope="module")
def keyframes(capture):
    adapter = CanonicalRGBDAdapter(capture)
    adapter.validate()
    return build_keyframes(adapter, LidarConfig())


@pytest.fixture(scope="module")
def on_run(capture, tmp_path_factory):
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path_factory.mktemp("on")))
    prepared = pipeline.prepare(select_adapter("lidar", capture))
    result = pipeline.process(prepared, lidar_config=LidarConfig())
    return prepared.output_dir, result


def edit(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_manifest_and_calibration(capture):
    adapter = CanonicalRGBDAdapter(capture)
    adapter.validate()
    assert adapter.manifest.frame_count == 25
    assert len(adapter.poses) == 25
    assert adapter.intrinsics.fx == 66
    assert adapter.manifest.registered_rgb_depth


def test_record3d_archive_converts_to_canonical_capture(tmp_path):
    liblzfse = pytest.importorskip("liblzfse")
    archive = tmp_path / "room.r3d"
    width, height = 32, 24
    k = np.array([[30.0, 0.0, 15.5], [0.0, 30.0, 11.5], [0.0, 0.0, 1.0]])
    metadata = {"w": width, "h": height, "dw": 16, "dh": 12,
                "K": k.flatten(order="F").tolist(), "poses": [[0, 0, 0, 1, 0, 0, 0]],
                "deviceName": "Synthetic iPhone"}
    image_buffer = __import__("io").BytesIO()
    Image.new("RGB", (width, height), (100, 120, 140)).save(image_buffer, format="JPEG")
    depth = np.full((12, 16), 2.25, dtype=np.float32)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("metadata", json.dumps(metadata))
        handle.writestr("rgbd/0.jpg", image_buffer.getvalue())
        handle.writestr("rgbd/0.depth", liblzfse.compress(depth.tobytes()))
    root = convert_record3d(archive, tmp_path / "canonical")
    adapter = CanonicalRGBDAdapter(root)
    adapter.validate()
    frames = list(adapter.load_frames(LidarConfig(min_valid_depth_pixels=10)))
    assert adapter.manifest.source_app == "Record3D"
    assert adapter.intrinsics.fx == 30
    assert len(frames) == 1
    assert frames[0].depth_m.shape == (height, width)
    assert np.median(frames[0].depth_m) == pytest.approx(2.25, abs=.002)
    assert np.linalg.det(frames[0].pose[:3, :3]) == pytest.approx(1)


@pytest.mark.parametrize("field,value", [("depth_unit", "unknown"), ("depth_scale", 1), ("registered_rgb_depth", False),
    ("synchronized_rgb_depth", False), ("frame_count", 3), ("pose_convention", "guess")])
def test_bad_manifest(capture, field, value):
    data = json.loads((capture/"manifest.json").read_text())
    data[field] = value
    with pytest.raises(ValidationError):
        CaptureManifest.model_validate(data)


@pytest.mark.parametrize("field,value", [("fx", 0), ("fy", -1), ("width", 0), ("cx", 129), ("cy", -1)])
def test_invalid_intrinsics(capture, field, value):
    data = json.loads((capture/"intrinsics.json").read_text())
    data[field] = value
    with pytest.raises(ValidationError):
        CameraIntrinsics.model_validate(data)


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "reflection", "scale", "last_row"])
def test_pose_validation(kind):
    matrix = np.eye(4)
    if kind == "shape": matrix = np.eye(3)
    elif kind == "nonfinite": matrix[0, 0] = np.nan
    elif kind == "reflection": matrix[0, 0] = -1
    elif kind == "scale": matrix[0, 0] = 2
    else: matrix[3, 0] = 1
    with pytest.raises(InvalidInputError):
        rigid_transform(matrix)


def test_pose_convention_and_axis_transform():
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    conversion = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.]])
    actual = normalize_pose(np.linalg.inv(pose), "world_to_camera", conversion)
    assert np.allclose(actual, conversion@pose)


@pytest.mark.parametrize("missing", ["rgb/000003.png", "depth/000003.png", "pose"])
def test_one_bad_frame_skipped(mutable_capture, missing):
    if missing == "pose":
        edit(mutable_capture/"poses.json", lambda d: d["poses"].pop(3))
    else:
        (mutable_capture/missing).unlink()
    adapter = CanonicalRGBDAdapter(mutable_capture)
    adapter.validate()
    frames = list(adapter.load_frames(LidarConfig()))
    assert len(frames) == 24
    assert adapter.counts.frames_invalid == 1
    assert adapter.counts.frames_loaded == 24
    assert all(frame.frame_id != 3 for frame in frames)


def test_missing_calibration_and_traversal(mutable_capture):
    (mutable_capture/"intrinsics.json").unlink()
    with pytest.raises(InvalidInputError):
        CanonicalRGBDAdapter(mutable_capture).validate()
    with pytest.raises(InvalidInputError):
        CanonicalRGBDAdapter(mutable_capture).path("../escape")


def test_depth_conversion_and_cloud(capture):
    adapter = CanonicalRGBDAdapter(capture)
    adapter.validate()
    frame = next(adapter.load_frames(LidarConfig()))
    with Image.open(capture/"depth/000000.png") as image:
        raw = np.asarray(image)
    assert np.allclose(frame.depth_m, raw/1000, atol=1e-6)
    cloud = frame_cloud(frame, adapter.intrinsics, LidarConfig())
    assert cloud.has_colors() and cloud.has_normals()
    points = np.asarray(cloud.points)
    assert np.median(points[:, 2]) > 1
    world = np.asarray(o3d.geometry.PointCloud(cloud).transform(frame.pose).points)
    assert np.allclose(world, points@frame.pose[:3, :3].T+frame.pose[:3, 3])


def test_meter_depth_and_nonfinite_frame(mutable_capture):
    with Image.open(mutable_capture/"depth/000000.png") as image:
        depth = np.asarray(image).astype(np.float32)/1000
    np.save(mutable_capture/"depth/frame.npy", depth)
    edit(mutable_capture/"manifest.json", lambda d: (d["frames"][0].update(depth_file="depth/frame.npy"), d.update(depth_unit="meter", depth_scale=1)))
    adapter = CanonicalRGBDAdapter(mutable_capture)
    adapter.validate()
    assert np.allclose(next(adapter.load_frames(LidarConfig())).depth_m, depth)
    depth[0, 0] = np.nan
    np.save(mutable_capture/"depth/frame.npy", depth)
    adapter.validate()
    frames = list(adapter.load_frames(LidarConfig()))
    assert all(frame.frame_id != 0 for frame in frames)
    assert adapter.counts.frames_invalid >= 1


def test_dimension_mismatch_fails_instead_of_guessing(mutable_capture):
    Image.new("RGB", (64, 48)).save(mutable_capture/"rgb/000000.png")
    adapter = CanonicalRGBDAdapter(mutable_capture)
    adapter.validate()
    with pytest.raises(InvalidInputError, match="registration"):
        list(adapter.load_frames(LidarConfig()))


def test_sampling_explicit_ids(capture):
    adapter = CanonicalRGBDAdapter(capture)
    adapter.validate()
    frames = build_keyframes(adapter, LidarConfig(frame_stride=3, max_frames=4))
    assert [f.frame_id for f in frames] == [0, 3, 6, 9]
    assert adapter.counts.frames_skipped == 21
    assert adapter.counts.frames_loaded == 25


def test_neighbor_registration(keyframes):
    record = register_pair(keyframes[0], keyframes[1], LidarConfig())
    assert record.accepted
    assert record.fitness > 0.5
    assert np.array(record.information_matrix).shape == (6, 6)
    assert record.inlier_rmse is not None


def test_loop_selection_acceptance_and_rejection(keyframes):
    config = LidarConfig()
    pairs = loop_candidates(keyframes, config)
    assert (0, 24) in pairs
    assert len(pairs) <= len(keyframes)*config.max_loop_candidates_per_frame
    assert register_pair(keyframes[0], keyframes[-1], config, loop=True).accepted
    distant = o3d.geometry.PointCloud(keyframes[-1].cloud).translate([50, 0, 0])
    bad = Keyframe(999, distant, keyframes[-1].pose)
    rejected = register_pair(keyframes[0], bad, config, loop=True)
    assert not rejected.accepted
    assert rejected.rejection_reason


def test_on_outputs_and_integration(on_run):
    output, result = on_run
    assert not result.processing_info.errors
    assert result.capture.tier == "lidar"
    assert len(result.property.rooms[0].walls) == 4
    assert result.property.total_floor_area.value == pytest.approx(20, abs=0.4)
    assert result.property.rooms[0].ceiling.height.value == pytest.approx(2.8, abs=0.05)
    assert result.damages == result.scope_line_items == []
    for name in ("lidar/raw_fused.ply", "lidar/corrected_fused.ply", "lidar/raw_poses.json", "lidar/optimized_poses.json",
                 "floorplan.png", "floorplan.svg", "ablation/floorplan_drift_off.png", "ablation/floorplan_drift_on.svg",
                 "ablation/trajectory_comparison.png", "ablation/drift_floorplan_comparison.png", "processing_config.json"):
        assert (output/name).stat().st_size > 0
    assert load_result(output/"result.json") == result
    graph = o3d.io.read_pose_graph(str(output/"diagnostics/lidar/pose_graph.json"))
    assert len(graph.nodes) == 25
    assert any(edge.uncertain for edge in graph.edges)


def test_synthetic_drift_improves_internal_residuals(on_run):
    output, _ = on_run
    comparison = json.loads((output/"ablation/drift_comparison.json").read_text())
    accepted = [e for e in comparison["edges"] if e["accepted"] and e["rmse_before"] is not None]
    assert np.mean([e["rmse_after"] for e in accepted]) < np.mean([e["rmse_before"] for e in accepted])
    loops = [e for e in accepted if e["kind"] == "loop"]
    assert loops
    assert np.mean([e["constraint_translation_residual_after"] for e in loops]) < np.mean([e["constraint_translation_residual_before"] for e in loops])
    assert "NOT benchmark accuracy" in comparison["interpretation"]
    raw = json.loads((output/"lidar/raw_poses.json").read_text())
    corrected = json.loads((output/"lidar/optimized_poses.json").read_text())
    assert raw != corrected


def test_off_never_runs_icp(capture, tmp_path, monkeypatch):
    import property_scanner.reconstruction.lidar.pipeline as lidar_pipeline
    monkeypatch.setattr(lidar_pipeline, "optimize", lambda *args: pytest.fail("OFF must not run optimization"))
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path))
    prepared = pipeline.prepare(select_adapter("lidar", capture))
    result = pipeline.process(prepared, lidar_config=LidarConfig(drift_correction="off"))
    assert not result.processing_info.errors
    assert (prepared.output_dir/"ablation/floorplan_drift_off.png").exists()
    assert not (prepared.output_dir/"lidar/corrected_fused.ply").exists()
    adapter = CanonicalRGBDAdapter(capture)
    adapter.validate()
    raw = json.loads((prepared.output_dir/"lidar/raw_poses.json").read_text())
    for pose in raw["poses"]:
        assert np.allclose(pose["matrix"], adapter.poses[pose["frame_id"]])


def test_geometry_failure_preserves_reconstruction(capture, tmp_path, monkeypatch):
    import property_scanner.reconstruction.lidar.pipeline as lidar_pipeline
    def fail(*args, **kwargs):
        raise ProcessingError("deliberate test geometry failure")
    monkeypatch.setattr(lidar_pipeline.GeometryEngine, "process_point_cloud", fail)
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path))
    prepared = pipeline.prepare(select_adapter("lidar", capture))
    result = pipeline.process(prepared, lidar_config=LidarConfig(drift_correction="off", max_frames=2))
    assert (prepared.output_dir/"lidar/raw_fused.ply").exists()
    assert (prepared.output_dir/"result.json").exists()
    assert result.processing_info.errors and not result.property.rooms


def test_cli_rejects_malformed_canonical_capture(mutable_capture):
    (mutable_capture/"intrinsics.json").unlink()
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/"run.py"), "--tier", "lidar", "--input", str(mutable_capture), "--drift-correction", "off"],
                            cwd=mutable_capture.parent, capture_output=True, text=True)
    assert result.returncode == 2 and "Traceback" not in result.stderr


def test_no_loop_closure_is_recoverable(capture, tmp_path):
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path))
    prepared = pipeline.prepare(select_adapter("lidar", capture))
    result = pipeline.process(prepared, lidar_config=LidarConfig(max_frames=4, loop_min_frame_separation=100))
    assert "NO_LOOP_CLOSURE" in [w.code for w in result.warnings]
    assert (prepared.output_dir/"lidar/corrected_fused.ply").exists()


def test_pose_ids_do_not_depend_on_file_order(mutable_capture):
    adapter = CanonicalRGBDAdapter(mutable_capture)
    adapter.validate()
    original = adapter.poses[0].copy()
    edit(mutable_capture/"poses.json", lambda d: d["poses"].reverse())
    adapter.validate()
    assert np.allclose(next(adapter.load_frames(LidarConfig())).pose, original)


def test_one_invalid_pose_skips_only_its_frame(mutable_capture):
    edit(mutable_capture/"poses.json", lambda d: d["poses"][2]["matrix"][0].__setitem__(0, 9))
    adapter = CanonicalRGBDAdapter(mutable_capture)
    adapter.validate()
    frames = list(adapter.load_frames(LidarConfig()))
    assert len(frames) == 24
    assert adapter.counts.frames_invalid == 1


def test_missing_units_and_manifest_fail(mutable_capture):
    edit(mutable_capture/"manifest.json", lambda d: d.pop("depth_unit"))
    with pytest.raises(InvalidInputError):
        CanonicalRGBDAdapter(mutable_capture).validate()
    (mutable_capture/"manifest.json").unlink()
    with pytest.raises(InvalidInputError):
        CanonicalRGBDAdapter(mutable_capture).validate()


def test_node_zero_is_anchored(on_run):
    output, _ = on_run
    raw = json.loads((output/"lidar/raw_poses.json").read_text())["poses"][0]["matrix"]
    optimized = json.loads((output/"lidar/optimized_poses.json").read_text())["poses"][0]["matrix"]
    assert np.allclose(raw, optimized, atol=1e-8)
