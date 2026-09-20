"""Deterministic video reconstruction tests; synthetic data is not an accuracy benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFilter
import pytest

o3d = pytest.importorskip("open3d", reason="Install .[video] for video tests")

from config.settings import Settings
from property_scanner.core.exceptions import InvalidInputError, ProcessingError
from property_scanner.inputs import select_adapter
from property_scanner.pipeline.processor import PropertyScanPipeline
from property_scanner.reconstruction.lidar.models import CameraIntrinsics, LidarConfig
from property_scanner.reconstruction.lidar.synthetic import generate_capture
from property_scanner.reconstruction.video.depth import MetricDepthEstimator, inference_device, normalize_metric_config
from property_scanner.reconstruction.video.frames import extract_keyframes, probe_video
from property_scanner.reconstruction.video.models import (
    Observation,
    RegisteredFrame,
    ScaleReport,
    SparsePoint,
    SparseReconstruction,
    VideoConfig,
)
from property_scanner.reconstruction.video.pipeline import metric_keyframe, process_video
from property_scanner.reconstruction.video.scale import estimate_scale, scaled_pose
from property_scanner.reconstruction.video.sfm import SfMReconstructor, parse_model
from property_scanner.schemas.serialization import load_result


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def encode_images(images: Path, destination: Path, *, fps: int = 5) -> Path:
    if not FFMPEG:
        pytest.skip("FFmpeg is required for video integration tests")
    process = subprocess.run(
        [FFMPEG, "-nostdin", "-y", "-v", "error", "-framerate", str(fps),
         "-i", str(images / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    return destination


@pytest.fixture(scope="module")
def rgbd_capture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_capture(tmp_path_factory.mktemp("video_rgbd") / "capture", drift=False, frames=25)


@pytest.fixture(scope="module")
def synthetic_video(rgbd_capture: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return encode_images(rgbd_capture / "rgb", tmp_path_factory.mktemp("video_file") / "walkthrough.mp4")


def make_quality_video(tmp_path: Path, suffix: str = ".mp4") -> Path:
    frames = tmp_path / "source_frames"
    frames.mkdir()
    yy, xx = np.mgrid[:96, :128]
    checker = (((xx // 6 + yy // 6) % 2) * 210 + 30).astype(np.uint8)
    images = [
        Image.fromarray(np.dstack([checker, np.roll(checker, 3, 1), checker])),
        Image.fromarray(np.dstack([checker, np.roll(checker, 3, 1), checker])),
        Image.new("RGB", (128, 96), "black"),
        Image.fromarray(np.dstack([checker] * 3)).filter(ImageFilter.GaussianBlur(8)),
        Image.fromarray(np.dstack([np.roll(checker, 9, 1)] * 3)),
        Image.fromarray(np.dstack([np.roll(checker, 18, 1)] * 3)),
    ]
    for index, image in enumerate(images):
        image.save(frames / f"{index:06d}.png")
    return encode_images(frames, tmp_path / f"quality{suffix}")


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="FFmpeg/ffprobe unavailable")
@pytest.mark.parametrize("suffix", [".mp4", ".mov"])
def test_video_metadata_and_supported_containers(tmp_path: Path, suffix: str) -> None:
    video = make_quality_video(tmp_path, suffix)
    metadata = probe_video(video)
    assert (metadata.width, metadata.height) == (128, 96)
    assert metadata.duration > 0
    assert metadata.frame_count >= 6
    assert metadata.codec == "h264"


def test_invalid_video_fails_cleanly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(InvalidInputError, match="Unreadable/invalid"):
        probe_video(bad)
    unsupported = tmp_path / "video.avi"
    unsupported.touch()
    with pytest.raises(InvalidInputError, match="mp4 or .mov"):
        probe_video(unsupported)


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="FFmpeg/ffprobe unavailable")
def test_quality_filtering_and_chronological_keyframes(tmp_path: Path) -> None:
    video = make_quality_video(tmp_path)
    config = VideoConfig(
        extraction_fps=5, max_extracted_frames=6, max_keyframes=6, min_keyframes=3,
        blur_threshold=20, brightness_threshold=10, duplicate_threshold=1,
        minimum_time_gap=0,
    )
    _, selected = extract_keyframes(video, tmp_path / "output", config)
    report = json.loads((tmp_path / "output/keyframe_selection.json").read_text())
    assert [frame.timestamp for frame in selected] == sorted(frame.timestamp for frame in selected)
    assert report["counts"]["frames_rejected_duplicate"] >= 1
    assert report["counts"]["frames_rejected_dark"] >= 1
    assert report["counts"]["frames_rejected_blur"] >= 1
    assert report["counts"]["frames_used"] == len(selected)


def test_colmap_text_model_parsing(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "cameras.txt").write_text("# cameras\n1 PINHOLE 64 48 50 51 31.5 23.5\n")
    (model / "points3D.txt").write_text("# points\n1 0 0 2 255 0 0 0.2 1 0\n")
    (model / "images.txt").write_text(
        "# images\n1 1 0 0 0 0 0 0 1 000000.jpg\n31.5 23.5 1\n"
        "2 1 0 0 0 -1 0 0 1 000001.jpg\n\n"
    )
    reconstruction = parse_model(model)
    assert len(reconstruction.frames) == 2
    assert reconstruction.frames[0].observations[0].point_id == 1
    assert np.allclose(reconstruction.frames[1].camera_to_world, np.array([
        [1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ]))
    assert reconstruction.frames[0].intrinsics.fy == 51


def scale_fixture(tmp_path: Path, *, noisy: bool = False) -> tuple[SparseReconstruction, dict[str, Path], VideoConfig]:
    rng = np.random.default_rng(9)
    intrinsics = CameraIntrinsics(width=64, height=48, fx=50, fy=50, cx=31.5, cy=23.5)
    points: list[SparsePoint] = []
    frames: list[RegisteredFrame] = []
    depths: dict[str, Path] = {}
    point_id = 0
    for frame_index in range(4):
        pose = np.eye(4)
        pose[0, 3] = frame_index * 0.1
        observations = []
        depth = np.zeros((48, 64), dtype=np.float32)
        for v in range(8, 41, 6):
            for u in range(8, 57, 6):
                point_id += 1
                sfm_z = 1.0 + 0.005 * point_id
                xyz_camera = np.array([(u - 31.5) / 50 * sfm_z, (v - 23.5) / 50 * sfm_z, sfm_z])
                xyz_world = pose[:3, :3] @ xyz_camera + pose[:3, 3]
                points.append(SparsePoint(point_id=point_id, xyz=tuple(xyz_world), reprojection_error=0.1))
                observations.append(Observation(x=u, y=v, point_id=point_id))
                ratio = 2.0 + (rng.normal(0, 0.02) if noisy else 0)
                if noisy and point_id % 29 == 0:
                    ratio = 8.0
                depth[v, u] = sfm_z * ratio
        filename = f"{frame_index:06d}.jpg"
        path = tmp_path / f"{frame_index:06d}.npy"
        np.save(path, depth)
        depths[filename] = path
        frames.append(RegisteredFrame(image_id=frame_index + 1, filename=filename, intrinsics=intrinsics,
            camera_to_world=pose.tolist(), observations=observations))
    config = VideoConfig(scale_min_correspondences=100, scale_min_frame_correspondences=20,
        scale_min_frames=3, scale_image_border=0, depth_min_m=0.1, depth_max_m=20)
    return SparseReconstruction(frames=frames, points=points), depths, config


@pytest.mark.parametrize("noisy", [False, True])
def test_robust_metric_scale_recovery(tmp_path: Path, noisy: bool) -> None:
    sfm, depths, config = scale_fixture(tmp_path, noisy=noisy)
    report = estimate_scale(sfm, depths, config)
    assert report.metric_scale_resolved
    assert report.global_scale == pytest.approx(2, abs=0.04)
    assert report.correspondences_used >= config.scale_min_correspondences
    assert report.scale_confidence_quality is not None
    assert "not benchmark accuracy" in report.interpretation


def test_scale_failure_and_translation_only_scaling(tmp_path: Path) -> None:
    sfm, depths, config = scale_fixture(tmp_path)
    report = estimate_scale(sfm, {next(iter(depths)): next(iter(depths.values()))}, config)
    assert not report.metric_scale_resolved and report.global_scale is None
    assert report.failure_reason == "Insufficient frames with reliable depth/SfM correspondences"
    assert report.correspondences_total == len(sfm.frames[0].observations)
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [1, 2, 3]
    actual = scaled_pose(pose.tolist(), 2)
    assert np.allclose(actual[:3, :3], pose[:3, :3])
    assert np.allclose(actual[:3, 3], [2, 4, 6])


def test_inference_device_prefers_mps_and_falls_back() -> None:
    class MPS:
        def __init__(self, available: bool): self.available = available
        def is_available(self) -> bool: return self.available
    class Backends:
        def __init__(self, available: bool): self.mps = MPS(available)
    class Torch:
        def __init__(self, available: bool): self.backends = Backends(available)
    assert inference_device(Torch(True)) == "mps"
    assert inference_device(Torch(False)) == "cpu"


def test_pinned_metric_checkpoint_legacy_config_is_normalized() -> None:
    legacy = SimpleNamespace(depth_estimation="metric", depth_estimation_type="relative")
    assert normalize_metric_config(legacy).depth_estimation_type == "metric"
    relative = SimpleNamespace(depth_estimation_type="relative")
    assert normalize_metric_config(relative).depth_estimation_type == "relative"


class SyntheticReconstructor(SfMReconstructor):
    """Return an exact arbitrary-scale SfM model for the synthetic RGB-D video."""

    def __init__(self, capture: Path, scale: float = 2.0):
        self.capture = capture
        self.scale = scale

    def reconstruct(self, images: Path, output: Path, config: VideoConfig) -> SparseReconstruction:
        output.mkdir(parents=True, exist_ok=True)
        intrinsics = CameraIntrinsics.model_validate_json((self.capture / "intrinsics.json").read_text())
        truth = {item["frame_id"]: np.asarray(item["matrix"], dtype=float)
                 for item in json.loads((self.capture / "synthetic_ground_truth.json").read_text())["poses"]}
        points, frames = [], []
        point_id = 0
        for image_id, image_path in enumerate(sorted(images.glob("*.jpg")), start=1):
            frame_id = int(image_path.stem)
            pose_metric = truth[frame_id]
            pose_sfm = pose_metric.copy()
            pose_sfm[:3, 3] /= self.scale
            depth = np.asarray(Image.open(self.capture / f"depth/{frame_id:06d}.png"), dtype=float) / 1000
            observations = []
            for v in range(8, intrinsics.height - 8, 10):
                for u in range(8, intrinsics.width - 8, 10):
                    metric_z = depth[v, u]
                    sfm_z = metric_z / self.scale
                    camera = np.array([(u-intrinsics.cx)/intrinsics.fx*sfm_z,
                                       (v-intrinsics.cy)/intrinsics.fy*sfm_z, sfm_z])
                    world = pose_sfm[:3, :3] @ camera + pose_sfm[:3, 3]
                    point_id += 1
                    points.append(SparsePoint(point_id=point_id, xyz=tuple(world), reprojection_error=0.05))
                    observations.append(Observation(x=u, y=v, point_id=point_id))
            frames.append(RegisteredFrame(image_id=image_id, filename=image_path.name, intrinsics=intrinsics,
                camera_to_world=pose_sfm.tolist(), observations=observations))
        reconstruction = SparseReconstruction(frames=frames, points=points,
            statistics={"registered_frames": len(frames), "sparse_points": len(points), "synthetic_test_double": True})
        (output / "sfm_summary.json").write_text(json.dumps(reconstruction.statistics))
        return reconstruction


class SyntheticDepth(MetricDepthEstimator):
    device = "cpu-test-double"

    def __init__(self, capture: Path):
        self.depths = [np.asarray(Image.open(path), dtype=np.float32) / 1000
                       for path in sorted((capture / "depth").glob("*.png"))]
        self.index = 0

    def predict(self, image: Image.Image) -> np.ndarray:
        result = self.depths[self.index]
        self.index += 1
        return result.copy()


@pytest.fixture(scope="module")
def video_run(rgbd_capture: Path, synthetic_video: Path, tmp_path_factory: pytest.TempPathFactory):
    output_root = tmp_path_factory.mktemp("video_output")
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=output_root))
    prepared = pipeline.prepare(select_adapter("video", synthetic_video))
    config = VideoConfig(
        extraction_fps=5, max_extracted_frames=25, max_keyframes=25, min_keyframes=5,
        blur_threshold=0, brightness_threshold=0, duplicate_threshold=0, minimum_time_gap=0,
        depth_min_m=0.1, depth_max_m=8, point_cloud_pixel_stride=1,
        scale_min_correspondences=100, scale_min_frame_correspondences=30, scale_min_frames=3,
        enable_icp_refinement=False,
        registration=LidarConfig(frame_voxel_size=0.05, fusion_voxel_size=0.03,
                                 min_valid_depth_pixels=100, max_fused_points=500000),
    )
    result = process_video(prepared, config, Path("models"),
        reconstructor=SyntheticReconstructor(rgbd_capture), depth_estimator=SyntheticDepth(rgbd_capture))
    return prepared.output_dir, result


def test_metric_point_cloud_backprojection() -> None:
    intrinsics = CameraIntrinsics(width=8, height=6, fx=6, fy=6, cx=3.5, cy=2.5)
    frame = RegisteredFrame(image_id=1, filename="frame.jpg", intrinsics=intrinsics,
                            camera_to_world=np.eye(4).tolist())
    image = Image.new("RGB", (8, 6), "red")
    depth = np.full((6, 8), 2, dtype=np.float32)
    config = VideoConfig(point_cloud_pixel_stride=1, depth_min_m=0.1, depth_max_m=5,
                         registration=LidarConfig(min_valid_depth_pixels=3, frame_voxel_size=0.01))
    keyframe = metric_keyframe(frame, image, depth, 1, config, 0)
    points = np.asarray(keyframe.cloud.points)
    assert len(points) >= 20
    assert np.median(points[:, 2]) == pytest.approx(2)
    assert keyframe.cloud.has_colors() and keyframe.cloud.has_normals()


def test_video_end_to_end_shared_geometry_renderer_and_json(video_run) -> None:
    output, result = video_run
    assert not result.processing_info.errors
    assert result.capture.tier == "video"
    assert result.processing_info.metadata["metric_scale"]["global_scale"] == pytest.approx(2, abs=0.01)
    assert result.property.total_floor_area.value == pytest.approx(20, abs=0.5)
    assert result.property.rooms[0].ceiling.height.value == pytest.approx(2.8, abs=0.08)
    assert len(result.property.rooms[0].walls) == 4
    assert result.damages == result.scope_line_items == []
    for name in ("result.json", "floorplan.png", "floorplan.svg", "measurements.csv",
                 "video/metadata.json", "video/keyframe_selection.json", "video/scale_estimation.json",
                 "video/initial_poses.json", "video/optimized_poses.json", "video/canonical_poses.json",
                 "video/fused_metric_sfm.ply", "video/fused_video.ply", "diagnostics/video/trajectory.png"):
        assert (output / name).stat().st_size > 0
    assert len(list((output / "video/depth").glob("*.npy"))) == 25
    assert load_result(output / "result.json") == result


def test_unresolved_scale_never_becomes_metric_output(rgbd_capture: Path, synthetic_video: Path, tmp_path: Path) -> None:
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path / "outputs"))
    prepared = pipeline.prepare(select_adapter("video", synthetic_video))
    config = VideoConfig(extraction_fps=5, max_extracted_frames=25, max_keyframes=25, min_keyframes=5,
        blur_threshold=0, brightness_threshold=0, duplicate_threshold=0, minimum_time_gap=0,
        scale_min_correspondences=100, scale_min_frame_correspondences=30, scale_min_frames=3,
        depth_min_m=0.1, depth_max_m=1.0, enable_icp_refinement=False)
    result = process_video(prepared, config, Path("models"),
        reconstructor=SyntheticReconstructor(rgbd_capture), depth_estimator=SyntheticDepth(rgbd_capture))
    assert [error.code for error in result.processing_info.errors] == ["METRIC_SCALE_UNRESOLVED"]
    assert not result.property.rooms
    assert (prepared.output_dir / "video/sfm/sparse_relative.ply").exists()
    assert not (prepared.output_dir / "video/fused_video.ply").exists()
    assert not (prepared.output_dir / "floorplan.png").exists()
