"""Photo-room tests use generated images and calibrated synthetic geometry only."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("open3d", reason="Install .[photo] for photo reconstruction tests")

from config.settings import Settings
from property_scanner.core.exceptions import ProcessingError
from property_scanner.inputs import select_adapter
from property_scanner.pipeline.processor import PropertyScanPipeline
from property_scanner.reconstruction.lidar.models import CameraIntrinsics, LidarConfig
from property_scanner.reconstruction.lidar.synthetic import generate_capture
from property_scanner.reconstruction.photo.images import discover_rooms, prepare_room_images
from property_scanner.reconstruction.photo.models import PhotoConfig
from property_scanner.reconstruction.photo.pipeline import process_photo
from property_scanner.reconstruction.video.depth import MetricDepthEstimator
from property_scanner.reconstruction.video.models import Observation, RegisteredFrame, SparsePoint, SparseReconstruction
from property_scanner.reconstruction.video.sfm import SfMReconstructor, _database_statistics
from property_scanner.schemas.serialization import load_result


def patterned_image(path: Path, *, shift: int = 0, size=(128, 96), exif=None) -> Path:
    yy, xx = np.mgrid[:size[1], :size[0]]
    checker = (((xx + shift) // 7 + yy // 7) % 2 * 190 + 30).astype(np.uint8)
    rgb = np.dstack([checker, np.roll(checker, 5, axis=1), np.roll(checker, 9, axis=0)])
    options = {"exif": exif} if exif is not None else {}
    Image.fromarray(rgb).save(path, **options)
    return path


def photo_property(root: Path, rooms: dict[str, int]) -> Path:
    root.mkdir()
    for name, count in rooms.items():
        room = root / name
        room.mkdir()
        for index in range(count):
            patterned_image(room / f"{index:02d}.jpg", shift=index * 11)
    return root


def quality_config(**updates) -> PhotoConfig:
    return PhotoConfig.model_validate({
        **PhotoConfig().model_dump(), "min_image_dimension": 16,
        "blur_threshold": 0, "brightness_min": 0, "brightness_max": 255,
        "contrast_min": 0, "duplicate_threshold": 0, **updates,
    })


def test_property_and_room_discovery(tmp_path: Path) -> None:
    root = photo_property(tmp_path / "property", {"living_room": 3, "room_01": 2})
    rooms = discover_rooms(root)
    assert [(room.room_id, room.label, len(room.images)) for room in rooms] == [
        ("living_room", "Living Room", 3), ("room_01", "Room 01", 2)]
    capture = select_adapter("photo", root).prepare()
    assert len(capture.prepared_files) == 5


def test_flat_room_discovery_remains_supported(tmp_path: Path) -> None:
    patterned_image(tmp_path / "a.jpg")
    patterned_image(tmp_path / "b.png", shift=10)
    room = discover_rooms(tmp_path)[0]
    assert room.room_id == tmp_path.name.lower().replace("-", "_")
    assert len(room.images) == 2


def test_too_few_room_images_fails_room_not_discovery(tmp_path: Path) -> None:
    root = photo_property(tmp_path / "property", {"small_room": 1})
    room = discover_rooms(root)[0]
    with pytest.raises(ProcessingError, match="need at least 2"):
        prepare_room_images(room, tmp_path / "output", quality_config())


def test_best_eight_selected_and_quality_recorded(tmp_path: Path) -> None:
    root = photo_property(tmp_path / "property", {"room": 10})
    prepared = prepare_room_images(discover_rooms(root)[0], tmp_path / "output", quality_config())
    assert len(prepared.selected) == 8
    assert prepared.images_total == 10 and prepared.images_rejected == 2
    report = json.loads((tmp_path / "output/image_quality.json").read_text())
    assert report["counts"]["images_selected"] == 8
    assert all(record["feature_count"] is not None for record in report["images"])


def test_corrupt_image_is_skipped_when_two_valid_remain(tmp_path: Path) -> None:
    root = photo_property(tmp_path / "property", {"room": 2})
    (root / "room/bad.jpg").write_bytes(b"not an image")
    prepared = prepare_room_images(discover_rooms(root)[0], tmp_path / "output", quality_config())
    assert len(prepared.selected) == 2
    assert prepared.images_valid == 2 and prepared.images_rejected == 1
    assert next(record for record in prepared.records if record.source_filename == "bad.jpg").readable is False


def test_exif_orientation_is_normalized(tmp_path: Path) -> None:
    root = tmp_path / "property"
    room = root / "room"
    room.mkdir(parents=True)
    exif = Image.Exif()
    exif[274] = 6
    patterned_image(room / "rotated.jpg", size=(40, 80), exif=exif)
    patterned_image(room / "normal.jpg", shift=20, size=(80, 40))
    prepared = prepare_room_images(discover_rooms(root)[0], tmp_path / "output", quality_config())
    rotated = next(record for record in prepared.selected if record.source_filename == "rotated.jpg")
    with Image.open(prepared.selected_directory / rotated.selected_filename) as image:
        assert image.size == (80, 40)
        assert image.getexif().get(274) == 1
    assert rotated.exif["orientation"] == 6


def test_duplicate_detection_never_reduces_below_two(tmp_path: Path) -> None:
    root = photo_property(tmp_path / "property", {"room": 0})
    room = root / "room"
    patterned_image(room / "a.jpg")
    shutil.copy2(room / "a.jpg", room / "b.jpg")
    patterned_image(room / "c.jpg", shift=30)
    prepared = prepare_room_images(discover_rooms(root)[0], tmp_path / "output",
                                   quality_config(duplicate_threshold=1))
    assert len(prepared.selected) == 2
    assert any(record.rejection_reason == "near_duplicate" for record in prepared.records)


def test_dark_and_blur_quality_rejections_are_structured(tmp_path: Path) -> None:
    root = photo_property(tmp_path / "property", {"room": 2})
    Image.new("RGB", (128, 96), "black").save(root / "room/dark.png")
    prepared = prepare_room_images(discover_rooms(root)[0], tmp_path / "output",
        PhotoConfig(min_image_dimension=16, blur_threshold=20, brightness_min=8,
                    brightness_max=247, contrast_min=5, duplicate_threshold=0))
    dark = next(record for record in prepared.records if record.source_filename == "dark.png")
    assert dark.rejection_reason == "invalid: image contains no non-zero pixel data"
    assert prepared.images_rejected == 1


def test_colmap_feature_matching_statistics(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE keypoints(image_id INTEGER, rows INTEGER);
            CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER);
            INSERT INTO images VALUES(1, 'a.jpg'), (2, 'b.jpg');
            INSERT INTO keypoints VALUES(1, 123), (2, 145);
            INSERT INTO two_view_geometries VALUES(1, 42), (2, 0);
        """)
    assert _database_statistics(database) == {
        "features_per_image": {"a.jpg": 123, "b.jpg": 145},
        "matched_image_pairs": 1, "inlier_matches": 42,
    }


@pytest.mark.skipif(__import__("importlib").util.find_spec("pillow_heif") is None,
                    reason="pillow-heif optional dependency is not installed")
def test_heic_input_when_photo_extra_installed(tmp_path: Path) -> None:
    from pillow_heif import from_pillow
    root = tmp_path / "property"
    room = root / "room"
    room.mkdir(parents=True)
    image = Image.open(patterned_image(tmp_path / "source.png"))
    from_pillow(image).save(room / "a.heic")
    patterned_image(room / "b.jpg", shift=20)
    prepared = prepare_room_images(discover_rooms(root)[0], tmp_path / "output", quality_config())
    assert len(prepared.selected) == 2


class PhotoSyntheticReconstructor(SfMReconstructor):
    def __init__(self, capture: Path, source_ids: list[int], scale: float = 2):
        self.capture, self.source_ids, self.scale = capture, source_ids, scale

    def reconstruct(self, images: Path, output: Path, config: object) -> SparseReconstruction:
        output.mkdir(parents=True, exist_ok=True)
        intrinsics = CameraIntrinsics.model_validate_json((self.capture / "intrinsics.json").read_text())
        truth = {item["frame_id"]: np.asarray(item["matrix"], dtype=float)
                 for item in json.loads((self.capture / "synthetic_ground_truth.json").read_text())["poses"]}
        points, frames, point_id = [], [], 0
        for image_id, (path, source_id) in enumerate(zip(sorted(images.glob("*.jpg")), self.source_ids), start=1):
            pose = truth[source_id].copy()
            pose[:3, 3] /= self.scale
            depth = np.asarray(Image.open(self.capture / f"depth/{source_id:06d}.png"), dtype=float) / 1000
            observations = []
            for v in range(8, intrinsics.height - 8, 10):
                for u in range(8, intrinsics.width - 8, 10):
                    z = depth[v, u] / self.scale
                    camera = np.array([(u-intrinsics.cx)/intrinsics.fx*z,
                                       (v-intrinsics.cy)/intrinsics.fy*z, z])
                    world = pose[:3, :3] @ camera + pose[:3, 3]
                    point_id += 1
                    points.append(SparsePoint(point_id=point_id, xyz=tuple(world), reprojection_error=0.05))
                    observations.append(Observation(x=u, y=v, point_id=point_id))
            frames.append(RegisteredFrame(image_id=image_id, filename=path.name, intrinsics=intrinsics,
                camera_to_world=pose.tolist(), observations=observations))
        statistics = {"registered_frames": len(frames), "sparse_points": len(points),
                      "features_per_image": {frame.filename: 500 for frame in frames},
                      "matched_image_pairs": len(frames)*(len(frames)-1)//2,
                      "inlier_matches": len(points), "matching_strategy": "exhaustive"}
        for name in ("sfm_summary.json", "feature_matching.json"):
            (output / name).write_text(json.dumps(statistics))
        return SparseReconstruction(frames=frames, points=points, statistics=statistics)


class PhotoSyntheticDepth(MetricDepthEstimator):
    device = "cpu-test-double"

    def __init__(self, capture: Path, source_ids: list[int]):
        self.depths = [np.asarray(Image.open(capture / f"depth/{source_id:06d}.png"), dtype=np.float32) / 1000
                       for source_id in source_ids]
        self.index = 0

    def predict(self, image: Image.Image) -> np.ndarray:
        result = self.depths[self.index]
        self.index += 1
        return result.copy()


@pytest.fixture(scope="module")
def synthetic_photo_capture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_capture(tmp_path_factory.mktemp("photo_rgbd") / "capture", drift=False, frames=25)


def copy_room_photos(capture: Path, room: Path, source_ids: list[int]) -> None:
    room.mkdir(parents=True)
    for index, source_id in enumerate(source_ids):
        shutil.copy2(capture / f"rgb/{source_id:06d}.png", room / f"{index:02d}.png")


@pytest.fixture(scope="module")
def photo_run(synthetic_photo_capture: Path, tmp_path_factory: pytest.TempPathFactory):
    source_ids = [0, 3, 6, 9, 12, 15, 18, 21]
    root = tmp_path_factory.mktemp("photo_property") / "property"
    copy_room_photos(synthetic_photo_capture, root / "living_room", source_ids)
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path_factory.mktemp("photo_output")))
    prepared = pipeline.prepare(select_adapter("photo", root))
    config = quality_config(depth_min_m=0.1, depth_max_m=8, point_cloud_pixel_stride=1,
        scale_min_correspondences=50, scale_min_frame_correspondences=20, scale_min_frames=2,
        enable_icp_refinement=False,
        registration=LidarConfig(frame_voxel_size=0.05, fusion_voxel_size=0.03,
                                 min_valid_depth_pixels=100, max_fused_points=500000))
    result = process_photo(prepared, config, Path("models"),
        reconstructor_factory=lambda: PhotoSyntheticReconstructor(synthetic_photo_capture, source_ids),
        depth_estimator=PhotoSyntheticDepth(synthetic_photo_capture, source_ids))
    return prepared.output_dir, result


def test_photo_room_end_to_end_geometry_renderer_and_json(photo_run) -> None:
    output, result = photo_run
    assert not result.processing_info.errors
    assert result.capture.tier == "photo"
    assert len(result.property.rooms) == 1
    room = result.property.rooms[0]
    assert room.name == "Living Room"
    assert room.metadata["coordinate_scope"] == "room_local_unstitched"
    assert room.floor.area.value == pytest.approx(20, abs=0.5)
    assert room.ceiling.height.value == pytest.approx(2.8, abs=0.08)
    assert len(room.walls) == 4
    assert not result.property.room_connections and not result.property.openings
    base = output / "photo/living_room"
    for path in (base / "room_fused.ply", base / "geometry.json", base / "room.json",
                 base / "floorplan.png", base / "floorplan.svg", base / "scale_estimation.json",
                 output / "result.json", output / "photo_summary.json"):
        assert path.stat().st_size > 0
    assert load_result(output / "result.json") == result


def test_multi_room_partial_failure_retains_success(synthetic_photo_capture: Path, tmp_path: Path) -> None:
    source_ids = [0, 3, 6, 9, 12, 15, 18, 21]
    root = tmp_path / "property"
    copy_room_photos(synthetic_photo_capture, root / "good_room", source_ids)
    copy_room_photos(synthetic_photo_capture, root / "sparse_room", [0])
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path / "outputs"))
    prepared = pipeline.prepare(select_adapter("photo", root))
    config = quality_config(depth_min_m=0.1, depth_max_m=8, point_cloud_pixel_stride=1,
        scale_min_correspondences=50, scale_min_frame_correspondences=20,
        registration=LidarConfig(frame_voxel_size=0.05, fusion_voxel_size=0.03,
                                 min_valid_depth_pixels=100, max_fused_points=500000))
    result = process_photo(prepared, config, Path("models"),
        reconstructor_factory=lambda: PhotoSyntheticReconstructor(synthetic_photo_capture, source_ids),
        depth_estimator=PhotoSyntheticDepth(synthetic_photo_capture, source_ids))
    assert [room.room_id for room in result.property.rooms] == ["good_room"]
    assert [error.code for error in result.processing_info.errors] == ["IMAGE_VALIDATION_FAILED"]
    assert result.processing_info.metadata["rooms_successful"] == 1
    assert result.processing_info.metadata["rooms_failed"] == 1
    assert (prepared.output_dir / "photo/good_room/floorplan.png").exists()
    assert not (prepared.output_dir / "photo/sparse_room/floorplan.png").exists()


def test_metric_scale_failure_emits_no_measurements(synthetic_photo_capture: Path, tmp_path: Path) -> None:
    source_ids = [0, 3, 6]
    root = tmp_path / "property"
    copy_room_photos(synthetic_photo_capture, root / "room", source_ids)
    pipeline = PropertyScanPipeline(Settings(_env_file=None, output_dir=tmp_path / "outputs"))
    prepared = pipeline.prepare(select_adapter("photo", root))
    config = quality_config(depth_min_m=0.1, depth_max_m=0.5,
                            scale_min_correspondences=50, scale_min_frame_correspondences=15)
    result = process_photo(prepared, config, Path("models"),
        reconstructor_factory=lambda: PhotoSyntheticReconstructor(synthetic_photo_capture, source_ids),
        depth_estimator=PhotoSyntheticDepth(synthetic_photo_capture, source_ids))
    assert [error.code for error in result.processing_info.errors] == ["METRIC_SCALE_UNRESOLVED"]
    assert not result.property.rooms and result.property.total_floor_area is None
    assert (prepared.output_dir / "photo/room/sfm/sparse_relative.ply").exists()
    assert not (prepared.output_dir / "photo/room/room_fused.ply").exists()
    assert not (prepared.output_dir / "photo/room/floorplan.png").exists()
