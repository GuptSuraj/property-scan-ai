"""Structural and semantic renderer tests, beyond image snapshots."""
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET
import pytest
pytest.importorskip("matplotlib", reason="Install .[rendering] for renderer tests")
pytest.importorskip("shapely", reason="Install .[rendering] for renderer tests")
import numpy as np
from PIL import Image
from shapely.geometry import Point
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.rendering.exceptions import InvalidGeometryError, FloorPlanRenderError
from property_scanner.rendering.scene import build_scene
from property_scanner.rendering.openings import locate_openings, wall_intervals
from property_scanner.schemas.geometry import PropertyGeometry, Opening
from property_scanner.schemas.geometry import Wall
from property_scanner.schemas.primitives import Polygon2D
from property_scanner.schemas.measurements import LengthMeasurement
from property_scanner.schemas.result import PropertyScanResult
from scripts.render_sample_floorplan import sample_room

ROOT = Path(__file__).resolve().parents[1]


def labels(drawing) -> str:
    return "\n".join(t.get_text() for t in drawing.figure.axes[0].texts)


def test_png_svg_and_labels(tmp_path):
    room, openings = sample_room()
    drawing = FloorPlanRenderer().render_room(room, openings=openings)
    try:
        files = drawing.save(tmp_path / "capture")
        with Image.open(files.png) as image:
            image.verify()
        assert files.png.stat().st_size > 1000
        root = ET.parse(files.svg).getroot()
        assert root.tag.endswith("svg") and "viewBox" in root.attrib
        assert not list(root.iter("{http://www.w3.org/2000/svg}image"))
        text = " ".join(root.itertext())
        for expected in ("Sample room", "20.00 m²", "Ceiling: 2.80 m", "4.00 m", "5.00 m"):
            assert expected in text
        assert drawing.figure.axes[0].get_aspect() == 1
        assert not drawing.warnings
    finally:
        drawing.close()


def test_irregular_shape_and_label_inside():
    room, _ = sample_room("l")
    drawing = FloorPlanRenderer().render_room(room)
    ax = drawing.figure.axes[0]
    patch = ax.patches[0]
    assert len(patch.get_xy()) == 7
    assert np.allclose(patch.get_xy()[:-1], [(p.x, p.y) for p in room.polygon.points])
    scene = build_scene(PropertyGeometry(property_id="p", rooms=[room]), RenderingConfig())
    label = next(text for text in ax.texts if text.get_gid() == "label-room_01")
    assert scene.rooms[0].polygon.contains(Point(label.get_position()))
    assert "19.00 m²" in labels(drawing)
    drawing.close()


def test_missing_height_and_unknown_area():
    room, _ = sample_room(ceiling=False)
    room.floor.area = None
    drawing = FloorPlanRenderer().render_room(room)
    assert "Ceiling: unavailable" in labels(drawing)
    assert "0.00" not in labels(drawing)
    assert "m²" not in labels(drawing)
    drawing.close()
    drawing = FloorPlanRenderer(RenderingConfig(show_missing_ceiling=False)).render_room(room)
    assert "Ceiling:" not in labels(drawing)
    drawing.close()


@pytest.mark.parametrize("kind", ["door", "window", "open_passage", "unknown"])
def test_openings_have_correct_positions_and_gap(kind):
    room, _ = sample_room()
    opening = Opening(opening_id="o", type=kind, wall_id="wall_1", width={"value": 0.9}, position_along_wall={"value": 1.2})
    prop = PropertyGeometry(property_id="p", rooms=[room], openings=[opening])
    config = RenderingConfig()
    scene = build_scene(prop, config)
    positions, warnings = locate_openings(prop, scene, config)
    assert not warnings
    assert np.allclose(positions[0].start, [1.2, 0])
    assert np.allclose(positions[0].end, [2.1, 0])
    assert wall_intervals(*scene.rooms[0].walls[0], positions, config) == pytest.approx([(1.2, 2.1)])
    drawing = FloorPlanRenderer().render_property(prop)
    wall_lines = [line for line in drawing.figure.axes[0].lines if (line.get_gid() or "").startswith("wall-wall_1-")]
    assert len(wall_lines) == 2
    assert np.allclose(wall_lines[0].get_xdata(), [0, 1.2])
    assert np.allclose(wall_lines[1].get_xdata(), [2.1, 4])
    if kind == "window":
        marker_lines = [line for line in drawing.figure.axes[0].lines if np.allclose(line.get_xdata(), [1.2, 2.1])]
        assert len(marker_lines) == 2
        assert sorted(float(line.get_ydata()[0]) for line in marker_lines) == pytest.approx([-0.03, 0.03])
    drawing.close()


def test_missing_opening_width_warns_and_bad_position_fails():
    room, openings = sample_room()
    openings[0].width = None
    drawing = FloorPlanRenderer().render_room(room, openings=openings)
    assert drawing.warnings[0].code == "OPENING_NOT_POSITIONED"
    drawing.close()
    openings[1].position_along_wall = LengthMeasurement(value=5)
    with pytest.raises(InvalidGeometryError, match="beyond"):
        FloorPlanRenderer().render_room(room, openings=openings)


def test_confidence_and_precision_preserve_input():
    room, _ = sample_room()
    room.walls[0].length = LengthMeasurement(value=4.523746, lower_bound=4.48, upper_bound=4.56, confidence_level=0.95)
    before = room.model_dump_json()
    drawing = FloorPlanRenderer(RenderingConfig(show_confidence=True)).render_room(room)
    assert "4.52 m\n95% CI: 4.48–4.56 m" in labels(drawing)
    drawing.close()
    drawing = FloorPlanRenderer(RenderingConfig(display_precision=3)).render_room(room)
    assert "4.524 m" in labels(drawing) and "CI:" not in labels(drawing)
    assert room.model_dump_json() == before
    drawing.close()


def test_multiple_positioned_rooms_and_shared_door():
    fixture = PropertyScanResult.model_validate_json((ROOT / "tests/fixtures/sample_property_result.json").read_text())
    drawing = FloorPlanRenderer().render_property(fixture.property)
    assert len(drawing.figure.axes[0].patches) == 2
    assert "Living room" in labels(drawing) and "Bedroom" in labels(drawing)
    for wall_id in ("wall_1_2", "wall_2_4"):
        lines = [line for line in drawing.figure.axes[0].lines if (line.get_gid() or "").startswith(f"wall-{wall_id}-")]
        assert len(lines) == 2
    drawing.close()


@pytest.mark.parametrize("fault", ["polygon", "walls", "endpoints", "self_intersection"])
def test_invalid_geometry(fault):
    room, _ = sample_room()
    if fault == "polygon":
        room.polygon = None
    elif fault == "walls":
        room.walls = []
    elif fault == "endpoints":
        room.walls[0].end_point = room.walls[0].start_point
    else:
        room.polygon.points[1], room.polygon.points[2] = room.polygon.points[2], room.polygon.points[1]
    with pytest.raises(InvalidGeometryError):
        FloorPlanRenderer().render_room(room)


def test_scale_bar_north_and_overall():
    room, _ = sample_room()
    renderer = FloorPlanRenderer(RenderingConfig(show_north_arrow=True, show_overall_dimensions=True))
    drawing = renderer.render_room(room)
    assert "N" not in [t.get_text() for t in drawing.figure.axes[0].texts]
    assert "Overall width: 4.00 m" in labels(drawing)
    scale = next(line for line in drawing.figure.axes[0].lines if line.get_gid() == "scale-bar")
    assert np.ptp(scale.get_xdata()) == 1
    drawing.close()
    drawing = renderer.render_room(room, north_angle_degrees=90)
    assert "N" in [t.get_text() for t in drawing.figure.axes[0].texts]
    drawing.close()


def test_export_error(tmp_path):
    room, _ = sample_room()
    drawing = FloorPlanRenderer().render_room(room)
    blocker = tmp_path / "file"
    blocker.touch()
    with pytest.raises(FloorPlanRenderError):
        drawing.save(blocker)
    drawing.close()


def test_sample_script(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/render_sample_floorplan.py"), "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Sample floor plan generated successfully." in result.stdout
    assert (tmp_path / "floorplan.svg").is_file()


def test_geometry_engine_result_renders(tmp_path):
    pytest.importorskip("open3d")
    from tests.synthetic_geometry import room_points, write_cloud
    from property_scanner.geometry.engine import GeometryEngine
    result = GeometryEngine().process_point_cloud(write_cloud(tmp_path / "synthetic.ply", room_points()))
    drawing = FloorPlanRenderer().render_room(result, name="Geometry room")
    assert "20.00 m²" in labels(drawing)
    assert "Ceiling: 2.80 m" in labels(drawing)
    assert len(drawing.figure.axes[0].patches) == 1
    drawing.close()


def test_label_fallback_for_outside_centroid():
    room, _ = sample_room()
    room.polygon = Polygon2D(points=[dict(x=x, y=y) for x, y in [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (4, 4), (4, 5), (0, 5)]])
    room.walls = [Wall(wall_id=f"w{i}", room_id=room.room_id, start_point=p,
                       end_point=room.polygon.points[(i+1)%8]) for i, p in enumerate(room.polygon.points)]
    room.floor.area = None
    scene = build_scene(PropertyGeometry(property_id="p", rooms=[room]), RenderingConfig())
    polygon = scene.rooms[0].polygon
    assert not polygon.contains(polygon.centroid)
    drawing = FloorPlanRenderer().render_room(room)
    text = next(t for t in drawing.figure.axes[0].texts if t.get_gid() == "label-room_01")
    assert polygon.contains(Point(text.get_position()))
    drawing.close()


def test_metric_scale_is_equal_and_translation_does_not_mutate_input():
    room, _ = sample_room()
    for point in room.polygon.points:
        point.x += 100
        point.y -= 200
    # Fixture walls share the polygon's point instances, so they are translated too.
    before = room.model_dump_json()
    drawing = FloorPlanRenderer().render_room(room)
    ax = drawing.figure.axes[0]
    a, b, c = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
    assert np.linalg.norm(b-a) == pytest.approx(np.linalg.norm(c-a))
    assert np.linalg.norm(b-a) == pytest.approx(drawing.config.dpi*drawing.config.inches_per_meter)
    assert room.model_dump_json() == before
    assert np.allclose(ax.patches[0].get_xy()[0], [0, 0])
    drawing.close()


def test_transparent_background(tmp_path):
    room, _ = sample_room()
    drawing = FloorPlanRenderer(RenderingConfig(background="transparent")).render_room(room)
    path = drawing.save_png(tmp_path / "transparent.png")
    with Image.open(path) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
    drawing.close()


def test_render_json_cli(tmp_path):
    room, openings = sample_room()
    source = tmp_path / "property.json"
    source.write_text(PropertyGeometry(property_id="p", rooms=[room], openings=openings).model_dump_json())
    result = subprocess.run([sys.executable, str(ROOT / "scripts/render_floorplan.py"), "--input", str(source),
                             "--input-kind", "property", "--output-dir", str(tmp_path / "capture")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "capture/floorplan.png").is_file()
    source.write_text("invalid")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/render_floorplan.py"), "--input", str(source)], capture_output=True, text=True)
    assert result.returncode == 2 and "Traceback" not in result.stderr


def test_pipeline_renderer_consumes_only_supplied_result(tmp_path):
    from property_scanner.pipeline.context import ScanContext
    from property_scanner.schemas.common import NormalizedCapture
    fixture = PropertyScanResult.model_validate_json((ROOT / "tests/fixtures/sample_property_result.json").read_text())
    context = ScanContext(NormalizedCapture(tier="photo", source_path=tmp_path), tmp_path, result=fixture)
    returned = FloorPlanRenderer().run(context)
    assert returned is context
    assert context.artifacts["floorplan_png"].is_file()
    assert context.artifacts["floorplan_svg"].is_file()
