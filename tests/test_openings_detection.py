"""Synthetic metric tests for shared semantic/geometry opening detection."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from property_scanner.openings.detector import OpeningDetector
from property_scanner.openings.models import OpeningConfig, OpeningFrame
from property_scanner.openings.semantic import SemanticSurfaceDetector
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.schemas.geometry import (CeilingSurface, FloorSurface, PropertyGeometry,
    Room, RoomConnection, Wall)
from property_scanner.schemas.measurements import AreaMeasurement, LengthMeasurement
from property_scanner.schemas.primitives import Point2D, Polygon2D


class FakeSemantic(SemanticSurfaceDetector):
    device = "cpu-test-double"
    def __init__(self, predictions): self.predictions, self.index = predictions, 0
    def predict(self, image):
        item = self.predictions[self.index % len(self.predictions)]; self.index += 1
        return {key: value.copy() for key, value in item.items()}


def room(room_id="room_a", below=False):
    points = [(0, -3), (5, -3), (5, 0), (0, 0)] if below else [(0, 0), (5, 0), (5, 3), (0, 3)]
    vertices = [Point2D(x=float(x), y=float(y)) for x, y in points]
    polygon = Polygon2D(points=vertices)
    walls=[]
    for index, start in enumerate(vertices):
        end=vertices[(index+1)%4]
        walls.append(Wall(wall_id=f"{room_id}:wall:{index}",room_id=room_id,start_point=start,end_point=end,
            length=LengthMeasurement(value=float(np.hypot(end.x-start.x,end.y-start.y)))))
    return Room(room_id=room_id,name=room_id,polygon=polygon,walls=walls,
        floor=FloorSurface(surface_id=f"{room_id}:floor",polygon=polygon,area=AreaMeasurement(value=15)),
        ceiling=CeilingSurface(surface_id=f"{room_id}:ceiling",polygon=polygon,height=LengthMeasurement(value=2.8)))


def mask_for(x0, x1, z0, z1, score=.95):
    # Camera looks +Y from (0,-2,2.2), optical X -> world X and optical down -> -Z.
    result=np.zeros((300,400),dtype=np.float32)
    u0,u1=round(100+50*x0),round(100+50*x1)
    row_top,row_bottom=round(150+(2.2-z1)*50),round(150+(2.2-z0)*50)
    result[row_top:row_bottom+1,u0:u1+1]=score
    return result


def evidence(tmp_path: Path, count, predictions, depth_value=2.0):
    pose=np.eye(4); pose[:3,:3]=np.array([[1,0,0],[0,0,1],[0,-1,0]]); pose[:3,3]=[0,-2,2.2]
    frames=[]
    for index in range(count):
        image=tmp_path/f"{index}.jpg"; depth=tmp_path/f"{index}.npy"
        Image.new("RGB",(400,300),(120,120,120)).save(image)
        np.save(depth,np.full((300,400),depth_value,dtype=np.float32))
        frames.append(OpeningFrame(frame_id=f"frame:{index}",image_path=image,depth_path=depth,
            intrinsics={"width":400,"height":300,"fx":100,"fy":100,"cx":100,"cy":150},
            camera_to_property=pose.tolist(),room_id="room_a",depth_source="sensor"))
    return frames, FakeSemantic(predictions)


def config(**updates):
    return OpeningConfig.model_validate({**OpeningConfig().model_dump(),
        "candidate_min_pixel_area":20,"minimum_3d_points":10,"depth_sample_stride":2,
        "minimum_supporting_views":1,"single_view_min_quality":.5,**updates})


def prediction(doors=(),windows=()):
    door=np.zeros((300,400),np.float32); window=door.copy()
    for values in doors: door=np.maximum(door,mask_for(*values))
    for values in windows: window=np.maximum(window,mask_for(*values))
    return {"door":door,"window":window,"wall":np.ones_like(door)}


def test_metric_door_measurement_and_wall_position(tmp_path):
    frames, semantic=evidence(tmp_path,1,[prediction(doors=[(1.48,2.42,0,2.1)])])
    result=OpeningDetector(config(),semantic).detect(PropertyGeometry(property_id="p",rooms=[room()]),frames)
    opening=result.property.openings[0]
    assert opening.type == "door"
    assert opening.width.value == pytest.approx(.9,abs=.04)
    assert opening.height.value == pytest.approx(2.02,abs=.05)
    assert opening.position_along_wall.value == pytest.approx(1.5,abs=.04)
    assert opening.sill_height is None and opening.width.lower_bound is None


def test_metric_window_measurement_and_sill(tmp_path):
    frames, semantic=evidence(tmp_path,1,[prediction(windows=[(2.97,4.43,.88,2.12)])])
    result=OpeningDetector(config(),semantic).detect(PropertyGeometry(property_id="p",rooms=[room()]),frames)
    opening=result.property.openings[0]
    assert opening.type == "window"
    assert opening.width.value == pytest.approx(1.4,abs=.05)
    assert opening.height.value == pytest.approx(1.2,abs=.05)
    assert opening.sill_height.value == pytest.approx(.9,abs=.04)


def test_two_doors_stay_separate_and_four_views_fuse(tmp_path):
    item=prediction(doors=[(.98,1.92,0,2.1),(3.08,4.02,0,2.1)])
    frames, semantic=evidence(tmp_path,4,[item])
    result=OpeningDetector(config(minimum_supporting_views=2),semantic).detect(
        PropertyGeometry(property_id="p",rooms=[room()]),frames)
    assert len(result.property.openings)==2
    assert all(opening.metadata["number_of_views"]==4 for opening in result.property.openings)


def test_multiview_median_rejects_measurement_outlier(tmp_path):
    items=[prediction(doors=[(1.48,2.42,0,2.1)]),prediction(doors=[(1.47,2.43,0,2.1)]),
           prediction(doors=[(1.25,2.65,0,2.1)])]
    frames,semantic=evidence(tmp_path,3,items)
    result=OpeningDetector(config(minimum_supporting_views=2),semantic).detect(
        PropertyGeometry(property_id="p",rooms=[room()]),frames)
    assert len(result.property.openings)==1
    assert result.property.openings[0].width.value==pytest.approx(.92,abs=.08)
    assert result.property.openings[0].metadata["width_mad_m"]<.1


def test_door_window_conflict_becomes_unknown(tmp_path):
    items=[prediction(doors=[(1.48,2.42,0,2.1)]),prediction(windows=[(1.48,2.42,.9,2.1)])]
    frames,semantic=evidence(tmp_path,2,items)
    result=OpeningDetector(config(minimum_supporting_views=2),semantic).detect(
        PropertyGeometry(property_id="p",rooms=[room()]),frames)
    assert result.property.openings[0].type=="unknown"
    assert "OPENING_TYPE_AMBIGUOUS" in {warning.code for warning in result.warnings}


def test_phantom_semantic_candidate_rejected_by_wall_geometry(tmp_path):
    frames, semantic=evidence(tmp_path,1,[prediction(doors=[(1.5,2.4,0,2.1)])],depth_value=4)
    result=OpeningDetector(config(),semantic).detect(PropertyGeometry(property_id="p",rooms=[room()]),frames)
    assert not result.property.openings
    assert "OPENING_WALL_ASSOCIATION_FAILED" in {warning.code for warning in result.warnings}


def test_geometry_only_fallback_is_unknown_or_passage(tmp_path):
    frames, semantic=evidence(tmp_path,1,[prediction()])
    xs=np.arange(.02,5,.04); zs=np.arange(.02,2.8,.04)
    points=np.array([(x,0,z) for x in xs for z in zs if not (1.5<x<2.42 and z<2.12)])
    result=OpeningDetector(config(occupancy_cell_size=.04),semantic).detect(
        PropertyGeometry(property_id="p",rooms=[room()]),frames,structural_points=points)
    assert len(result.property.openings)==1
    assert result.property.openings[0].type in ("unknown","open_passage")


def test_connection_reference_and_renderer_outputs(tmp_path):
    first,second=room(),room("room_b",below=True)
    geometry=PropertyGeometry(property_id="p",rooms=[first,second],
        room_connections=[RoomConnection(connection_id="c",room_a_id="room_a",room_b_id="room_b",connection_type="inferred")],
        metadata={"coordinate_scope":"property_shared"})
    frames,semantic=evidence(tmp_path,1,[prediction(doors=[(1.48,2.42,0,2.1)])])
    result=OpeningDetector(config(),semantic).detect(geometry,frames)
    opening=result.property.openings[0]
    assert opening.room_ids==["room_a","room_b"]
    assert result.property.room_connections[0].opening_id==opening.opening_id
    assert result.property.room_connections[0].connection_type=="door"
    # Structural dimensions are unchanged.
    assert result.property.rooms[0].walls[0].length.value==5
    assert result.property.rooms[0].floor.area.value==15
    drawing=FloorPlanRenderer(RenderingConfig(show_opening_labels=True)).render_property(result.property)
    try:
        files=drawing.save(tmp_path/"rendered")
        assert files.png.stat().st_size>1000 and files.svg.stat().st_size>1000
        expected=f"{opening.width.value:.2f} m"
        assert any("Door" in text.get_text() and expected in text.get_text() for text in drawing.figure.axes[0].texts)
    finally: drawing.close()
