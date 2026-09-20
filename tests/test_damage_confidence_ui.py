"""Integration tests for damage, uncertainty, final export, and UI helpers."""
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import pytest

from app import result_summary, safe_extract_zip
from property_scanner.confidence.estimator import ConfidenceConfig, ConfidenceEstimator
from property_scanner.damage.detector import DamageDetector, build_scope
from property_scanner.damage.models import DamageConfig, DamageFrame, DamagePrediction
from property_scanner.damage.vision import DamageVisionModel
from property_scanner.damage.vision import _damage_type
from property_scanner.pipeline.finalize import write_measurements_csv
from property_scanner.schemas.capture import CaptureMetadata
from property_scanner.schemas.damage import DamageRegion, DamageType
from property_scanner.schemas.geometry import CeilingSurface, FloorSurface, PropertyGeometry, Room, Wall
from property_scanner.schemas.measurements import AreaMeasurement, ConfidenceMethod, LengthMeasurement
from property_scanner.schemas.primitives import Point2D, Polygon2D
from property_scanner.schemas.result import ProcessingInfo, PropertyScanResult


class FakeDamage(DamageVisionModel):
    device = "cpu-test-double"
    def __init__(self, mask): self.mask = mask
    def predict(self, image): return [DamagePrediction(DamageType.WATER_DAMAGE, self.mask, .9)]


def geometry():
    points = [Point2D(x=-2,y=0),Point2D(x=2,y=0),Point2D(x=2,y=3),Point2D(x=-2,y=3)]
    polygon=Polygon2D(points=points); walls=[]
    for index,a in enumerate(points):
        b=points[(index+1)%4]; walls.append(Wall(wall_id=f"r:wall:{index}",room_id="r",start_point=a,end_point=b,
            length=LengthMeasurement(value=float(np.hypot(b.x-a.x,b.y-a.y)))))
    room=Room(room_id="r",polygon=polygon,walls=walls,
        floor=FloorSurface(surface_id="r:floor",polygon=polygon,area=AreaMeasurement(value=12)),
        ceiling=CeilingSurface(surface_id="r:ceiling",polygon=polygon,height=LengthMeasurement(value=2.8)))
    return PropertyGeometry(property_id="p",rooms=[room])


def frame(tmp_path, index):
    image=tmp_path/f"{index}.jpg"; depth=tmp_path/f"{index}.npy"
    Image.new("RGB",(100,100),(120,120,120)).save(image); np.save(depth,np.full((100,100),2,dtype=np.float32))
    pose=np.array([[1,0,0,0],[0,0,-1,2],[0,-1,0,1],[0,0,0,1]],float)
    return DamageFrame(frame_id=f"f{index}",image_path=image,depth_path=depth,
        intrinsics={"width":100,"height":100,"fx":100,"fy":100,"cx":50,"cy":50},
        camera_to_property=pose.tolist(),room_id="r",depth_source="sensor")


def test_metric_damage_fusion_scope_and_concealed_rule(tmp_path):
    mask=np.zeros((100,100),bool); mask[20:70,30:70]=True
    detector=DamageDetector(DamageConfig(minimum_pixel_area=20,minimum_3d_points=10,
        depth_sample_stride=2,minimum_supporting_views=2),FakeDamage(mask))
    prop,regions,flags,scope,warnings=detector.detect(geometry(),[frame(tmp_path,0),frame(tmp_path,1)])
    assert not warnings and len(regions)==1
    assert regions[0].surface_id=="r:wall:0" and regions[0].metric_area.value==pytest.approx(.72,abs=.12)
    assert regions[0].metadata["supporting_views"]==2 and prop.rooms[0].damage_ids==[regions[0].damage_id]
    assert flags[0].rule_id=="RULE_WATER_STAIN_CONCEALED_MOISTURE"
    assert all(item.damage_id==regions[0].damage_id and item.quantity for item in scope)


def test_damage_vocabulary_mapping_is_conservative():
    assert _damage_type("water damage") == DamageType.WATER_DAMAGE
    assert _damage_type("crack") == DamageType.CRACK
    assert _damage_type("water bottle") == DamageType.UNKNOWN
    assert _damage_type("molding") == DamageType.UNKNOWN


def result():
    return PropertyScanResult(capture=CaptureMetadata(capture_id="00000000-0000-4000-8000-000000000001",tier="lidar",processing_timestamp=datetime.now(timezone.utc)),
        property=geometry(),processing_info=ProcessingInfo())


def test_confidence_profile_precedence_and_uncalibrated_label(tmp_path):
    profile=tmp_path/"lidar.json"; profile.write_text('{"tier":"lidar","confidence_level":0.95,"absolute_error_by_type":{"wall_length":0.02},"benchmark_rows":20}')
    scan=result(); ConfidenceEstimator(ConfidenceConfig(profile_directory=tmp_path)).apply(scan)
    assert scan.property.rooms[0].walls[0].length.confidence_method==ConfidenceMethod.BENCHMARK_CALIBRATED
    assert scan.property.rooms[0].walls[0].length.lower_bound <= 4 <= scan.property.rooms[0].walls[0].length.upper_bound
    assert scan.property.rooms[0].floor.area.confidence_method==ConfidenceMethod.UNAVAILABLE
    uncal=result(); ConfidenceEstimator(ConfidenceConfig(profile_directory=tmp_path/"missing",allow_tier_prior_uncalibrated=True)).apply(uncal)
    assert uncal.property.rooms[0].walls[0].length.confidence_method==ConfidenceMethod.TIER_PRIOR_UNCALIBRATED
    assert uncal.property.rooms[0].floor.area.confidence_method==ConfidenceMethod.TIER_PRIOR_UNCALIBRATED
    assert uncal.property.rooms[0].ceiling.height.lower_bound <= 2.8 <= uncal.property.rooms[0].ceiling.height.upper_bound


def test_measurement_csv_and_ui_helpers(tmp_path):
    scan=result(); write_measurements_csv(tmp_path/"measurements.csv",scan)
    text=(tmp_path/"measurements.csv").read_text(); assert "confidence_method" in text and "wall_length" in text
    assert result_summary(scan.model_dump(mode="json"))["Rooms detected"]==1
    archive=tmp_path/"safe.zip"
    with zipfile.ZipFile(archive,"w") as handle: handle.writestr("room/01.jpg",b"x")
    assert (safe_extract_zip(archive,tmp_path/"safe")/"01.jpg").is_file()
    bad=tmp_path/"bad.zip"
    with zipfile.ZipFile(bad,"w") as handle: handle.writestr("../escape",b"x")
    with pytest.raises(ValueError,match="Unsafe ZIP"): safe_extract_zip(bad,tmp_path/"unsafe")
