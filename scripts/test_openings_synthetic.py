"""Generate synthetic metric door/window evidence, JSON, diagnostics, and floor plan."""

from pathlib import Path
import shutil
from uuid import UUID
import numpy as np
from PIL import Image

from property_scanner.openings.detector import OpeningDetector
from property_scanner.openings.models import OpeningConfig, OpeningFrame
from property_scanner.openings.semantic import SemanticSurfaceDetector
from property_scanner.reconstruction.lidar.diagnostics import write_json
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.schemas.capture import CaptureMetadata
from property_scanner.schemas.geometry import CeilingSurface, FloorSurface, PropertyGeometry, Room, Wall
from property_scanner.schemas.measurements import AreaMeasurement, LengthMeasurement
from property_scanner.schemas.primitives import Point2D, Polygon2D
from property_scanner.schemas.result import ProcessingInfo, PropertyScanResult
from property_scanner.schemas.serialization import save_result


class SyntheticSemantic(SemanticSurfaceDetector):
    device = "cpu-synthetic"
    def __init__(self, maps): self.maps = maps
    def predict(self, image): return {key:value.copy() for key,value in self.maps.items()}


def main() -> int:
    output=Path("outputs/sample_openings")
    if output.exists(): shutil.rmtree(output)
    evidence=output/"lidar/opening_frames"; evidence.mkdir(parents=True,exist_ok=True)
    image=evidence/"frame.jpg"; depth=evidence/"depth.npy"
    Image.new("RGB",(400,300),(150,150,150)).save(image); np.save(depth,np.full((300,400),2,dtype=np.float32))
    door=np.zeros((300,400),np.float32); door[155:261,175:222]=.96
    window=np.zeros_like(door); window[154:217,250:323]=.94
    pose=np.eye(4); pose[:3,:3]=np.array([[1,0,0],[0,0,1],[0,-1,0]]); pose[:3,3]=[0,-2,2.2]
    frame=OpeningFrame(frame_id="synthetic:0",image_path=image,depth_path=depth,
        intrinsics={"width":400,"height":300,"fx":100,"fy":100,"cx":100,"cy":150},
        camera_to_property=pose.tolist(),room_id="room_01",depth_source="sensor")
    write_json(evidence/"frames.json",{"frames":[{**frame.model_dump(mode="json"),
        "image_path":"lidar/opening_frames/frame.jpg","depth_path":"lidar/opening_frames/depth.npy"}]})
    vertices=[Point2D(x=x,y=y) for x,y in ((0.,0.),(5.,0.),(5.,3.),(0.,3.))]
    polygon=Polygon2D(points=vertices)
    walls=[Wall(wall_id=f"wall_{i+1}",room_id="room_01",start_point=start,
        end_point=vertices[(i+1)%4],length=LengthMeasurement(value=float(np.hypot(
        vertices[(i+1)%4].x-start.x,vertices[(i+1)%4].y-start.y)))) for i,start in enumerate(vertices)]
    room=Room(room_id="room_01",name="Synthetic Room",polygon=polygon,walls=walls,
        floor=FloorSurface(surface_id="floor_01",polygon=polygon,area=AreaMeasurement(value=15)),
        ceiling=CeilingSurface(surface_id="ceiling_01",polygon=polygon,height=LengthMeasurement(value=2.8)))
    config=OpeningConfig(candidate_min_pixel_area=20,minimum_3d_points=10,depth_sample_stride=2,
        minimum_supporting_views=1,single_view_min_quality=.5)
    detected=OpeningDetector(config,SyntheticSemantic({"door":door,"window":window,"wall":np.ones_like(door)})).detect(
        PropertyGeometry(property_id="synthetic_property",rooms=[room]),[frame],diagnostics_dir=output/"diagnostics/openings")
    scan=PropertyScanResult(capture=CaptureMetadata(capture_id=UUID("00000000-0000-4000-8000-000000000009"),
        tier="lidar",source_type="directory",source_reference="synthetic_opening_fixture"),property=detected.property,
        warnings=detected.warnings,processing_info=ProcessingInfo(pipeline_version="synthetic-openings-1.0",
        modules_used=["opening_detection","floorplan_renderer"],metadata={"openings":detected.diagnostics.model_dump(mode="json")}),
        metadata={"synthetic":True,"benchmark_accuracy":False})
    drawing=FloorPlanRenderer().render_property(scan.property)
    try: drawing.save(output)
    finally: drawing.close()
    write_json(output/"openings/openings.json",[item.model_dump(mode="json") for item in scan.property.openings])
    write_json(output/"openings/candidates.json",[item.model_dump(mode="json") for item in detected.candidates])
    write_json(output/"openings/detection_summary.json",detected.diagnostics.model_dump(mode="json"))
    write_json(output/"openings/wall_occupancy.json",{"structural_point_count":0,
        "note":"Synthetic smoke uses semantic plus metric depth evidence; occupancy fallback is unit-tested separately."})
    diagnostic=FloorPlanRenderer(RenderingConfig(show_opening_labels=True)).render_property(scan.property)
    try: diagnostic.save_png(output/"diagnostics/openings/fused_openings.png")
    finally: diagnostic.close()
    save_result(scan,output/"result.json")
    for opening in scan.property.openings:
        print(f"{opening.type.value}: width={opening.width.value:.2f} m, height={opening.height.value:.2f} m"
              + (f", sill={opening.sill_height.value:.2f} m" if opening.sill_height else ""))
    print(f"PNG: {output/'floorplan.png'}\nJSON: {output/'result.json'}")
    return 0


if __name__=="__main__": raise SystemExit(main())
