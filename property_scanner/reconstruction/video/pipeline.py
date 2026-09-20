"""Video orchestration; shared fusion, registration, geometry, renderer and result schema."""
from datetime import datetime, timezone
import csv
import logging
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import open3d as o3d
from PIL import Image
from property_scanner.core.exceptions import PropertyScannerError, ProcessingError
from property_scanner.geometry.engine import GeometryEngine
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.schemas.capture import CaptureMetadata
from property_scanner.schemas.common import PreparationResult
from property_scanner.schemas.geometry import PropertyGeometry
from property_scanner.schemas.result import PropertyScanResult, ProcessingInfo, ProcessingIssue, ResultWarning
from property_scanner.schemas.serialization import save_result
from property_scanner.reconstruction.lidar.adapter import RGBDFrame
from property_scanner.reconstruction.lidar.models import CameraIntrinsics, LidarConfig
from property_scanner.reconstruction.lidar.fusion import Keyframe, frame_cloud, fuse
from property_scanner.reconstruction.lidar.registration import optimize
from property_scanner.reconstruction.lidar.diagnostics import write_json, save_poses, trajectory_plot
from .models import VideoConfig, ScaleReport, RegisteredFrame
from .frames import extract_keyframes, executable
from .sfm import SfMReconstructor, ColmapReconstructor
from .depth import MetricDepthEstimator, DepthAnythingMetric
from .scale import estimate_scale, scaled_pose
from .orientation import canonicalize

logger=logging.getLogger(__name__)
VERSION='video-metric-1.0.0'


def _tool_version(name: str) -> str:
    """Return a short reproducibility string without failing a partial result."""
    import subprocess

    try:
        process = subprocess.run(
            [executable(name), "-version" if name.startswith("ff") else "-h"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (process.stdout + process.stderr).splitlines()[0].strip()
    except (OSError, IndexError, PropertyScannerError, subprocess.TimeoutExpired):
        return "unavailable"


def metric_keyframe(frame: RegisteredFrame, image: Image.Image, depth: np.ndarray, scale: float, config: VideoConfig, frame_id: int) -> Keyframe:
    """Bounded optical RGB-D backprojection using calibration scaled with pixel stride."""
    if depth.shape != (frame.intrinsics.height,frame.intrinsics.width) or image.size != (frame.intrinsics.width,frame.intrinsics.height):
        raise ProcessingError('Depth/RGB/calibration dimensions disagree')
    stride=config.point_cloud_pixel_stride
    rgb=np.asarray(image.convert('RGB'))[::stride,::stride].copy()
    z=depth[::stride,::stride].astype(np.float32,copy=True)
    z[~np.isfinite(z)|(z<config.depth_min_m)|(z>config.depth_max_m)]=0
    if np.count_nonzero(z)<config.registration.min_valid_depth_pixels:
        raise ProcessingError('Too few valid metric-depth pixels')
    k=frame.intrinsics
    intrinsics=CameraIntrinsics(
        width=z.shape[1], height=z.shape[0], fx=k.fx/stride, fy=k.fy/stride,
        cx=(k.cx+0.5)/stride-0.5, cy=(k.cy+0.5)/stride-0.5,
    )
    pose=scaled_pose(frame.camera_to_world,scale)
    fusion_config=LidarConfig.model_validate({**config.registration.model_dump(),'depth_min_m':config.depth_min_m,'depth_max_m':config.depth_max_m})
    cloud=frame_cloud(RGBDFrame(frame_id,rgb,z,pose),intrinsics,fusion_config)
    if len(cloud.points)<20: raise ProcessingError('Too few backprojected points')
    return Keyframe(frame_id,cloud,pose)


def process_video(prepared: PreparationResult, config: VideoConfig, model_dir: Path, *, reconstructor: SfMReconstructor | None=None, depth_estimator: MetricDepthEstimator | None=None) -> PropertyScanResult:
    started,timer=datetime.now(timezone.utc),perf_counter()
    output=prepared.output_dir; video=output/'video'; diagnostics=output/'diagnostics'/'video'
    for directory in (video,diagnostics,video/'depth'): directory.mkdir(parents=True,exist_ok=True)
    write_json(output/'processing_config.json',{
        'pipeline_version':VERSION,'python_version':platform.python_version(),'open3d_version':o3d.__version__,
        'ffmpeg_version':_tool_version('ffmpeg'),'colmap_version':_tool_version('colmap'),
        'config':config.model_dump(mode='json')})
    warnings=[]; errors=[]; modules=[]; geometry=None; metadata=None; scale=ScaleReport(); orientation=None
    stage='VIDEO_INVALID'
    try:
        metadata,selected=extract_keyframes(prepared.capture.source_path,video,config)
        modules.append('ffmpeg_keyframes')
        stage='SFM_FAILED'
        logger.info('Starting CPU COLMAP reconstruction')
        sfm=(reconstructor or ColmapReconstructor()).reconstruct(video/'keyframes',video/'sfm',config)
        modules.append('colmap_sfm')
        logger.info('%s/%s frames registered',len(sfm.frames),len(selected))
        if len(sfm.frames)<config.min_registered_frames: raise ProcessingError('Too few registered frames')
        if len(sfm.frames)<len(selected): warnings.append(ResultWarning(code='PARTIAL_SFM',message=f'{len(sfm.frames)}/{len(selected)} keyframes registered; largest connected component used.'))
        relative=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([p.xyz for p in sfm.points]).reshape(-1,3)))
        o3d.io.write_point_cloud(str(video/'sfm'/'sparse_relative.ply'),relative)
        stage='DEPTH_FAILED'
        estimator=depth_estimator or DepthAnythingMetric(model_dir,config)
        logger.info('Estimating metric depth')
        paths={}
        for frame in sfm.frames[::config.depth_frame_stride]:
            try:
                with Image.open(video/'keyframes'/frame.filename) as image:
                    prediction=estimator.predict(image.convert('RGB'))
                prediction=np.asarray(prediction,dtype=np.float32)
                if prediction.shape!=(frame.intrinsics.height,frame.intrinsics.width): raise ProcessingError('Metric depth shape differs from camera image')
                path=video/'depth'/f'{Path(frame.filename).stem}.npy'; np.save(path,prediction); paths[frame.filename]=path
            except (PropertyScannerError, OSError, RuntimeError, ValueError) as exc:
                warnings.append(ResultWarning(code='DEPTH_FRAME_SKIPPED',message=f'{frame.filename}: {exc}'))
        modules.append('metric_depth')
        stage='METRIC_SCALE_UNRESOLVED'
        logger.info('Recovering metric scale')
        scale=estimate_scale(sfm,paths,config)
        write_json(video/'scale_estimation.json',scale.model_dump())
        if not scale.metric_scale_resolved: raise ProcessingError(scale.failure_reason or 'Metric scale unresolved; relative reconstruction retained')
        logger.info('Metric scale resolved: %.6f',scale.global_scale)
        modules.append('robust_metric_scale')
        warnings.append(ResultWarning(code='LEARNED_METRIC_SCALE',message='Metric scale is inferred from monocular learned depth; internal consistency is not measurement accuracy or a calibrated confidence interval.'))
        stage='METRIC_FUSION_FAILED'
        relative.scale(scale.global_scale,center=(0,0,0))
        o3d.io.write_point_cloud(str(video/'sfm'/'sparse_metric.ply'),relative)
        keyframes=[]
        for frame in sfm.frames:
            if frame.filename not in scale.frames_used: continue
            with Image.open(video/'keyframes'/frame.filename) as image:
                keyframes.append(metric_keyframe(frame,image,np.load(paths[frame.filename],allow_pickle=False),
                                                  scale.global_scale,config,int(Path(frame.filename).stem)))
        if len(keyframes)<config.min_registered_frames: raise ProcessingError('Too few scale-consistent frames for fusion')
        initial=[f.pose.copy() for f in keyframes]
        save_poses(video/'initial_poses.json',keyframes,initial,coordinate_system='sfm_metric_arbitrary_orientation')
        poses=initial
        if config.enable_icp_refinement:
            logger.info('Refining trajectory with shared ICP and pose graph')
            registration=config.registration
            if not config.enable_loop_closure:
                registration=registration.model_copy(update={'loop_min_frame_separation':len(sfm.frames)+1})
            poses,records,graph,fallback=optimize(keyframes,registration)
            write_json(diagnostics/'registrations.json',[r.model_dump() for r in records])
            o3d.io.write_pose_graph(str(diagnostics/'pose_graph.json'),graph)
            if fallback: warnings.append(ResultWarning(code='ICP_NEIGHBOR_REJECTED',message=f'{fallback} rejected registrations use weak SfM pose priors.'))
            modules.append('shared_icp_pose_graph')
        save_poses(video/'optimized_poses.json',keyframes,poses,coordinate_system='sfm_metric_arbitrary_orientation')
        trajectory_plot(diagnostics/'trajectory.png',keyframes,poses)
        logger.info('Fusing video reconstruction')
        fuse(keyframes,poses,config.registration,video/'fused_metric_sfm.ply')
        modules.append('shared_metric_fusion')
        stage='ORIENTATION_UNRESOLVED'
        cloud=o3d.io.read_point_cloud(str(video/'fused_metric_sfm.ply'))
        cloud,_=cloud.remove_statistical_outlier(config.registration.statistical_nb_neighbors,config.registration.statistical_std_ratio)
        cloud,transform=canonicalize(cloud,poses,config)
        orientation=transform.tolist()
        write_json(video/'canonical_transform.json',{'sfm_metric_to_canonical':orientation,'method':'upright-camera prior, supported and refined by structural floor plane'})
        if not o3d.io.write_point_cloud(str(video/'fused_video.ply'),cloud): raise ProcessingError('Cannot save canonical point cloud')
        save_poses(video/'canonical_poses.json',keyframes,[transform@p for p in poses])
        stage='GEOMETRY_FAILED'
        logger.info('Running geometry engine')
        geometry=GeometryEngine(config.registration.geometry).process_point_cloud(video/'fused_video.ply',diagnostics_dir=output/'diagnostics'/'geometry')
        modules.append('geometry'); warnings.extend(geometry.warnings)
        write_json(video/'geometry_result.json',geometry.model_dump(mode='json'))
        with (output/'measurements.csv').open('w',newline='') as file:
            writer=csv.writer(file); writer.writerow(['measurement','value','unit'])
            for wall in geometry.wall_segments_2d: writer.writerow([wall.wall_id,wall.length.value,'m'])
            for name,value in [('floor_area',geometry.floor_area),('ceiling_height',geometry.ceiling_height)]:
                if value is not None: writer.writerow([name,value.value,value.unit])
        if geometry.room_polygon is None: raise ProcessingError('No closed room polygon; detected geometry and reconstruction retained')
        stage='RENDERING_FAILED'
        logger.info('Rendering floor plan')
        drawing=FloorPlanRenderer().render_room(geometry)
        try: drawing.save(output); warnings.extend(drawing.warnings)
        finally: drawing.close()
        modules.append('floorplan_renderer')
    except (PropertyScannerError, OSError, RuntimeError, ValueError) as exc:
        errors.append(ProcessingIssue(code=stage,message=str(exc),module='video'))
        logger.warning('%s: %s',stage,exc)
    result=PropertyScanResult(
        capture=CaptureMetadata(capture_id=prepared.capture.capture_id,tier='video',source_type='file',source_reference=prepared.capture.source_path.name,processing_timestamp=started,metadata=metadata.model_dump() if metadata else {}),
        property=PropertyGeometry(property_id=f'property:{prepared.capture.capture_id}',rooms=[geometry.to_room('room_01')] if geometry else [],total_floor_area=geometry.floor_area if geometry else None),
        warnings=warnings,processing_info=ProcessingInfo(pipeline_version=VERSION,started_at=started,completed_at=datetime.now(timezone.utc),processing_seconds=perf_counter()-timer,modules_used=modules,errors=errors,
            model_versions={config.depth_model:config.depth_revision} if 'metric_depth' in modules else {},
            metadata={'metric_scale':scale.model_dump(),'inference_device':getattr(locals().get('estimator'),'device',None),'open3d_version':o3d.__version__,'python_version':platform.python_version()}),
        metadata={'sfm_metric_to_canonical':orientation})
    save_result(result,output/'result.json')
    return result
