"""CPU COLMAP boundary; only normalized pinhole cameras and optical c2w poses escape."""
from abc import ABC, abstractmethod
from pathlib import Path
import os
import shutil
import sqlite3
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from property_scanner.core.exceptions import ProcessingError
from property_scanner.reconstruction.lidar.models import CameraIntrinsics
from property_scanner.reconstruction.lidar.poses import normalize_pose
from property_scanner.reconstruction.lidar.diagnostics import write_json
from .frames import executable
from .models import SparseReconstruction, SparsePoint, RegisteredFrame, Observation


class SfMReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, images: Path, output: Path, config: object) -> SparseReconstruction:
        """Return arbitrary-scale world coordinates, never implicitly metric coordinates."""


def parse_model(directory: Path) -> SparseReconstruction:
    """Parse COLMAP's documented text format, including empty observation lines."""
    try:
        cameras = {}
        for line in (directory/'cameras.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'): continue
            fields = line.split()
            if fields[1] != 'PINHOLE' or len(fields) != 8:
                raise ValueError('Expected undistorted PINHOLE cameras')
            cameras[int(fields[0])] = CameraIntrinsics(width=int(fields[2]), height=int(fields[3]), **dict(zip(('fx','fy','cx','cy'), map(float, fields[4:]))))
        points = []
        for line in (directory/'points3D.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'): continue
            f = line.split()
            points.append(SparsePoint(point_id=int(f[0]), xyz=tuple(map(float, f[1:4])), reprojection_error=float(f[7])))
        lines = iter((directory/'images.txt').read_text().splitlines())
        frames = []
        for line in lines:
            if not line.strip() or line.startswith('#'): continue
            f = line.split(maxsplit=9)
            name = f[9]
            if Path(name).name != name: raise ValueError('Unsafe image filename in sparse model')
            q = np.array(list(map(float, f[1:5])))
            if not np.isclose(np.linalg.norm(q), 1, atol=1e-3): raise ValueError('Invalid camera quaternion')
            w2c = np.eye(4)
            w2c[:3,:3] = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            w2c[:3,3] = list(map(float, f[5:8]))
            values = next(lines).split()
            if len(values) % 3: raise ValueError('Malformed sparse observations')
            obs = [Observation(x=float(values[i]), y=float(values[i+1]), point_id=int(values[i+2])) for i in range(0,len(values),3) if int(values[i+2]) >= 0]
            frames.append(RegisteredFrame(image_id=int(f[0]), filename=name, intrinsics=cameras[int(f[8])], camera_to_world=normalize_pose(w2c, 'world_to_camera', np.eye(4)).tolist(), observations=obs))
        return SparseReconstruction(frames=sorted(frames,key=lambda f:f.filename), points=points, statistics={'registered_frames':len(frames),'sparse_points':len(points)})
    except (OSError, ValueError, KeyError, IndexError, StopIteration) as exc:
        raise ProcessingError(f'Cannot parse COLMAP reconstruction: {exc}') from exc


class ColmapReconstructor(SfMReconstructor):
    def reconstruct(self, images: Path, output: Path, config: object) -> SparseReconstruction:
        binary = executable('colmap')
        output.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'QT_QPA_PLATFORM':'offscreen'}
        def help_text(command: str) -> str:
            p = subprocess.run([binary,command,'-h'],capture_output=True,text=True,timeout=60,env=env)
            if p.returncode:
                raise ProcessingError(f'Cannot inspect COLMAP {command} options')
            return p.stdout+p.stderr
        def run(command: str, arguments: list[str]) -> None:
            with (output/f'{command}.log').open('a') as log:
                try:
                    p = subprocess.run([binary,command,*arguments],stdout=log,stderr=log,timeout=config.command_timeout,env=env)
                except subprocess.TimeoutExpired as exc:
                    raise ProcessingError(f'COLMAP {command} timed out; inspect video/sfm/{command}.log') from exc
            if p.returncode: raise ProcessingError(f'COLMAP {command} failed; inspect video/sfm/{command}.log')
        extraction = help_text('feature_extractor')
        matching_strategy = getattr(config, 'colmap_matching_strategy', 'sequential')
        matcher = 'exhaustive_matcher' if matching_strategy == 'exhaustive' else 'sequential_matcher'
        matching = help_text(matcher)
        ep = 'FeatureExtraction' if 'FeatureExtraction.use_gpu' in extraction else 'SiftExtraction'
        mp = 'FeatureMatching' if 'FeatureMatching.use_gpu' in matching else 'SiftMatching'
        database = str(output/'database.db')
        database_path = Path(database)
        database_path.unlink(missing_ok=True)
        sparse=output/'sparse'
        if sparse.exists():
            shutil.rmtree(sparse)
        args = ['--default_random_seed','0','--database_path',database,'--image_path',str(images),
                '--ImageReader.single_camera','1' if getattr(config, 'single_camera', True) else '0',
                '--ImageReader.camera_model','PINHOLE',f'--{ep}.use_gpu','0',f'--{ep}.num_threads',str(config.num_threads),
                '--SiftExtraction.max_num_features',str(config.colmap_max_features)]
        if config.intrinsics:
            k=config.intrinsics
            first = next(images.glob('*.jpg'), None)
            if first is None:
                raise ProcessingError('No keyframes supplied to COLMAP')
            with Image.open(first) as image: width,height=image.size
            sx,sy=width/k.width,height/k.height
            if not np.isclose(sx, sy, rtol=1e-3):
                raise ProcessingError('Configured intrinsics cannot be anisotropically scaled to extracted frames')
            args += ['--ImageReader.camera_params',','.join(map(str,(k.fx*sx,k.fy*sy,(k.cx+0.5)*sx-0.5,(k.cy+0.5)*sy-0.5)))]
        run('feature_extractor',args)
        match_args=['--default_random_seed','0','--database_path',database,f'--{mp}.use_gpu','0',f'--{mp}.num_threads',str(config.num_threads)]
        if matching_strategy == 'sequential':
            match_args += ['--SequentialMatching.overlap',str(config.sequential_overlap),'--SequentialMatching.loop_detection','0']
        run(matcher,match_args)
        match_statistics = _database_statistics(database_path)
        write_json(output/'feature_matching.json',match_statistics)
        sparse.mkdir(exist_ok=True)
        mapper=['--default_random_seed','0','--database_path',database,'--image_path',str(images),'--output_path',str(sparse),
                '--Mapper.num_threads',str(config.num_threads),'--Mapper.random_seed','0',
                '--Mapper.min_model_size',str(config.min_registered_frames)]
        if config.intrinsics: mapper += ['--Mapper.ba_refine_focal_length','0','--Mapper.ba_refine_extra_params','0']
        run('mapper',mapper)
        models=[]
        for folder in sorted(sparse.iterdir()):
            if not folder.is_dir(): continue
            run('model_converter',['--input_path',str(folder),'--output_path',str(folder),'--output_type','TXT'])
            models.append(parse_model(folder))
        if not models: raise ProcessingError('COLMAP produced no registered reconstruction; improve overlap, texture and translation')
        result=max(models,key=lambda m:len(m.frames))
        result.statistics.update(match_statistics)
        result.statistics.update(components=len(models), matching_strategy=matching_strategy,
            camera_calibration='supplied' if config.intrinsics else 'COLMAP estimated',
            pose_convention='camera_to_world', coordinate_system='SfM arbitrary orientation and scale')
        write_json(output/'reconstruction.json',result.model_dump())
        write_json(output/'sfm_summary.json',result.statistics)
        if len(result.frames)<config.min_registered_frames: raise ProcessingError('Too few COLMAP registered frames; relative artifacts retained')
        return result


def _database_statistics(database: Path) -> dict:
    """Read stable aggregate matching evidence from COLMAP's SQLite database."""
    with sqlite3.connect(database) as connection:
        images = dict(connection.execute("SELECT image_id, name FROM images"))
        features = {images[image_id]: rows for image_id, rows in
                    connection.execute("SELECT image_id, rows FROM keypoints") if image_id in images}
        matched_pairs, inliers = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(rows), 0) FROM two_view_geometries WHERE rows > 0"
        ).fetchone()
    return {"features_per_image": features, "matched_image_pairs": matched_pairs,
            "inlier_matches": inliers}
