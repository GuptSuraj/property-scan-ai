"""XYZ-only loading, axis normalization, filtering, and normal estimation."""
import logging
from pathlib import Path
import numpy as np
import open3d as o3d
from property_scanner.geometry.config import GeometryConfig
from property_scanner.geometry.exceptions import PointCloudLoadError, InsufficientGeometryError

logger = logging.getLogger(__name__)


def load_cloud(path: Path, config: GeometryConfig) -> o3d.geometry.PointCloud:
    path = Path(path).expanduser()
    if not path.is_file():
        raise PointCloudLoadError(f"Point-cloud file does not exist: {path}")
    if path.suffix.lower() not in {".ply", ".pcd"}:
        raise PointCloudLoadError("Supported point-cloud formats are .ply and .pcd")
    try:
        cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=False, remove_infinite_points=False)
    except (RuntimeError, OSError) as exc:
        raise PointCloudLoadError(f"Cannot load point cloud: {path}") from exc
    points = np.asarray(cloud.points)
    if not len(points):
        raise PointCloudLoadError(f"Point cloud is empty or unreadable: {path}")
    if not np.isfinite(points).all():
        raise PointCloudLoadError("Point cloud contains NaN or infinite coordinates")
    if len(points) < config.min_points:
        raise InsufficientGeometryError(f"Need at least {config.min_points} points; found {len(points)}")
    return cloud


def axis_rotation(up_axis: str) -> np.ndarray:
    """Right-handed rotation mapping the configured signed axis to +Z."""
    up = np.zeros(3)
    up["xyz".index(up_axis[-1])] = -1 if up_axis.startswith("-") else 1
    reference = np.array([1., 0., 0.]) if up_axis[-1] != "x" else np.array([0., 1., 0.])
    y = np.cross(up, reference)
    return np.vstack([reference, y, up])


def preprocess(cloud: o3d.geometry.PointCloud, config: GeometryConfig) -> tuple[o3d.geometry.PointCloud, int, int]:
    original = len(cloud.points)
    cloud = cloud.voxel_down_sample(config.voxel_size)
    downsampled = len(cloud.points)
    if downsampled > config.max_downsampled_points:
        raise InsufficientGeometryError("Downsampled cloud exceeds point budget; increase voxel_size")
    cloud.rotate(axis_rotation(config.up_axis), center=(0, 0, 0))
    if config.remove_outliers and downsampled > config.statistical_nb_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(config.statistical_nb_neighbors, config.statistical_std_ratio)
    if len(cloud.points) < config.min_points:
        raise InsufficientGeometryError("Too few points remain after preprocessing")
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=config.normal_radius, max_nn=config.normal_max_neighbors))
    cloud.normalize_normals()
    cloud.orient_normals_to_align_with_direction((0, 0, 1))
    logger.info("Original points: %s; downsampled: %s; after filtering: %s", original, downsampled, len(cloud.points))
    return cloud, original, downsampled
