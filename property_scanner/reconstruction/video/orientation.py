"""Camera-up prior plus the existing structural floor selector; no SfM-axis assumption."""
import numpy as np
import open3d as o3d
from property_scanner.core.exceptions import ProcessingError
from property_scanner.geometry.preprocessing import preprocess
from property_scanner.geometry.planes import detect_planes, choose_floor, floor_transform
from .models import VideoConfig


def canonicalize(cloud: o3d.geometry.PointCloud, poses: list[np.ndarray], config: VideoConfig) -> tuple[o3d.geometry.PointCloud,np.ndarray]:
    # Upright decoded images supply only a gravity-sign/orientation prior. A real
    # large, low structural plane must independently support this hypothesis.
    ups=np.array([-p[:3,1] for p in poses])
    up=ups.mean(axis=0)
    if np.linalg.norm(up)<config.orientation_min_up_coherence:
        raise ProcessingError('ORIENTATION_UNRESOLVED: camera-up directions inconsistent; capture upright with limited roll')
    up/=np.linalg.norm(up)
    reference=np.eye(3)[np.argmin(np.abs(up))]
    x=reference-up*(reference@up); x/=np.linalg.norm(x)
    initial=np.eye(4); initial[:3,:3]=np.vstack([x,np.cross(up,x),up])
    aligned=o3d.geometry.PointCloud(cloud).transform(initial)
    cleaned,_,_=preprocess(aligned,config.registration.geometry)
    floor=choose_floor(detect_planes(cleaned,config.registration.geometry),np.asarray(cleaned.points),config.registration.geometry)
    transform=floor_transform(floor)@initial
    return o3d.geometry.PointCloud(cloud).transform(transform), transform
