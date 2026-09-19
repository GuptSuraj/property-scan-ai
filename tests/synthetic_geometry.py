"""Deterministic synthetic algorithm fixtures, never real scans or benchmarks."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon
from shapely import contains_xy


def room_points(*, shape="rectangle", ceiling=True, outliers=0, seed=19, noise=0.004):
    rng = np.random.default_rng(seed)
    vertices = np.array([(0, 0), (4, 0), (4, 5), (0, 5)] if shape == "rectangle" else
                        [(0, 0), (5, 0), (5, 2), (3, 2), (3, 5), (0, 5)], dtype=float)
    polygon = Polygon(vertices)
    xy = rng.uniform(vertices.min(axis=0), vertices.max(axis=0), (5000, 2))
    xy = xy[contains_xy(polygon, xy[:, 0], xy[:, 1])][:2500]
    surfaces = [np.column_stack([xy, np.zeros(len(xy))])]
    if ceiling:
        surfaces.append(np.column_stack([xy, np.full(len(xy), 2.8)]))
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        count = int(np.linalg.norm(end-start)*400)
        positions = start + rng.uniform(0, 1, (count, 1))*(end-start)
        surfaces.append(np.column_stack([positions, rng.uniform(0, 2.8, count)]))
    points = np.concatenate(surfaces)
    points += rng.normal(0, noise, points.shape)
    if outliers:
        points = np.concatenate([points, rng.uniform([-1, -1, -1], [6, 6, 4], (outliers, 3))])
    return points


def write_cloud(path: Path, points: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    assert o3d.io.write_point_cloud(str(path), cloud)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("inputs/synthetic_room.ply"))
    parser.add_argument("--shape", choices=["rectangle", "l"], default="rectangle")
    args = parser.parse_args()
    print(f"SYNTHETIC test fixture: {write_cloud(args.output, room_points(shape=args.shape))}")
