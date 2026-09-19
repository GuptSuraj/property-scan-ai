"""Optional headless debug export, not the product floor-plan renderer."""
from pathlib import Path
import open3d as o3d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from property_scanner.core.exceptions import ProcessingError
from property_scanner.geometry.models import RoomGeometryResult, DetectedPlane
from property_scanner.geometry.planes import PlaneCandidate
from property_scanner.geometry.config import GeometryConfig


def export_diagnostics(directory: Path, cloud: o3d.geometry.PointCloud, planes: list[PlaneCandidate],
                       models: list[DetectedPlane], result: RoomGeometryResult, config: GeometryConfig) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def write_cloud(name, data):
        if not o3d.io.write_point_cloud(str(directory / name), data):
            raise ProcessingError(f"Cannot write geometry diagnostic cloud: {name}")
    write_cloud("cleaned.ply", cloud)
    for plane, model in zip(planes, models):
        write_cloud(f"{model.plane_id}_inliers.ply", cloud.select_by_index(plane.indices.tolist()))
    (directory / "geometry.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "config.json").write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    figure = Figure(figsize=(7, 7))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    if result.room_polygon:
        points = result.room_polygon.points
        axes.fill([p.x for p in points], [p.y for p in points], alpha=0.15)
    for segment in result.wall_segments_2d:
        a, b = segment.start, segment.end
        axes.plot([a.x, b.x], [a.y, b.y], "b-")
        axes.text((a.x+b.x)/2, (a.y+b.y)/2, f"{segment.length.value:.2f} m", fontsize=8)
    for corner in result.corners:
        axes.plot(corner.point.x, corner.point.y, "ro")
    axes.set(xlabel="X (m)", ylabel="Y (m)", title="Geometry diagnostics — not a final floor plan")
    axes.set_aspect("equal")
    axes.grid(alpha=0.2)
    figure.savefig(directory / "geometry.png", dpi=140)
    figure.clear()
