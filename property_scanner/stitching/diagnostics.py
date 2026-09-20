"""JSON and lightweight plots for inspection of inferred property layouts."""

from pathlib import Path
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from shapely.geometry import Polygon

from property_scanner.reconstruction.lidar.diagnostics import write_json
from property_scanner.schemas.geometry import Room
from property_scanner.stitching.engine import StitchingResult
from property_scanner.stitching.transforms import se2, transform_room


def room_scale_quality(output_dir: Path, room_ids: list[str], config) -> tuple[dict, list[str]]:
    """Compare dimensionless per-room scale quality; SfM gauge scale values are not comparable."""
    records, inconsistent = {}, []
    for room_id in room_ids:
        path = Path(output_dir) / "photo" / room_id / "scale_estimation.json"
        try:
            import json
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        scale = report.get("global_scale")
        mad = report.get("scale_mad")
        quality = report.get("scale_confidence_quality")
        relative_mad = mad/scale if isinstance(mad, (int, float)) and isinstance(scale, (int, float)) and scale > 0 else None
        weak = ((isinstance(quality, (int, float)) and quality < config.min_room_scale_quality) or
                (relative_mad is not None and relative_mad > config.max_room_scale_relative_mad))
        records[room_id] = {"scale_confidence_quality": quality, "relative_mad": relative_mad,
                            "quality_warning": weak,
                            "note": "Global SfM gauge scale is intentionally not compared between rooms."}
        if weak:
            inconsistent.append(room_id)
    return {"rooms": records, "inconsistent_room_ids": inconsistent}, inconsistent


def _layout_plot(path: Path, rooms: dict[str, Room], transforms, title: str) -> None:
    figure = Figure(figsize=(8, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for room_id, record in transforms.items():
        room = transform_room(rooms[room_id], se2(record.translation_x, record.translation_y, record.rotation_yaw))
        polygon = Polygon([(point.x, point.y) for point in room.polygon.points])
        x, y = polygon.exterior.xy
        axis.plot(x, y, linewidth=2)
        point = polygon.representative_point()
        axis.text(point.x, point.y, room.name or room_id, ha="center", va="center")
    axis.set(title=title, xlabel="X (m)", ylabel="Y (m)")
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.clear()


def _graph_plot(path: Path, result: StitchingResult) -> None:
    figure = Figure(figsize=(7, 5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = {room: np.array([item.translation_x, item.translation_y])
                 for room, item in result.transforms.items()}
    if positions and max(np.linalg.norm(point) for point in positions.values()) < 1e-6:
        count = len(positions)
        positions = {room: np.array([np.cos(2*np.pi*i/max(count, 1)), np.sin(2*np.pi*i/max(count, 1))])
                     for i, room in enumerate(sorted(positions))}
    for edge in result.accepted_evidence:
        if edge.room_a_id in positions and edge.room_b_id in positions:
            a, b = positions[edge.room_a_id], positions[edge.room_b_id]
            axis.plot([a[0], b[0]], [a[1], b[1]], color="#666666", zorder=1)
            midpoint = (a+b)/2
            axis.text(*midpoint, f"{edge.combined_score:.2f}", fontsize=8)
    for room, point in positions.items():
        axis.scatter(*point, s=500, color="#eeeeee", edgecolor="#222222", zorder=2)
        axis.text(*point, room, ha="center", va="center", zorder=3)
    axis.set_title("Accepted inferred room graph")
    axis.set_aspect("equal")
    axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.clear()


def export_stitching_diagnostics(directory: Path, local_rooms: list[Room],
                                 candidates, result: StitchingResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "candidate_connections.json", [item.model_dump(mode="json") for item in candidates])
    write_json(directory / "accepted_connections.json", [item.model_dump(mode="json") for item in result.accepted_evidence])
    write_json(directory / "rejected_connections.json", [item.model_dump(mode="json") for item in result.rejected_evidence])
    write_json(directory / "initial_room_transforms.json",
               [item.model_dump(mode="json") for item in result.initial_transforms.values()])
    write_json(directory / "optimized_room_transforms.json",
               [item.model_dump(mode="json") for item in result.transforms.values()])
    write_json(directory / "layout_metrics.json", result.diagnostics.model_dump(mode="json"))
    write_json(directory / "cycle_consistency.json", {
        "maximum_cycle_residual": result.diagnostics.cycle_residual,
        "threshold_exceeded": any(warning.code == "HIGH_CYCLE_CLOSURE_ERROR" for warning in result.warnings)})
    write_json(directory / "overlap_report.json", {
        "overlap_area_total_m2": result.diagnostics.overlap_area_total,
        "maximum_pair_overlap_m2": result.diagnostics.maximum_pair_overlap,
        "sum_room_area_m2": result.diagnostics.room_area_sum_m2,
        "union_area_m2": result.diagnostics.polygon_union_area_m2})
    write_json(directory / "room_graph.json", {
        "root_room_id": result.diagnostics.root_room_id,
        "components": result.diagnostics.connected_components,
        "edges": [{"connection_id": edge.connection_id, "room_a_id": edge.room_a_id,
                   "room_b_id": edge.room_b_id, "quality": edge.combined_score}
                  for edge in result.accepted_evidence]})
    room_map = {room.room_id: room for room in local_rooms}
    _graph_plot(directory / "room_graph.png", result)
    _layout_plot(directory / "initial_layout.png", room_map, result.initial_transforms, "Initial room layout")
    _layout_plot(directory / "optimized_layout.png", room_map, result.transforms, "Optimized room layout")
