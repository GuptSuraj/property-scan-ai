"""Internal reconstruction evidence, never ground-truth accuracy claims."""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from property_scanner.reconstruction.lidar.poses import rotation_degrees


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def save_poses(path, frames, poses) -> None:
    write_json(path, {"pose_convention": "camera_to_world", "coordinate_system": "canonical_z_up",
                      "poses": [{"frame_id": f.frame_id, "matrix": p.tolist()} for f, p in zip(frames, poses, strict=True)]})


def comparison_metrics(frames, poses, records, config) -> dict:
    lookup = {frame.frame_id: i for i, frame in enumerate(frames)}
    edges = []
    for record in records:
        i, j = lookup[record.source_frame], lookup[record.target_frame]
        relative = np.linalg.inv(poses[j]) @ poses[i]
        fitted = o3d.pipelines.registration.evaluate_registration(frames[i].cloud, frames[j].cloud, config.icp_max_correspondence_fine, relative)
        target = np.array(record.optimized_transform)
        before = np.linalg.inv(target) @ np.array(record.initial_transform)
        after = np.linalg.inv(target) @ relative
        edges.append({"source_frame": record.source_frame, "target_frame": record.target_frame, "kind": record.kind,
            "accepted": record.accepted, "fitness_before": record.fitness_before, "fitness_after": fitted.fitness,
            "rmse_before": record.rmse_before, "rmse_after": fitted.inlier_rmse if fitted.fitness else None,
            "constraint_translation_residual_before": float(np.linalg.norm(before[:3, 3])),
            "constraint_translation_residual_after": float(np.linalg.norm(after[:3, 3])),
            "constraint_rotation_residual_before_degrees": rotation_degrees(before[:3, :3]),
            "constraint_rotation_residual_after_degrees": rotation_degrees(after[:3, :3])})
    corrections = [{"frame_id": f.frame_id, "translation_m": float(np.linalg.norm(p[:3, 3]-f.pose[:3, 3])),
                    "rotation_degrees": rotation_degrees(f.pose[:3, :3].T @ p[:3, :3])} for f, p in zip(frames, poses, strict=True)]
    return {"interpretation": "Internal reconstruction metrics, NOT benchmark accuracy; no ground truth used.",
            "edges": edges, "pose_corrections": corrections}


def trajectory_plot(path, frames, poses) -> None:
    figure = Figure(figsize=(6, 6))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    raw = np.array([f.pose[:3, 3] for f in frames])
    ax.plot(raw[:, 0], raw[:, 1], "o-", label="Initial device poses", markersize=3)
    if poses is not None:
        corrected = np.array([p[:3, 3] for p in poses])
        ax.plot(corrected[:, 0], corrected[:, 1], "o-", label="Optimized poses", markersize=3)
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Camera trajectory — internal diagnostic")
    ax.set_aspect("equal")
    ax.legend()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    figure.clear()


def floorplan_comparison(path, raw_path, corrected_path) -> None:
    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    for ax, source, title in zip(figure.subplots(1, 2), (raw_path, corrected_path), ("DRIFT OFF", "DRIFT ON")):
        if source.exists():
            with Image.open(source) as image:
                ax.imshow(np.asarray(image))
        else:
            ax.text(0.5, 0.5, "Plan unavailable", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    figure.clear()
