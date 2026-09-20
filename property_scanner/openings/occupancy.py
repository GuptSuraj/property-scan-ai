"""Conservative wall-local occupancy fallback for unknown structural openings."""

import numpy as np
from scipy import ndimage

from property_scanner.openings.models import OpeningCandidate, OpeningConfig
from property_scanner.schemas.geometry import PropertyGeometry


def geometry_opening_candidates(geometry: PropertyGeometry, points: np.ndarray,
                                config: OpeningConfig) -> list[OpeningCandidate]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        return []
    candidates = []
    for room in geometry.rooms:
        ceiling = room.ceiling.height.value if room.ceiling and room.ceiling.height else 3.0
        for wall in room.walls:
            if wall.start_point is None or wall.end_point is None:
                continue
            start = np.array([wall.start_point.x, wall.start_point.y])
            end = np.array([wall.end_point.x, wall.end_point.y])
            length = float(np.linalg.norm(end-start))
            if length < config.door_min_width*2:
                continue
            direction = (end-start)/length
            relative = points[:, :2]-start
            u = relative@direction
            distance = np.abs(relative[:, 0]*direction[1]-relative[:, 1]*direction[0])
            keep = ((u >= 0) & (u <= length) & (points[:, 2] >= 0) & (points[:, 2] <= ceiling)
                    & (distance <= config.wall_association_max_distance))
            if keep.sum() < 100:
                continue
            cell = config.occupancy_cell_size
            u_edges = np.arange(0, length+cell, cell)
            v_edges = np.arange(0, ceiling+cell, cell)
            counts, _, _ = np.histogram2d(u[keep], points[keep, 2], bins=(u_edges, v_edges))
            occupied = counts >= config.occupancy_min_points_per_cell
            # One-cell closing prevents isolated sampling misses from becoming openings.
            occupied = ndimage.binary_closing(occupied, structure=np.ones((3, 3)), border_value=1)
            low = ~occupied
            labels, count = ndimage.label(low)
            for label in range(1, count+1):
                cells = labels == label
                indices = np.argwhere(cells)
                if not len(indices):
                    continue
                u0, v0 = indices.min(axis=0); u1, v1 = indices.max(axis=0)+1
                # Exclude capture boundaries: only a bounded hole has structural support.
                if u0 == 0 or u1 == occupied.shape[0] or v1 == occupied.shape[1]:
                    continue
                width, height = (u1-u0)*cell, (v1-v0)*cell
                if width < config.door_min_width or height < config.opening_min_height:
                    continue
                ring = ndimage.binary_dilation(cells, iterations=1) & ~cells
                support = float(occupied[ring].mean()) if ring.any() else 0
                if support < config.occupancy_min_boundary_support:
                    continue
                floor_contact = v0*cell <= config.floor_contact_tolerance
                kind = "open_passage" if floor_contact and height >= 1.6 else "unknown"
                candidates.append(OpeningCandidate(candidate_id=f"geometry:{wall.wall_id}:{label}",
                    frame_id="geometry", semantic_type=kind, semantic_score=0,
                    wall_id=wall.wall_id, room_id=room.room_id, u_min=float(u0*cell),
                    u_max=float(min(length, u1*cell)), v_min=float(v0*cell),
                    v_max=float(min(ceiling, v1*cell)), sample_count=int(keep.sum()),
                    geometry_score=support, quality=0.55+0.35*support, accepted=True,
                    metadata={"evidence": "wall_local_low_occupancy", "boundary_support": support}))
    return candidates


def export_wall_occupancy_plots(directory, geometry: PropertyGeometry, points: np.ndarray,
                                config: OpeningConfig) -> None:
    """Export wall-local point-density images; this does not affect detection."""
    from pathlib import Path
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    points=np.asarray(points,dtype=float)
    if points.ndim != 2 or points.shape[1] != 3: return
    for room in geometry.rooms:
        ceiling=room.ceiling.height.value if room.ceiling and room.ceiling.height else 3.0
        for wall in room.walls:
            if wall.start_point is None or wall.end_point is None: continue
            start=np.array([wall.start_point.x,wall.start_point.y]); end=np.array([wall.end_point.x,wall.end_point.y])
            length=float(np.linalg.norm(end-start))
            if length <= 0: continue
            direction=(end-start)/length; relative=points[:,:2]-start
            u=relative@direction; distance=np.abs(relative[:,0]*direction[1]-relative[:,1]*direction[0])
            keep=(u>=0)&(u<=length)&(points[:,2]>=0)&(points[:,2]<=ceiling)&(distance<=config.wall_association_max_distance)
            if keep.sum()<20: continue
            figure=Figure(figsize=(8,3)); FigureCanvasAgg(figure); axis=figure.subplots()
            image=axis.hist2d(u[keep],points[keep,2],bins=(max(2,int(length/config.occupancy_cell_size)),
                max(2,int(ceiling/config.occupancy_cell_size))),cmap="Greys")
            axis.set(xlabel="Position along wall (m)",ylabel="Height above floor (m)",title=f"{wall.wall_id} occupancy")
            figure.colorbar(image[3],ax=axis,label="Point count")
            figure.savefig(directory/f"{wall.wall_id.replace(':','_')}.png",dpi=150,bbox_inches="tight"); figure.clear()
