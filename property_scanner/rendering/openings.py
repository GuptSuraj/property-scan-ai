"""Render supplied opening intervals; never infer widths or door swings."""
from dataclasses import dataclass
import numpy as np
from matplotlib.axes import Axes
from property_scanner.schemas.geometry import PropertyGeometry, Opening, OpeningType
from property_scanner.schemas.result import ResultWarning
from property_scanner.rendering.scene import RenderScene
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.rendering.exceptions import InvalidGeometryError


@dataclass
class PositionedOpening:
    opening: Opening
    start: np.ndarray
    end: np.ndarray


def locate_openings(geometry: PropertyGeometry, scene: RenderScene,
                    config: RenderingConfig) -> tuple[list[PositionedOpening], list[ResultWarning]]:
    walls = {wall.wall_id: (wall, a, b) for room in scene.rooms for wall, a, b in room.walls}
    positions, warnings = [], []
    for opening in geometry.openings:
        if opening.wall_id is None or opening.width is None or opening.position_along_wall is None:
            warnings.append(ResultWarning(code="OPENING_NOT_POSITIONED", message=f"Opening {opening.opening_id} omitted: host wall, width, or position is unavailable."))
            continue
        wall_entry = walls.get(opening.wall_id)
        if wall_entry is None:
            warnings.append(ResultWarning(code="OPENING_NOT_POSITIONED", message=f"Opening {opening.opening_id} omitted: wall {opening.wall_id} not found in scene."))
            continue
        wall, a, b = wall_entry
        length = float(np.linalg.norm(b-a))
        offset, width = opening.position_along_wall.value, opening.width.value
        if width <= 0 or offset+width > length+1e-8:
            raise InvalidGeometryError(f"Opening {opening.opening_id} extends beyond its wall or has zero width")
        direction = (b-a)/length
        positions.append(PositionedOpening(opening, a+direction*offset, a+direction*(offset+width)))
    return positions, warnings


def wall_intervals(wall, a: np.ndarray, b: np.ndarray, openings: list[PositionedOpening], config: RenderingConfig) -> list[tuple[float, float]]:
    """Project known gaps onto explicitly linked or coincident shared-room walls."""
    direction = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-direction[1], direction[0]])
    intervals = []
    length = float(np.linalg.norm(b-a))
    for positioned in openings:
        opening = positioned.opening
        if not (opening.wall_id == wall.wall_id or opening.opening_id in wall.opening_ids or wall.room_id in opening.room_ids):
            continue
        if max(abs((positioned.start-a)@normal), abs((positioned.end-a)@normal)) > config.geometry_tolerance:
            continue
        lo, hi = sorted([float((positioned.start-a)@direction), float((positioned.end-a)@direction)])
        if lo >= -config.geometry_tolerance and hi <= length+config.geometry_tolerance:
            intervals.append((max(0., lo), min(length, hi)))
    intervals.sort()
    for first, second in zip(intervals, intervals[1:]):
        if second[0] < first[1]-1e-8:
            raise InvalidGeometryError(f"Overlapping opening intervals on wall {wall.wall_id}")
    return intervals


def draw_opening(ax: Axes, item: PositionedOpening, config: RenderingConfig) -> None:
    a, b = item.start, item.end
    direction = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-direction[1], direction[0]])*config.symbol_size
    style = dict(color=config.opening_color, linewidth=config.dimension_line_width+0.3, zorder=6)
    kind = item.opening.type
    # Jambs are common to all known intervals. Door remains an unadorned gap.
    for end in (a, b):
        ax.plot([end[0]-normal[0], end[0]+normal[0]], [end[1]-normal[1], end[1]+normal[1]], **style)
    if kind == OpeningType.WINDOW:
        for side in (-0.5, 0.5):
            p, q = a+normal*side, b+normal*side
            ax.plot([p[0], q[0]], [p[1], q[1]], **style)
    elif kind in (OpeningType.OPEN_PASSAGE, OpeningType.UNKNOWN):
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle="--" if kind == OpeningType.OPEN_PASSAGE else ":", **style)
        if kind == OpeningType.UNKNOWN:
            ax.text(*((a+b)/2), "?", ha="center", va="center", fontsize=config.font_size,
                    color=config.opening_color, parse_math=False, zorder=7)
    if config.show_opening_labels and item.opening.width is not None:
        labels = {OpeningType.DOOR: "Door", OpeningType.WINDOW: "Window",
                  OpeningType.OPEN_PASSAGE: "Passage", OpeningType.UNKNOWN: "Opening"}
        midpoint = (a+b)/2 + normal*2.2
        ax.text(*midpoint, f"{labels[kind]}\n{item.opening.width.value:.{config.display_precision}f} m",
                ha="center", va="center", fontsize=config.font_size, color=config.opening_color,
                parse_math=False, zorder=7)
