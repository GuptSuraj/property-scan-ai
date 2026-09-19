"""Headless Matplotlib scene drawing; all physical measurements come from input."""
import math
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from property_scanner.schemas.geometry import PropertyGeometry, OpeningType
from property_scanner.schemas.result import ResultWarning
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.rendering.scene import build_scene
from property_scanner.rendering.openings import locate_openings, wall_intervals, draw_opening
from property_scanner.rendering.dimensions import draw_dimension, outward_normal, measurement_label
from property_scanner.rendering.exceptions import InvalidGeometryError


def draw_plan(geometry: PropertyGeometry, config: RenderingConfig, title: str | None,
              north_angle_degrees: float | None) -> tuple[Figure, list[ResultWarning]]:
    scene = build_scene(geometry, config)
    openings, warnings = locate_openings(geometry, scene, config)
    margin = config.dimension_offset*(config.dimension_lanes+2)+config.layout_padding
    width, height = scene.width+2*margin, scene.height+2*margin
    size = (width*config.inches_per_meter, height*config.inches_per_meter)
    if max(size) > config.max_figure_inches:
        raise InvalidGeometryError("Plan exceeds configured canvas size; reduce inches_per_meter")
    figure = Figure(figsize=size, dpi=config.dpi, facecolor="white" if config.background == "white" else "none")
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-margin, scene.width+margin), ylim=(-margin, scene.height+margin))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    occupied = []
    dimension_requests = []
    seen_dimensions = set()
    for index, positioned in enumerate(scene.rooms):
        room, polygon = positioned.room, positioned.polygon
        x, y = polygon.exterior.xy
        patch = ax.fill(x, y, facecolor=config.fill_color, edgecolor="none", zorder=0)[0]
        patch.set_gid(f"room-{room.room_id}")
        for wall, a, b in positioned.walls:
            direction = (b-a)/np.linalg.norm(b-a)
            cursor = 0.
            intervals = wall_intervals(wall, a, b, openings, config)
            for lo, hi in [*intervals, (float(np.linalg.norm(b-a)), float(np.linalg.norm(b-a)))]:
                if lo > cursor:
                    p, q = a+direction*cursor, a+direction*lo
                    line = ax.plot([p[0], q[0]], [p[1], q[1]], color=config.wall_color,
                                   linewidth=config.wall_line_width, solid_capstyle="butt", zorder=3)[0]
                    line.set_gid(f"wall-{wall.wall_id}-{cursor:g}")
                cursor = hi
            key = tuple(sorted((tuple(np.round(a, 6)), tuple(np.round(b, 6)))))
            if config.show_wall_dimensions and wall.length is not None and key not in seen_dimensions:
                seen_dimensions.add(key)
                dimension_requests.append((a, b, outward_normal(a, b, polygon), measurement_label(wall.length, config)))
        anchor = polygon.centroid
        if not polygon.contains(anchor):
            anchor = polygon.representative_point()
        lines = [room.name or f"Room {index+1}"]
        if room.floor and room.floor.area:
            lines.append(measurement_label(room.floor.area, config, "m²"))
        if room.ceiling and room.ceiling.height:
            lines.append("Ceiling: " + measurement_label(room.ceiling.height, config))
        elif config.show_missing_ceiling:
            lines.append("Ceiling: unavailable")
        text = ax.text(anchor.x, anchor.y, "\n".join(lines), ha="center", va="center",
                       fontsize=config.room_label_font_size, color=config.text_color,
                       linespacing=1.6, parse_math=False, zorder=8)
        text.set_gid(f"label-{room.room_id}")
        occupied.append(text)
    for opening in openings:
        draw_opening(ax, opening, config)
    if title:
        ax.text(scene.width/2, scene.height+margin-config.layout_padding/2, title,
                ha="center", va="top", fontsize=config.title_font_size, color=config.text_color, parse_math=False)
    if config.show_overall_dimensions:
        offset = config.dimension_offset*(config.dimension_lanes+1)
        for a, b, normal, label in [
            (np.array([0., 0.]), np.array([scene.width, 0.]), np.array([0., -1.]), f"Overall width: {scene.width:.{config.display_precision}f} m"),
            (np.array([scene.width, 0.]), np.array([scene.width, scene.height]), np.array([1., 0.]), f"Overall height: {scene.height:.{config.display_precision}f} m"),
        ]:
            occupied.append(draw_dimension(ax, a, b, normal, offset, label, config).text)
    # Use actual text bounds to stagger nearby dimensions instead of changing geometry.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer).expanded(1.08, 1.12) for text in occupied]
    for a, b, normal, label in dimension_requests:
        selected = None
        for lane in range(1, config.dimension_lanes+1):
            candidate = draw_dimension(ax, a, b, normal, config.dimension_offset*lane, label, config)
            box = candidate.text.get_window_extent(renderer).expanded(1.08, 1.12)
            if not any(box.overlaps(existing) for existing in boxes):
                selected = candidate
                boxes.append(box)
                break
            candidate.remove()
        if selected is None:
            warnings.append(ResultWarning(code="DIMENSION_LABEL_OMITTED", message="A crowded wall dimension was omitted; increase scale, offset, or dimension_lanes."))
    if config.show_scale_bar:
        target = scene.width/3
        power = 10**math.floor(math.log10(target))
        length = max(n*power for n in (0.1, 0.2, 0.5, 1, 2, 5) if n*power <= target)
        base = np.array([0., -margin+config.layout_padding/2])
        ax.plot([base[0], base[0]+length], [base[1], base[1]], color=config.wall_color, linewidth=1.3, gid="scale-bar")
        for value in (0., length/2, length):
            ax.plot([value, value], [base[1]-config.symbol_size/2, base[1]+config.symbol_size/2], color=config.wall_color, linewidth=1)
            ax.text(value, base[1]-config.symbol_size*2, f"{value:g}" + (" m" if value == length else ""),
                    ha="center", va="top", fontsize=config.font_size, color=config.text_color)
    angle = north_angle_degrees if north_angle_degrees is not None else geometry.metadata.get("north_angle_degrees")
    if config.show_north_arrow and angle is not None:
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle) or not 0 <= angle < 360:
            raise InvalidGeometryError("north_angle_degrees must be a supplied finite angle in [0, 360), counterclockwise from +X")
        base = np.array([scene.width+margin*0.65, scene.height+margin*0.35])
        delta = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])*config.layout_padding
        ax.annotate("N", xy=base+delta, xytext=base, ha="center", va="center", fontsize=config.font_size,
                    color=config.text_color, arrowprops=dict(arrowstyle="->", color=config.text_color))
    if config.show_legend:
        handles = [Line2D([], [], color=config.wall_color, linewidth=config.wall_line_width, label="Wall")]
        for kind in sorted({o.opening.type for o in openings}, key=str):
            # Gaps stay unfilled; labels explain their markers without assigning a swing.
            styles = {OpeningType.DOOR: ("None", "|", "Door gap"), OpeningType.WINDOW: ("-", "|", "Window (double line)"),
                      OpeningType.OPEN_PASSAGE: ("--", None, "Open passage"), OpeningType.UNKNOWN: (":", None, "Unknown opening (?)")}
            line, marker, label = styles[kind]
            handles.append(Line2D([], [], color=config.opening_color, linestyle=line, marker=marker, label=label))
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.012), frameon=False,
                  fontsize=config.font_size, labelcolor=config.text_color)
    return figure, warnings
