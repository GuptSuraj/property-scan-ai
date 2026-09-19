"""Draw supplied measurements; calculate only graphic annotation placement."""
from dataclasses import dataclass
import numpy as np
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.text import Text
from shapely.geometry import Point, Polygon
from property_scanner.schemas.measurements import MeasuredValue
from property_scanner.rendering.styles import RenderingConfig


def measurement_label(value: MeasuredValue, config: RenderingConfig, suffix: str = "m") -> str:
    text = f"{value.value:.{config.display_precision}f} {suffix}"
    if config.show_confidence and value.lower_bound is not None and value.upper_bound is not None:
        prefix = f"{value.confidence_level:.0%} CI" if value.confidence_level is not None else "Bounds"
        text += f"\n{prefix}: {value.lower_bound:.{config.display_precision}f}–{value.upper_bound:.{config.display_precision}f} {suffix}"
    return text


def outward_normal(a: np.ndarray, b: np.ndarray, polygon: Polygon) -> np.ndarray:
    direction = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-direction[1], direction[0]])
    midpoint = (a+b)/2
    if polygon.contains(Point(midpoint + normal*0.01)):
        normal = -normal
    return normal


@dataclass
class DimensionArtists:
    artists: list[Artist]
    text: Text

    def remove(self) -> None:
        for artist in self.artists:
            artist.remove()


def draw_dimension(ax: Axes, a: np.ndarray, b: np.ndarray, normal: np.ndarray,
                   offset: float, label: str, config: RenderingConfig) -> DimensionArtists:
    start, end = a+normal*offset, b+normal*offset
    artists = []
    style = dict(color=config.dimension_color, linewidth=config.dimension_line_width, zorder=4)
    for point, shifted in ((a, start), (b, end)):
        p, q = point+normal*config.symbol_size, shifted+normal*config.symbol_size
        artists.extend(ax.plot([p[0], q[0]], [p[1], q[1]], **style))
        tick = normal*config.dimension_tick_size
        artists.extend(ax.plot([shifted[0]-tick[0], shifted[0]+tick[0]],
                               [shifted[1]-tick[1], shifted[1]+tick[1]], **style))
    artists.extend(ax.plot([start[0], end[0]], [start[1], end[1]], **style))
    center = (start+end)/2+normal*config.dimension_text_gap
    angle = float(np.degrees(np.arctan2(b[1]-a[1], b[0]-a[0])))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    text = ax.text(*center, label, ha="center", va="center", rotation=angle,
                   rotation_mode="anchor", fontsize=config.font_size,
                   color=config.dimension_color, parse_math=False, zorder=5,
                   bbox=dict(facecolor="white" if config.background == "white" else "none", edgecolor="none", pad=1.5))
    artists.append(text)
    return DimensionArtists(artists, text)
