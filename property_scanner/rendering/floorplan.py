"""Reusable typed room/property rendering and PNG/SVG export."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
from pydantic import ValidationError
from property_scanner.pipeline.context import ScanContext
from property_scanner.geometry.models import RoomGeometryResult
from property_scanner.schemas.geometry import Room, Opening, PropertyGeometry
from property_scanner.schemas.result import ResultWarning
from property_scanner.schemas.damage import DamageRegion
from property_scanner.rendering.styles import RenderingConfig
from property_scanner.rendering.exceptions import FloorPlanRenderError, InvalidGeometryError

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class FloorPlanFiles:
    png: Path
    svg: Path


@dataclass
class RenderedFloorPlan:
    """An owned headless figure; call close() after exporting to release artists."""

    figure: "Figure"
    config: RenderingConfig
    warnings: list[ResultWarning]

    def _save(self, path: Path, format: str) -> Path:
        from matplotlib import rc_context
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with rc_context({"svg.fonttype": "none", "svg.hashsalt": "property-scanner", "text.usetex": False}):
                self.figure.savefig(path, format=format, dpi=self.config.dpi,
                    transparent=self.config.background == "transparent", bbox_inches="tight", pad_inches=0.1,
                    metadata={"Date": None, "Creator": "property-scanner"} if format == "svg" else None)
        except (OSError, ValueError, RuntimeError) as exc:
            raise FloorPlanRenderError(f"Cannot export {format.upper()} to {path}: {exc}") from exc
        return path

    def save_png(self, path: Path) -> Path:
        return self._save(path, "png")

    def save_svg(self, path: Path) -> Path:
        return self._save(path, "svg")

    def save(self, output_dir: Path) -> FloorPlanFiles:
        return FloorPlanFiles(self.save_png(Path(output_dir)/"floorplan.png"), self.save_svg(Path(output_dir)/"floorplan.svg"))

    def close(self) -> None:
        self.figure.clear()


class FloorPlanRenderer:
    """Draw supplied geometry; scientific rendering dependencies are loaded on demand."""

    def __init__(self, config: RenderingConfig | None = None) -> None:
        self.config = config or RenderingConfig()

    def render_room(self, geometry: RoomGeometryResult | Room, *, room_id: str = "room_01",
                    name: str | None = None, openings: Sequence[Opening] = (),
                    title: str | None = None, north_angle_degrees: float | None = None) -> RenderedFloorPlan:
        try:
            room = geometry.to_room(room_id) if isinstance(geometry, RoomGeometryResult) else geometry.model_copy(deep=True)
            if name is not None:
                room.name = name
            prop = PropertyGeometry(property_id="render_scene", rooms=[room], openings=list(openings))
            return self.render_property(prop, title=title, north_angle_degrees=north_angle_degrees)
        except ValidationError as exc:
            raise InvalidGeometryError(f"Invalid room geometry: {exc}") from exc

    def render_property(self, geometry: PropertyGeometry, *, title: str | None = None,
                        north_angle_degrees: float | None = None,
                        damages: Sequence[DamageRegion] = ()) -> RenderedFloorPlan:
        try:
            from matplotlib import rc_context
            from shapely.errors import ShapelyError
            from property_scanner.rendering.drawing import draw_plan
        except ImportError as exc:
            raise FloorPlanRenderError("Install renderer dependencies: pip install -e '.[rendering]'") from exc
        try:
            validated = PropertyGeometry.model_validate(geometry.model_dump())
            with rc_context({"font.family": ["DejaVu Sans", "Arial", "sans-serif"], "text.usetex": False}):
                figure, warnings = draw_plan(validated, self.config, title, north_angle_degrees, damages)
            return RenderedFloorPlan(figure, self.config, warnings)
        except (ValidationError, ShapelyError, ValueError, RuntimeError) as exc:
            raise InvalidGeometryError(f"Cannot render supplied property geometry: {exc}") from exc

    def run(self, context: ScanContext) -> ScanContext:
        """Render only when upstream stages actually supplied a unified result."""
        if context.result is None:
            raise NotImplementedError("Result assembly is not implemented yet; renderer requires supplied geometry.")
        drawing = self.render_property(context.result.property)
        try:
            files = drawing.save(context.output_dir)
            context.artifacts.update(floorplan_png=files.png, floorplan_svg=files.svg)
            context.result.warnings.extend(drawing.warnings)
        finally:
            drawing.close()
        return context
