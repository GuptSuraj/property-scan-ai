"""Centralized metric layout and restrained architectural styling."""
from typing import Annotated, Literal
from pydantic import Field
from property_scanner.schemas.base import ContractModel

Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]


class RenderingConfig(ContractModel):
    wall_line_width: float = Field(default=3.0, gt=0)
    dimension_line_width: float = Field(default=0.7, gt=0)
    font_size: float = Field(default=9, gt=0)
    title_font_size: float = Field(default=15, gt=0)
    room_label_font_size: float = Field(default=12, gt=0)
    dimension_offset: float = Field(default=0.35, gt=0)
    dimension_lanes: int = Field(default=3, ge=1, le=8)
    dimension_text_gap: float = Field(default=0.10, gt=0)
    dimension_tick_size: float = Field(default=0.07, gt=0)
    symbol_size: float = Field(default=0.06, gt=0)
    layout_padding: float = Field(default=0.65, gt=0)
    display_precision: int = Field(default=2, ge=0, le=6)
    show_confidence: bool = False
    show_wall_dimensions: bool = True
    show_overall_dimensions: bool = False
    show_scale_bar: bool = True
    show_legend: bool = True
    show_north_arrow: bool = False
    show_missing_ceiling: bool = True
    show_opening_labels: bool = False
    dpi: int = Field(default=250, ge=72, le=600)
    background: Literal["white", "transparent"] = "white"
    inches_per_meter: float = Field(default=1, gt=0, le=5)
    max_figure_inches: float = Field(default=40, ge=5, le=100)
    geometry_tolerance: float = Field(default=0.02, gt=0, le=0.2)
    wall_color: Color = "#24282C"
    dimension_color: Color = "#62686D"
    text_color: Color = "#24282C"
    fill_color: Color = "#F3F2EE"
    opening_color: Color = "#62686D"
