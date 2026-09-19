# Dimensioned 2D floor-plan renderer

The renderer converts supplied typed geometry to a headless PNG and vector SVG.
It performs no reconstruction, wall/corner detection, area/height calculation,
room positioning, or stitching. Geometry diagnostics from Prompt 3 remain a
separate engineering visualization.

```text
RoomGeometryResult / Room / positioned PropertyGeometry
    → Validate polygons, walls and opening references
    → Translate the entire scene to a local drawing origin
    → Neutral polygon fill + final wall segments with opening gaps
    → Supplied wall measurements + room information
    → Optional scale, legend, overall extents and supplied north direction
    → floorplan.png + floorplan.svg
```

## Install and run

Rendering needs only NumPy, Shapely and Matplotlib, supplied by its optional
extra. Open3D, models, cloud services, and API keys are not needed to render.

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,rendering]'
python scripts/render_sample_floorplan.py
python scripts/render_sample_floorplan.py --shape l --output-dir outputs/sample_floorplan_irregular
pytest
```

The default command writes `outputs/sample_floorplan/floorplan.png` and
`floorplan.svg`. The fixture is explicitly synthetic: 4 × 5 m, 20 m², 2.8 m
ceiling, one supplied door and window. The L-shaped option uses the same six-wall
footprint and ground-truth values as the geometry engine's synthetic L room.
Values are fixture declarations, not measured scan results.

Render saved geometry-engine JSON without processing its point cloud again:

```bash
python scripts/render_floorplan.py --input outputs/<capture_id>/diagnostics/geometry/geometry.json --output-dir outputs/<capture_id>
```

Use `--input-kind property` for `PropertyGeometry` JSON or `--input-kind result`
for `PropertyScanResult`. Without `--output-dir`, a root result uses its capture
UUID under `outputs/`; other input kinds receive a fresh run UUID. A missing or
malformed file, invalid polygon, or impossible opening exits cleanly with code 2.
`--config path.json` loads a `RenderingConfig`; `--name` labels geometry input.
Existing photo/video/LiDAR CLI commands are unchanged and remain preparation-only.

## Python API

```python
from pathlib import Path
from property_scanner.geometry.models import RoomGeometryResult
from property_scanner.rendering.floorplan import FloorPlanRenderer
from property_scanner.rendering.styles import RenderingConfig

geometry = RoomGeometryResult.model_validate_json(Path("geometry_result.json").read_text())
renderer = FloorPlanRenderer(RenderingConfig(display_precision=2, show_confidence=False))
drawing = renderer.render_room(geometry, name="Room 1")
try:
    files = drawing.save(Path("outputs/my_capture"))
    # Or drawing.save_png(path), drawing.save_svg(path).
    for warning in drawing.warnings:
        print(warning.code, warning.message)
finally:
    drawing.close()
```

`render_room()` accepts `RoomGeometryResult` or `Room`. Openings can be passed
as a typed sequence. When adding openings to geometry-engine output, first call
`geometry.to_room(room_id)` and reference its wall IDs (`room_id:wall_01`, etc.).
For a positioned property, call `renderer.render_property(property_geometry)`.
It draws each room in its supplied position without fitting or moving rooms.

`run(ScanContext)` can export a real, already supplied `context.result.property`
and register PNG/SVG artifact paths. It still raises `NotImplementedError` when
there is no assembled result; the unfinished capture pipeline never creates a
fake result to invoke rendering. Render warnings are returned on the drawing
and propagated to an existing result by this stage adapter.

## Coordinates and validation

Inputs must already use meters. A single translation subtracts the scene's
minimum X/Y from every polygon, wall, and opening. No rotation, scale recovery,
or independent X/Y stretching occurs. Equal aspect is enforced. The default
`inches_per_meter=1` retains a consistent physical drawing scale across renders;
PNG pixel density is controlled separately by DPI. Source objects remain unchanged.

Every room needs a valid simple polygon and nonempty final wall list. Wall
endpoints must exist, be distinct, follow the polygon boundary, and collectively
cover that boundary within `geometry_tolerance` (0.02 m by default). These checks
validate supplied geometry; they never repair it or invent missing segments.
A missing polygon fails with `InvalidGeometryError`. Missing wall measurements
omit their dimension annotation; coordinates are not used to synthesize one.

Room labels use the supplied name, or a neutral `Room 1`, `Room 2`, etc. The label
anchor uses the polygon centroid when interior and a Shapely representative
point otherwise. Floor area is read from `floor.area` and ceiling height from
`ceiling.height` (or the corresponding geometry-result fields). An unavailable
ceiling is explicitly labeled by default. Unknown area is omitted.

## Dimensions and layout

Wall dimension labels use the supplied `LengthMeasurement`. Only the display is
rounded (two decimals by default). Dimension lines, extension lines, and end ticks
are offset toward the outside of the room when possible. Text extents are checked
against room labels, overall dimension labels, and earlier wall dimensions;
collisions try progressively farther lanes. If all lanes collide, a structured
`DIMENSION_LABEL_OMITTED` warning replaces the crowded annotation. Wall geometry
and its stored value remain intact. Coincident shared walls get one dimension.

`show_confidence=True` displays supplied lower/upper bounds and the supplied
coverage level when available. Bounds without a coverage level are labeled
`Bounds`, not a made-up confidence percentage. No uncertainty is estimated.

Optional overall width/height annotations use the scene's XY bounding extents
only for drawing and are explicitly labeled `Overall`. Here height means Y span,
not ceiling elevation. They are not written back into domain measurements.

The scale bar is drawn in data coordinates with a readable numeric length.
`show_north_arrow=True` only shows north when `PropertyGeometry.metadata`
contains `north_angle_degrees`, or the caller explicitly supplies that keyword
to a render method. The angle is degrees counterclockwise from +X, in [0, 360).
Capture orientation can be passed explicitly using this parameter; it is never
inferred from the drawing or device. Missing north data produces no arrow.

## Openings

`Opening.position_along_wall` is the distance from the host wall's start toward
its end to the **first edge of the opening**. Width is in meters. A host wall,
position, and positive width are needed; missing data omits the opening and
returns `OPENING_NOT_POSITIONED`. Out-of-range or overlapping intervals raise
`InvalidGeometryError` rather than clipping or relocating the opening.

Walls are drawn as split segments, so a gap is real geometry in both PNG and SVG,
not a white patch hiding an unbroken line. Door gaps have jamb ticks and no swing.
The current schema has no standardized hinge/swing fields; arbitrary metadata
is not interpreted as swing information. Windows use two parallel thin lines.
Open passages have a dashed neutral line; unknown openings use a dotted line
and question mark. The optional legend distinguishes the displayed categories.

For a shared opening, the same supplied interval can cut a coincident wall in
the opposite room when the opening explicitly references that room or wall.
This projects known opening geometry; it does not position rooms or detect doors.

## Styling and exports

All style defaults are in `rendering/styles.py`:

| Setting | Default |
| --- | --- |
| wall_line_width / dimension_line_width | 3 / 0.7 points |
| font_size / title_font_size / room_label_font_size | 9 / 15 / 12 points |
| dimension_offset / dimension_lanes | 0.35 m / 3 |
| dimension_text_gap / dimension_tick_size | 0.10 m / 0.07 m |
| symbol_size / layout_padding | 0.06 m / 0.65 m |
| display_precision | 2 |
| show_confidence / show_overall_dimensions / show_north_arrow | false |
| show_wall_dimensions / show_scale_bar / show_legend / show_missing_ceiling | true |
| dpi / background | 250 / white (transparent also supported) |
| inches_per_meter / max_figure_inches | 1 / 40 |
| geometry_tolerance | 0.02 m |

Palette fields (`wall_color`, `dimension_color`, `text_color`, `fill_color`,
`opening_color`) accept six-digit hex colors. Defaults are dark gray on a muted
neutral fill. No point-plane IDs, RANSAC candidates, or diagnostic corners are
displayed. Wall IDs are retained as SVG group identifiers, not visible clutter.

Matplotlib's Agg canvas avoids GUI requirements. PNG honors DPI and transparency;
SVG uses vector paths and editable text (`svg.fonttype=none`) with a viewBox
preserving proportions. Export creates missing parent directories and overwrites
the explicit destination. `FloorPlanRenderError` wraps normal filesystem/export
failures. Close the drawing after use to release its artists. Oversized canvases
are rejected; lower `inches_per_meter` for a large property.

## Current limitations

There is no room stitching, capture processing, reconstruction, scale recovery,
door/window/damage detection, repair scope, confidence calibration, benchmarking,
or UI. This phase only draws supplied geometry and measurements. It has no door
swing contract, polygon holes, curved walls, or automatic multi-page layouts.
Extremely narrow rooms, long names, many openings, and dense multi-room plans may
need smaller fonts or larger drawing scale. Collision handling covers dimension
text; it is not a general architectural layout solver. Room labels may require
manual naming/font choices in very small spaces. Print/export at the intended
size to retain physical scale; arbitrary resizing changes print scale while the
metric proportions and scale bar remain correct.
