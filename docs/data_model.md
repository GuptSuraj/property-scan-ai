# Unified property scan data model

`PropertyScanResult` is the single output contract for photos, video, and LiDAR.
It uses Pydantic v2 and schema version `1.0.0`. Capture tier is provenance, not a
selector for a different output shape. Consumers can render and report the same
fields regardless of acquisition mode.

No processing is implemented by these models. They store and validate supplied
facts; they do not compute polygons, areas, dimensions, damage, uncertainty,
repair recommendations, or prices.

## Hierarchy and storage

```text
PropertyScanResult
├── schema_version
├── capture: CaptureMetadata
├── property: PropertyGeometry
│   ├── rooms: Room[]
│   │   ├── walls: Wall[]
│   │   ├── floor: FloorSurface | null
│   │   ├── ceiling: CeilingSurface | null
│   │   ├── opening_ids: ID[]
│   │   └── damage_ids: ID[]
│   ├── openings: Opening[]
│   ├── room_connections: RoomConnection[]
│   ├── footprint_polygon
│   ├── total_floor_area
│   └── bounding_dimensions
├── damages: DamageRegion[]
├── concealed_damage_flags: ConcealedDamageFlag[]
├── scope_line_items: ScopeLineItem[]
├── warnings: ResultWarning[]
├── processing_info: ProcessingInfo | null
└── metadata
```

The root requires `capture` and `property`. Geometry may be incomplete. Individual
models are in `schemas/capture.py`, `measurements.py`, `primitives.py`,
`geometry.py`, `damage.py`, and `result.py`; preparation-only models remain in
`schemas/common.py` for compatibility.

Room floor area has one canonical location, `room.floor.area`, and room ceiling
height is `room.ceiling.height`. If only a height is known, a ceiling may contain
just its surface ID and that height. Room-level copies of these measurements
are intentionally absent. The property total area is optional and is not
automatically summed or checked against individual rooms in this phase.

Openings live once at `property.openings`; rooms and walls link via `opening_ids`.
A shared door can list two rooms without duplicating the door object. Each wall
belongs to one room; `Opening.wall_id` optionally identifies the primary host
wall. The opposite room's wall may reference the same opening. No coincident
wall geometry, thickness, offsets, or reciprocal references are inferred.

Damage objects live once at the root. Room, wall, floor, and ceiling `damage_ids`
link to these observations. Scope items link to a damage ID when applicable.
Adjacency is represented by top-level typed room connections, not free-form
room-name strings. All warnings live at the root to avoid a second warning list
inside processing information.

## IDs and references

Capture IDs are UUIDs, matching `NormalizedCapture`. Other IDs are nonempty
strings matching `[A-Za-z0-9][A-Za-z0-9_.:-]*` (for example `room_01`, `wall_04`,
or UUID text). No IDs or timestamps are generated automatically for results.

IDs are unique within their entity collection. Walls, floors, and ceilings
additionally share a property-wide **surface ID namespace**, so a damage's
`surface_id` can reference any of them without ambiguity. Room-owned wall IDs
must be unique across rooms, even when two walls represent opposite sides of
the same physical partition.

At property validation, wall ownership, opening references, connection endpoints,
and surface ID uniqueness are checked. At root validation, damage references,
scope references, warnings, and concealed-risk locations must resolve. If both
room and surface are supplied, they must agree. Damage references cannot
contradict a known room or surface. Unknown optional references should be null
or omitted, not a made-up ID. A partially known entity may be supplied as an
ID-only stub where its model permits it; a non-null dangling reference is rejected.
Reciprocal reference lists are optional rather than automatically populated.

## Measurements and coordinate conventions

`MeasuredValue` stores `value`, `unit`, optional lower/upper bounds,
`confidence_level`, `confidence_score`, and `method`.

| Field type | Unit | Meaning |
| --- | --- | --- |
| `LengthMeasurement` | `m` | Wall dimensions, ceiling height, opening dimensions/offsets, damage length |
| `AreaMeasurement` | `m2` | Floor, ceiling, property, and damage areas |
| `AngleMeasurement` | `degree` | Counterclockwise wall orientation from +X, 0–360 |
| `QuantityMeasurement` | `m`, `m2`, or `item` | Scope quantity with its unit inside the measurement |

Measurements and bounds must be finite and nonnegative. Strings and booleans
are not accepted as numeric values. Specialized measurement fields reject
incompatible units. Optional bounds must satisfy `lower_bound <= value <=
upper_bound` wherever supplied. A one-sided bound is permitted. Confidence
values are in [0, 1], including endpoints. Interval coverage and quality score
are distinct and may independently be unknown. Nothing derives one from the
other or fills in uncertainty automatically.

`Point2D`/`Point3D` coordinates are finite meters, and negative coordinates are
valid. Property geometry uses one local right-handed frame: X/Y are horizontal
and Z is up. No geodetic origin or building heading is assumed. Room/floor/
ceiling footprint polygons and wall endpoints use this shared XY frame.
Ceiling height means distance above the room floor, not global Z elevation.

Damage polygon and box coordinates are metric and surface-local. For walls,
local X runs from the wall start toward its end and local Y is height above its
base. For horizontal surfaces, coordinates use the property XY frame. A damage
location whose surface/frame is not known should omit polygon/box fields until
it can be located reliably. Image-pixel geometry is not part of version 1.0.0;
source image/frame references can still be recorded.

Polygons require at least three distinct ordered points, with implicit closure
(a repeated final vertex is also accepted). Bounding-box minima must not exceed
maxima. No geometry algorithms check self-intersection, collinearity, wall-length
agreement, area consistency, containment, or physical plausibility in this phase.

## Partial results, semantics, and provenance

Absent dimensions, damage size, ceiling height, and confidence are `null`.
Zero is a known numeric value, never an unknown-value marker. An entire floor,
ceiling, polygon, or processing record may also be absent. Empty collections mean
no entities have been supplied; they do not establish that detection ran or that
there is no damage. Producers should record real modules used and structured
warnings/errors to explain incomplete analysis.

Opening and connection categories use enums including `unknown`. Room types use
an open nonempty string vocabulary (`bedroom`, `kitchen`, and custom labels).
Damage types offer `DamageType` constants plus nonempty custom strings. Severity
uses minor/moderate/severe/unknown. Visible damage can have length, area, both, or
neither. Hidden-damage suspicions are separate `ConcealedDamageFlag` records;
both a rule identifier and nonblank rule explanation are required. Evidence is
supplied, never manufactured. A suspicion is not a detected concealed defect.

`ScopeLineItem` requires an action and description. Its optional `quantity`
includes the unit once; it has no duplicate top-level `unit` and no pricing.

Capture device fields and actual capture/processing timestamps are optional.
`NormalizedCapture.created_at` is ingestion time and must not be silently mapped
to capture time. Source references are portable strings; loading a result does
not resolve paths, read media, or check whether those sources still exist.

`ProcessingInfo` records only supplied pipeline version, timestamps, duration,
module names, model versions, and errors. Timestamps require timezone offsets;
completion cannot precede start. Duration is optional and not computed from wall
clock timestamps because timing semantics may differ between implementations.

## JSON API

After installing the project, use the shared helpers:

```python
from pathlib import Path
from property_scanner.schemas import PropertyScanResult
from property_scanner.schemas.serialization import (
    load_result, result_to_json, save_result, validate_result_json,
)

# This is explicitly synthetic test data, not a scan.
result: PropertyScanResult = load_result(
    Path("tests/fixtures/sample_property_result.json")
)
payload = result_to_json(result)  # Pretty JSON; use pretty=False for compact JSON.
validated = validate_result_json(payload)
save_result(validated, Path("outputs/synthetic-schema-example.json"))
```

Serialization retains nulls, uses UTF-8, sorts object keys, preserves list order,
and adds a final newline. It is deterministic for the same validated data. Loading
and saving do not stamp timestamps or allocate new IDs. Saving revalidates nested
data, creates parent directories, and overwrites the explicit destination.
Pydantic validation and filesystem errors propagate to the calling application.
Model objects are mutable; validate again after edits or use the save helper,
which does this automatically.

## Versioning and JSON Schema

```bash
python scripts/export_schema.py
# Optional destination:
python scripts/export_schema.py --output /tmp/property_scan_result.schema.json
```

The default output is `schemas/property_scan_result.schema.json` in the project
root, directly generated by `PropertyScanResult.model_json_schema()` using
Pydantic's JSON Schema dialect (Draft 2020-12). Commit regenerated output with
model changes; a test checks it matches the model and validates the fixture with
the lightweight development-only `jsonschema` package.

**JSON Schema validates structure, scalar constraints, and enums.** Pydantic also
enforces cross-field bounds, distinct polygon vertices, uniqueness, and graph
references through Python validators. Consumers needing equivalent full
validation must use Pydantic or implement those documented checks themselves.

The current reader accepts exactly `schema_version: "1.0.0"` (also the default
when constructing a model); unknown versions are rejected. Existing fields must
retain their meaning. Introduce new optional fields in an explicitly released
version, and use major versions plus explicit migrations for incompatible changes.
Do not silently reinterpret old measurements or units. Models reject unknown
fields to catch typos; namespaced JSON `metadata` dictionaries provide an
extension mechanism without changing the declared schema. Extensions are not
automatically trusted or interpreted by core consumers.

## Integration boundary

`PropertyScanPipeline.process()` now returns `PropertyScanResult` in its type
contract for every tier. `ScanContext.result` is initially `None`, enabling future
stages to pass an actual result to downstream consumers without inventing one.
Photo reconstruction still raises `NotImplementedError`. Canonical LiDAR RGB-D
and metric video reconstruction populate this contract with genuinely available
geometry, provenance, and processing warnings; see [lidar_pipeline.md](lidar_pipeline.md)
and [video_pipeline.md](video_pipeline.md).
Geometry has a standalone point-cloud entry point;
the renderer consumes supplied geometry and can export a real context result.
See [floorplan_renderer.md](floorplan_renderer.md). The CLI processes canonical
LiDAR captures and MP4/MOV video while photo remains preparation-only. Result serialization is
shared by all consumers. The schema sample fixture is marked synthetic at root level,
in a structured warning, and in observation notes.
