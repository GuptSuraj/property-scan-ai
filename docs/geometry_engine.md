# Core point-cloud geometry

This engine processes **one already reconstructed room**, independently of how
the cloud was acquired. It accepts `.ply` and `.pcd` XYZ data without colors,
uses CPU Open3D/NumPy/Shapely, and returns `RoomGeometryResult`. No capture,
reconstruction, stitching, openings, or damage algorithms are introduced.

```text
Metric point cloud → Validation → Voxel/SOR preprocessing → Normals
    → RANSAC candidates → Floor / ceiling selection → Vertical wall planes
    → Duplicate merging → Floor-plane projection → Bounded intersections
    → Ordered room polygon → Lengths / area / plane-based height
```

## Installation and a reproducible synthetic example

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,geometry]'
python -m tests.synthetic_geometry --output inputs/synthetic_room.ply
python scripts/test_geometry.py --input inputs/synthetic_room.ply --diagnostics
pytest
```

The generated cloud is labeled synthetic and is only an algorithm fixture, not
a real scan or benchmark. Its rectangular ground truth is 4 × 5 × 2.8 meters.
Use `--shape l` for a six-wall concave room with 19 m² floor area. No generated
clouds or diagnostics are committed. For an existing cloud:

```bash
python scripts/test_geometry.py --input /path/to/room.pcd --up-axis y --diagnostics
```

Geometry dependencies are an explicit optional extra so startup/schema users
can retain the lightweight installation. Open3D has substantial transitive
dependencies, but no CUDA or ML model is required. Without the extra, the geometry
test module skips; full geometry verification requires the installation above.
The engine imports scientific dependencies only when processing a cloud.

## API and integration

```python
from pathlib import Path
from property_scanner.geometry.engine import GeometryEngine
from property_scanner.geometry.config import GeometryConfig

engine = GeometryEngine(GeometryConfig(up_axis="z", voxel_size=0.03))
geometry = engine.process_point_cloud(
    Path("inputs/synthetic_room.ply"),
    diagnostics_dir=Path("outputs/manual/diagnostics/geometry"),  # optional
)
room = geometry.to_room("room_01")
```

`to_room()` populates measured walls, floor polygon/area, and ceiling height.
It creates no capture metadata, openings, damage, or complete property result.
Both polygon and area remain null when closure fails. Ceiling height remains
null when there is no supported ceiling. Original capture commands still only
prepare inputs; `GeometryEngine.run(ScanContext)` intentionally raises until a
reconstruction artifact contract exists. The implemented entry point is
`process_point_cloud()`.

## Coordinates and scale

Coordinates **must already be metric**. This engine cannot recover unknown scale
or detect whether a number represents meters versus centimeters.

`up_axis` accepts x/y/z and their negative variants, defaulting to z. A
right-handed rotation maps that axis to +Z. It then detects an approximately
horizontal floor and levels the cloud to that fitted plane. The resulting X/Y
frame is horizontal and the floor is at Z=0. A 4×4
`source_to_floor_transform` is returned (column-vector convention), so later
adapters/stitching can relate output geometry to source coordinates.

This is a configured vertical prior with limited floor-based refinement, not
automatic gravity discovery. A box can have several indistinguishable dominant
directions. Wrong up axes, large device tilt, and nonmetric scale can prevent
floor detection. Future adapters can transform scenes before calling the engine.

## Algorithms

Loading rejects missing/unsupported/unreadable/empty files, nonfinite points,
and insufficient point counts. Voxel downsampling happens before statistical
outlier removal and normal estimation. Normals are normalized and aligned to
the +Z hemisphere where possible; vertical normals retain sign ambiguity.
Plane classification uses the fitted plane normals, independent of their sign.

Open3D iteratively extracts bounded-count RANSAC planes with a fixed random seed.
Inlier sets are removed between fits, so floor/ceiling support is not reused as
wall support. Each candidate is refit by total least squares (SVD). The floor
must be horizontal, low in the cloud, and cover enough XY extent. Selection
uses both spatial support and inlier count; the largest plane alone is not
necessarily the floor. Plane area estimates are convex support envelopes and
can overestimate area around occlusions; they are not reported floor areas.

Ceiling selection separately requires near-parallel orientation, plausible
height, significant floor-relative coverage, and proximity to the cloud top.
These checks reject ordinary low furniture planes but cannot guarantee semantic
classification in every scene. Height comes from the two fitted plane equations,
along the floor normal through the floor support centroid. For exactly parallel
planes this is their separation; for slightly nonparallel planes it is a local
height at that documented anchor, not a constant distance between intersecting
infinite planes. No raw bounding-box height or statistical interval is substituted.

Walls must be approximately vertical, tall/wide enough, and extend near the
floor. Duplicate candidates merge only with similar normals, small symmetric
plane distances, and overlapping/nearby horizontal and vertical extents. The
combined support is refit. Close parallel partitions can still be ambiguous;
use merge tolerances appropriate to the data.

Each wall intersects the leveled floor to produce a 2D line. Observed support
quantiles bound its initial segment. Pairwise wall-line intersections are
accepted only near both segment ends and within a configured extension distance;
near-parallel intersections are rejected. Nearby corners are merged. The small
wall graph must have two incident walls per corner and two corners per wall.
Shapely polygonization must yield one valid simple polygon using all walls.
Disconnected loops, branches, incomplete walls, or self-intersections yield a
warning, preserve observed segments, and leave polygon/area unavailable. There
is no convex-hull or rectangular fallback. Concave L-shaped rooms are supported.

Successful polygons are counterclockwise. Wall lengths come from final bounded
corner endpoints; unsuccessful closures retain observed plane-extent lengths.
Area comes from the actual Shapely polygon. Internal values are not rounded.

## Configuration defaults

All thresholds live in `geometry/config.py`; distances/areas are SI and angles
are degrees. `--config path.json` loads the same validated configuration; omitted
fields retain defaults and `--up-axis` overrides the file's axis.

| Setting | Default |
| --- | --- |
| voxel_size | 0.03 m |
| remove_outliers | true |
| statistical_nb_neighbors / statistical_std_ratio | 20 / 2.5 |
| normal_radius / normal_max_neighbors | 0.15 m / 30 |
| min_points / max_downsampled_points | 200 / 300,000 |
| ransac_distance_threshold / ransac_n / ransac_iterations | 0.025 m / 3 / 1,000 |
| random_seed / max_planes / min_plane_inliers | 7 / 24 / 100 |
| horizontal_angle_tolerance / vertical_angle_tolerance | 12° / 12° |
| ceiling_parallel_tolerance | 5° |
| min_horizontal_area | 1 m² |
| min_floor_coverage / min_ceiling_coverage | 0.2 / 0.4 |
| floor_elevation_tolerance / ceiling_top_tolerance | 0.25 m / 0.3 m |
| min_ceiling_height / max_ceiling_height | 2 m / 5 m |
| min_wall_length / min_wall_height | 0.7 m / 1.8 m |
| wall_base_tolerance | 0.35 m |
| wall_merge_angle_tolerance / distance / gap | 5° / 0.06 m / 0.08 m |
| corner_merge_tolerance / intersection_extension | 0.04 m / 0.3 m |
| min_intersection_angle / min_polygon_area | 15° / 1 m² |
| extent_quantile | 0.005 (trim each tail) |

## Diagnostics and quality

Optional diagnostics write to `outputs/<run_uuid>/diagnostics/geometry/` in the
development CLI: cleaned cloud, floor/ceiling/wall inlier clouds, `geometry.json`
(planes, projected segments, corners, polygon, measurements and metrics),
`config.json`, and headless `geometry.png`. The PNG is an engineering debug view,
not a final floor-plan renderer. API callers supply the destination explicitly.

Diagnostics include original/downsampled/filtered counts, structural inlier
ratios relative to the filtered cloud, unique explained-point fraction, residual
RMSE per plane, rejected/merged wall counts, and polygon closure error. Closure
error is the maximum distance from a reconstructed corner to the nearest
observed endpoint of its wall, not an automatically zero closed-ring gap.
It is unavailable if the graph cannot close.

Plane quality is explicitly a fit heuristic, `exp(-RMSE / RANSAC threshold)`;
corner quality uses the minimum of its contributing wall scores. These are not
calibrated probabilities or confidence intervals. Inlier counts, coverage, and
residuals remain separately available for future confidence estimation.

## Limits and performance

This first implementation may struggle with incomplete walls, heavy furniture
occlusion, curved walls, glass, mirrors, severe noise, open-plan spaces, very
sparse clouds, and multiple rooms in one unsegmented cloud. Short walls below
the configured minimum and nearly collinear corners may be rejected. It assumes
a flat floor and approximately parallel ceiling. It does not infer missing
walls, estimate units, partition rooms, or identify object semantics.

The raw file must fit in memory. Downsampling precedes normal estimation and
RANSAC; a point budget fails cleanly instead of processing arbitrarily large
downsampled clouds. Structural extraction stops at `max_planes`; increase this
only with care. Pairwise operations run over this small candidate set, not raw
points. Debug inlier clouds are materialized only while writing diagnostics.
Open3D's global RANSAC seed is intended for sequential runs; concurrent callers
should use separate processes for deterministic execution. Numerical results can
vary slightly across Open3D/platform versions; tests use metric tolerances.
