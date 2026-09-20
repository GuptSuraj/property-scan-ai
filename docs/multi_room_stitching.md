# Multi-room stitching

Photo reconstruction produces metric polygons in separate room-local frames.
The stitching stage estimates a rigid transform for each supported room and
places it in one property frame. It never rescales, stretches, or edits geometry
engine measurements.

```text
Room A ─┐
Room B ─┼→ Connection evidence → Room graph → Initial layout
Room C ─┘                                      ↓
                                      Global optimization
                                               ↓
                                      Overlap validation
                                               ↓
                                      PropertyGeometry → floor plan
```

## Frames and evidence

Room artifacts remain in `photo/<room_id>/`. A connection matrix maps room B
local XY into room A local XY. Final transforms contain only X/Y translation and
counterclockwise yaw; the deterministic root has identity. Original `room.json`
files stay local while positioned output uses `coordinate_scope=property_shared`.

For a bounded set of room pairs, the implementation runs SIFT on normalized
selected images, Lowe ratio filtering, and fundamental-matrix RANSAC. It
back-projects verified features through saved metric depth and pinhole intrinsics.
Deterministic RANSAC fits a fixed-scale SE(2) transform between canonical room
frames. Raw match count alone cannot establish adjacency. Optional conservative
point-to-plane ICP refines a transform already supported by visual and metric
correspondences; failed ICP retains its input transform.

Photograph near doorways and, where practical, include a view through each
doorway into the neighboring room in both room folders. Names are labels only
and never adjacency evidence.

## Graph, optimization, and failure behavior

Accepted evidence forms a weighted graph. Root selection uses graph degree,
summed evidence quality, room area, and room ID as deterministic tie breakers. A
maximum spanning tree gives the initial layout. Inconsistent non-tree cycle edges
are rejected. `scipy.optimize.least_squares` refines X/Y/yaw with Huber loss;
room dimensions are not variables. Manhattan regularization is off by default.

Disconnected rooms receive no property transform. The pipeline retains their
per-room artifacts and reports `DISCONNECTED_ROOM_GRAPH`; it does not arrange
them in a grid. Ambiguous constraints remain in rejected evidence.

Shapely checks each polygon pair. Tiny intersections inside configurable area
and ratio tolerances are allowed. A material intersection emits
`EXCESSIVE_ROOM_OVERLAP` and `PROPERTY_LAYOUT_CONFLICT`, and the photo pipeline
does not publish a whole-property plan. Polygons are never shrunk.

The footprint is `unary_union` of positioned rooms. A single union populates
`footprint_polygon`; all components are retained in `footprint_components`.
Total usable area is the sum of unchanged room areas. Union area and the
difference from that sum are diagnostic values.

## Outputs and rerunning

A valid layout writes root `floorplan.png`, `floorplan.svg`, and updated
`result.json`. Per-room artifacts remain unchanged. `stitching/` contains graph,
evidence, transform, cycle, overlap, and initial/optimized layout diagnostics.

Rerun from cached Prompt 7 artifacts without COLMAP or depth inference:

```bash
python scripts/test_stitching.py --input ./outputs/<capture_id>
```

Generate a synthetic end-to-end plan with:

```bash
python scripts/test_stitching_synthetic.py
```

Defaults include 30 ratio-filtered matches, 12 inliers, 0.25 inlier ratio,
0.18 m correspondence RANSAC tolerance, 0.35 acceptance score, Huber global
optimization, 0.35 cycle residual threshold, and overlap tolerances of 0.08 m²
and 2%. All values are centralized in `StitchingConfig`.

This version supports single-floor properties. It may fail when room folders
share no transition content, doorway views are absent, walls are blank, textures
repeat, mirrors/glass create false correspondences, hallways are uniform,
independent room scales disagree, overlap is weak, or a room reconstruction
failed. It does not detect semantic doors/windows or handle multiple storeys.
Real-photo testing remains necessary to tune thresholds and quantify reliability.
