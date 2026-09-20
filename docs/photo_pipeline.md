# Photo reconstruction pipeline

Photo mode reconstructs each room independently and passes its metric point cloud
to the existing geometry engine and floor-plan renderer. For multiple successful
rooms it then attempts evidence-based rigid stitching into one property frame.
Failed or disconnected layouts retain local per-room outputs and do not fabricate
a whole-property plan.

```text
Room Photos
    ↓
Decode + EXIF orientation + quality selection
    ↓
CPU COLMAP exhaustive matching and sparse SfM
    ↓
Camera-to-world poses + sparse geometry
                  +
Depth Anything V2 Small indoor metric depth
    ↓
Robust multi-image metric scale recovery
    ↓
Metric RGB-D point clouds → optional shared ICP → voxel fusion
    ↓
Floor-supported canonical Z-up orientation
    ↓
Existing GeometryEngine → room polygon and measurements
    ↓
Existing FloorPlanRenderer → per-room PNG/SVG
    ↓
Cross-room verified features + metric depth → SE(2) room graph
    ↓
Existing FloorPlanRenderer → whole-property PNG/SVG when valid
```

## Input layout

The official input is a property directory containing immediate room folders:

```text
property_01/
├── living_room/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
└── bedroom/
    ├── 01.heic
    ├── 02.jpg
    └── 03.png
```

JPG, JPEG, PNG, and HEIC are supported. Folder names become labels by replacing
underscores/hyphens with spaces and applying title case; no room classifier runs.
A legacy directory containing 2–8 photos directly is accepted as one room for
development and backward compatibility.

Each room needs at least two usable images. More than eight are quality-ranked and
the best diverse eight are selected; the pipeline never silently processes an
unbounded set. One invalid room does not prevent other rooms from completing.

## Image normalization and diagnostics

Pillow decodes images and applies `ImageOps.exif_transpose` before any downstream
stage. The normalized JPEG receives orientation 1, so COLMAP and depth inference
see identical upright pixels. HEIC uses `pillow-heif`, installed by the `photo`
extra. Corrupt, empty, undersized, or all-zero images are rejected independently.

Available EXIF make, model, focal length, 35 mm-equivalent focal length,
orientation, and timestamp are recorded in `image_quality.json`. EXIF is optional.
Photos share a COLMAP camera only when selected records contain matching camera,
resolution, and focal metadata. Otherwise COLMAP creates independent cameras;
the implementation does not assume one device.

Quality scoring uses variance of Laplacian, mean brightness, contrast, and ORB
feature count. A 64×48 grayscale mean absolute difference identifies near
duplicates. Thresholds reject only clearly unusable images, and duplicate removal
never leaves fewer than two. These are capture-quality heuristics, not confidence
intervals.

## SfM and metric scale

The shared command-line COLMAP adapter runs CPU SIFT extraction, exhaustive
matching, and incremental mapping. It uses a `PINHOLE` camera model and allows
COLMAP to estimate intrinsics unless calibrated values are supplied. Diagnostics
include features per image, matched image pairs, verified inlier matches,
registered images, sparse points, components, and normalized optical
camera-to-world poses.

The same pinned Depth Anything V2 Small indoor metric model used by video mode runs
once per registered image. It uses Apple MPS when available, with CPU fallback,
and stores image-aligned `float32` meter maps under each room's `depth/` directory.

Scale recovery is the shared video algorithm. It projects visible sparse points,
checks reprojection error, samples predicted metric depth, rejects invalid ratios
with per-image MAD filtering, rejects inconsistent image estimates, and takes the
median of accepted image medians. The scale multiplies sparse points and camera
translations only. If support is insufficient, the room records
`METRIC_SCALE_UNRESOLVED`, retains relative SfM, and creates no metric measurements
or plan. Scale dispersion is internal reconstruction evidence, not benchmark
accuracy.

## Fusion, geometry, and output

Depth range, pixel stride, frame/fusion voxel size, and point budgets are
configurable. Photo clouds are generated using normalized pinhole intrinsics,
transformed with metric COLMAP poses, and fused incrementally. Optional
point-to-plane ICP reuses the LiDAR/video registration and pose-graph module; it is
off by default because 2–8 learned-depth clouds can be fragile. Rejected ICP keeps
the COLMAP prior.

Camera-up directions provide an orientation prior which must be supported by a
structural floor plane. The final room cloud is metric, right-handed, and Z-up,
with floor near Z=0. No north direction is inferred. `GeometryEngine` alone
calculates walls, corners, polygon, wall lengths, floor area, and ceiling height.
`FloorPlanRenderer` only draws that typed geometry.

Room stitching reuses saved selected images, depth maps, intrinsics, canonical
poses, and room clouds. SIFT matches are geometrically verified, back-projected
to metric 3D correspondences, and fitted with fixed-scale SE(2) RANSAC. A robust
room-graph optimizer may rotate and translate rooms but cannot resize them. See
[multi_room_stitching.md](multi_room_stitching.md).

```text
outputs/<capture_id>/
├── result.json
├── photo_summary.json
├── processing_config.json
├── floorplan.png                 # only when the whole layout is valid
├── floorplan.svg                 # only when the whole layout is valid
├── stitching/
│   ├── candidate_connections.json
│   ├── accepted_connections.json
│   ├── rejected_connections.json
│   ├── initial_room_transforms.json
│   ├── optimized_room_transforms.json
│   ├── room_graph.json
│   ├── layout_metrics.json
│   ├── overlap_report.json
│   └── *.png
├── photo/<room_id>/
│   ├── selected_images/
│   ├── depth/*.npy
│   ├── sfm/
│   ├── image_quality.json
│   ├── feature_matching.json
│   ├── scale_estimation.json
│   ├── registration.json
│   ├── initial_poses.json
│   ├── optimized_poses.json
│   ├── canonical_poses.json
│   ├── room_metric_sfm.ply
│   ├── room_fused.ply
│   ├── geometry.json
│   ├── room.json
│   ├── floorplan.png
│   └── floorplan.svg
└── diagnostics/photo/<room_id>/
    ├── room_status.json
    ├── trajectory.png
    └── geometry/
```

When all successful rooms have a valid layout, the root result contains their
positioned property-frame geometry, inferred graph connections, union footprint,
and unchanged summed room area. Otherwise it retains room-local geometry and
explicit warnings. Openings, damage, and scope items remain empty.

## Installation and commands

```bash
brew install ffmpeg colmap
python -m pip install -e '.[dev,photo]'
python scripts/download_models.py --photo
python run.py --tier photo --input ./inputs/property_01
python scripts/test_photo_room.py --input ./inputs/property_01/living_room
```

The `--photo` and `--video` downloader flags deliberately populate the same pinned
checkpoint cache. Advanced settings can be supplied as
`--photo-config config.json`.

## Capture guidance and limitations

For a real room, use 4–8 photos from different positions, keep roughly 50% or more
overlap between neighboring views, include floor-wall and ceiling-wall boundaries,
and retain textured objects or surfaces. Avoid taking every image from one point.
These are recommendations, not hard validation rules.

Photo mode can struggle with two views, poor overlap, blank walls, mirrors, glass,
low light, repeated textures, small rooms, strong unmodeled lens distortion,
nearly identical viewpoints, heavy furniture occlusion, learned-depth bias,
invisible ceilings, missing corners, or absent cross-room doorway views. Real-photo
metric accuracy and automatic stitching reliability have not been benchmarked.
Multi-storey layouts, semantic door/window detection, damage, repair scope, UI,
mobile capture, and cloud processing are outside this phase.
