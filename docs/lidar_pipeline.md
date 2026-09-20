# Canonical LiDAR / RGB-D reconstruction

The LiDAR path reconstructs one already segmented room from registered RGB-D
frames, calibrated intrinsics, and device poses. It runs locally on CPU and uses
the existing geometry engine, renderer, and `PropertyScanResult` contract.
There is no iPhone app integration, cloud API, photo reconstruction,
TSDF or ground-truth benchmarking. Shared opening and optional damage analysis run downstream from cached calibrated frames.

```text
LidarCaptureAdapter → Canonical RGB-D frames + intrinsics + normalized poses
    → bounded keyframes → raw device-pose fusion (always preserved)
    ├─ OFF → existing geometry → existing renderer
    └─ ON  → neighbor ICP → candidate loop ICP → initialized pose graph
             → global LM optimization → corrected fusion
             → existing geometry → existing renderer
    → unified JSON + internal ablation diagnostics
```

## Install and smoke test

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,lidar]'
python scripts/generate_synthetic_lidar_capture.py
python run.py --tier lidar --input ./inputs/synthetic_lidar --drift-correction on
python run.py --tier lidar --input ./inputs/synthetic_lidar --drift-correction off
pytest
```

The generator refuses to replace an existing capture. For another sample, use
`--output inputs/another_synthetic_capture`; `--no-drift` omits the injected pose
errors. The sample has 25 views at 128×96 pixels of a 4×5×2.8 m room. It uses
ray-cast optical-axis depth and a closed camera trajectory. The drift variant
adds increasing translation (up to 0.035, -0.015, 0.004 m) and yaw (0.4°) to
exported device poses only. The reconstruction never reads
`synthetic_ground_truth.json`. This fixture tests algorithms, not real LiDAR
accuracy. Lower ICP residuals do not guarantee more accurate floor areas.

`--lidar-config config.json` accepts a `LidarConfig` JSON object. An explicit
`--drift-correction` overrides its value. Canonical manifests default to ON.
Photo and Video have their own reconstruction backends. To preserve the original startup behavior,
noncanonical LiDAR folders without `manifest.json` and without LiDAR-specific
CLI flags remain **preparation-only** with `prepared_not_processed` status.
Explicit LiDAR processing flags require a valid manifest and fail if it is missing.

## Canonical format, version 1.0.0

```text
capture/
├── manifest.json
├── intrinsics.json
├── poses.json
├── rgb/000000.png, 000001.png, ...
└── depth/000000.png, 000001.png, ...
```

The manifest declares exact file pairs by ID. Filenames need not sort in temporal
order. The `frames` list **is temporal order**, and its IDs must be unique.
Pose records are looked up by ID, never by filename order or list position.
All paths must be relative and resolve inside the capture directory, including
symlinks. Minimal manifest example (illustrative; use actual calibration/poses):

```json
{
  "format_version": "1.0.0",
  "capture_id": "11111111-1111-4111-8111-111111111111",
  "poses_file": "poses.json",
  "intrinsics_file": "intrinsics.json",
  "depth_unit": "millimeter",
  "depth_scale": 1000,
  "depth_truncation_m": 8,
  "pose_convention": "camera_to_world",
  "camera_convention": "opencv_x_right_y_down_z_forward",
  "coordinate_system": "exporter_world_z_up",
  "source_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "registered_rgb_depth": true,
  "synchronized_rgb_depth": true,
  "frame_count": 1,
  "frames": [{"frame_id": 17, "rgb_file": "rgb/000017.png", "depth_file": "depth/000017.png"}]
}
```

At least two usable keyframes are required to process a capture. Optional
manifest fields include `source_app`, `source_app_version`, `device_model`,
timezone-aware `timestamp`, and JSON `metadata`.

`intrinsics.json` contains `width`, `height`, `fx`, `fy`, `cx`, `cy`. Focal lengths
must be positive; principal points must be inside the image. One pinhole
calibration serves all registered frames. No default calibration is substituted.

`poses.json` contains `{"poses": [{"frame_id": 17, "matrix": [[...], ...]}]}`.
Duplicate IDs fail. Missing/invalid individual poses invalidate their frames.
Malformed files or missing calibration fail the capture.

## Registration, depth, and coordinates

RGB must be synchronized and registered to the depth view, with matching pixel
dimensions and the declared intrinsics. RGB files are decoded by Pillow to RGB8;
depth supports single-channel PNG (normally uint16) and numeric NPY arrays
(`allow_pickle=False`). RGB/depth dimension mismatch fails with a registration
error; there is no resizing, calibration substitution, or invented alignment.
App-specific registration/remapping belongs in a future adapter.

Depth is **camera optical-axis Z**, not radial distance. Convert using
`depth_m = stored_depth / depth_scale`. Required combinations are:

| depth_unit | depth_scale | Example |
| --- | --- | --- |
| meter | 1 | 2.450 → 2.450 m |
| millimeter | 1000 | 2450 → 2.450 m |

Units are never inferred from dtype or magnitude. Unknown/inconsistent units
fail. Zero means invalid depth. Negative or nonfinite values invalidate the
frame; too few usable pixels also skips it. Range filtering uses the configured
minimum and the smaller of manifest truncation/configured maximum. Isolated
corrupted or missing files are warned and skipped; too few surviving keyframes
fails. Frame totals, valid decoded frames, invalid frames, unselected frames,
and total skipped frames are saved. `frames_loaded` counts valid decoded frames,
including those intentionally excluded from keyframe processing.

The local optical camera is X right, Y down, Z forward. Pose translations are
meters. Each source pose must be a finite 4×4 rigid transform with last row
approximately [0,0,0,1], orthonormal rotation, and determinant +1. Inputs may
declare `camera_to_world` or `world_to_camera`; one utility converts to the
canonical **camera-to-world** convention.

`source_to_canonical` is a required explicit right-handed rigid transform from
the exporter's world frame to application X/Y floor, Z up coordinates:

```text
canonical_camera_to_world = source_to_canonical × source_camera_to_world
```

Identity must be explicitly declared if no conversion is needed. No tutorial
axis flips are applied. Other optical-camera conventions, left-handed coordinate
frames, distortion, per-frame intrinsics, and unregistered resolutions need an
adapter before this format. The geometry engine's own floor-level transform is
preserved separately in geometry/result metadata.

## Keyframes and fusion

Frames are read one at a time. A deterministic manifest-index stride and maximum
count bound candidates; after the first keyframe, translation OR rotation since
the last selected pose must exceed its configured threshold. Invalid frames do
not shift the manifest stride. Retained local point clouds are voxel-downsampled,
optionally statistically filtered, capped by deterministic point sampling, and
given normals. Full-resolution clouds are not retained.

Raw fusion transforms each keyframe by its normalized **unchanged device pose**,
merges, and incrementally voxel-downsamples. OFF never calls ICP or optimization.
Raw/optimized pose files contain selected keyframe IDs; configurations and the
manifest preserve the sampling recipe. Point budgets bound processing and fail
clearly rather than silently allocating unlimited memory.

Corrected fusion uses the same local keyframe clouds, voxel settings, and fusion
routine, substituting optimized poses. It never overwrites raw fusion. This
isolates pose correction for the ablation; it is not a different reconstruction
algorithm. Future TSDF can be another fusion backend, but is not implemented.

## ICP, loops, and pose graph

Neighbor pairs use `inverse(target_pose) × source_pose` as source-to-target
initialization. Coarse then fine point-to-plane ICP uses estimated normals.
Each record includes initial/fitted transforms, initial/final fitness and RMSE,
reverse overlap, information matrix, acceptance, and rejection reason. No-match
RMSE is null, not a zero error implying success.

Loop candidates use a KD-tree over device camera centers, minimum **keyframe
index** separation, orientation similarity, and a per-frame cap. Only these
bounded candidates run ICP. Acceptance checks fitness, bidirectional overlap,
RMSE, and translation/rotation deviation from the device-pose initialization.
False matches are still possible in repetitive or weakly constrained scenes;
these checks are conservative gates, not semantic loop verification.

Graph nodes start at the supplied camera-to-world device poses. Accepted neighbor
edges are certain; accepted loops are uncertain. Rejected neighbor ICP is not
used: a separately counted, weak device-relative-pose prior maintains connectivity.
Rejected loops are omitted. Open3D Levenberg–Marquardt global optimization anchors
node 0 and prunes uncertain edges. Both pre-optimization loop acceptance and
retained loop-edge counts are reported. No accepted loops produces a warning,
but neighboring/device constraints still allow processing.

## Defaults

All defaults are in `reconstruction/lidar/models.py`; use a JSON override file.

| Settings | Defaults |
| --- | --- |
| drift_correction | on |
| frame_stride / max_frames / min_keyframes | 1 / 80 / 2 |
| min_translation_between_keyframes / min_rotation_between_keyframes | 0.08 m / 8° |
| depth_min_m / depth_max_m / min_valid_depth_pixels | 0.15 m / 8 m / 100 |
| frame_voxel_size / fusion_voxel_size | 0.05 m / 0.03 m |
| max_points_per_frame / max_fused_points | 20,000 / 500,000 |
| normal_radius / normal_max_nn | 0.2 m / 30 |
| remove_frame_outliers / neighbors / std ratio | false / 20 / 2.5 |
| icp coarse / fine correspondence / iterations | 0.25 m / 0.08 m / 40 |
| neighbor_min_fitness / neighbor_max_rmse | 0.25 / 0.06 m |
| loop_min_frame_separation / loop_search_radius | 8 keyframes / 0.8 m |
| loop_orientation_tolerance / max candidates per frame | 35° / 2 |
| loop_min_fitness / loop_min_overlap / loop_max_rmse | 0.5 / 0.5 / 0.05 m |
| max_transform_translation / max_transform_rotation | 0.5 m / 15° |
| pose_graph_edge_prune_threshold / device_prior_information | 0.25 / 0.01 |

The nested `geometry` field accepts the existing `GeometryConfig`, but `up_axis`
must remain z because source normalization is already complete. No room geometry
algorithm is duplicated here.

## Outputs and failure behavior

Each invocation gets a fresh output run/capture UUID so ON/OFF or repeated runs
cannot overwrite each other. The manifest's source capture UUID remains in
capture metadata and `processing_config.json`.

```text
outputs/<run_capture_id>/
├── result.json
├── processing_config.json
├── floorplan.png / floorplan.svg          # selected mode, when geometry permits
├── lidar/
│   ├── raw_fused.ply / raw_poses.json
│   └── corrected_fused.ply / optimized_poses.json  # ON only
├── ablation/
│   ├── raw_fused.ply / raw_poses.json
│   ├── corrected_fused.ply / optimized_poses.json  # ON only
│   ├── floorplan_drift_off.png / .svg
│   ├── floorplan_drift_on.png / .svg              # ON only
│   ├── trajectory_comparison.png
│   ├── drift_floorplan_comparison.png            # ON only
│   └── drift_comparison.json
└── diagnostics/
    ├── lidar/registrations.json, frame_counts.json, pose_graph.json (ON)
    ├── geometry_drift_off/
    └── geometry_drift_on/                        # ON only
```

ON computes both branches in the same run for a direct comparison. OFF computes
only raw outputs; it does not create fake optimized poses or corrected outputs.
Raw/corrected aliases in ablation are actual copies for portable comparison.

Geometry or polygon failure retains reconstruction/registration artifacts and
records structured processing errors/warnings in result JSON. Available partial
geometry remains represented; unavailable area/height stays null. No plan is
fabricated. A completed run with downstream errors prints
`completed_with_processing_errors` and exits 0 so retained outputs can be used;
fatal capture/reconstruction errors exit 2. Consumers must inspect status/errors,
not only the presence of result.json.

`PropertyScanResult` contains actual LiDAR capture provenance, one room's known
geometry, processing timestamps/duration, warnings, and module/library versions.
When model weights are unavailable, opening/damage collections remain empty with structured warnings; no detections are invented.
No capture timestamp is invented if the manifest omits it. Saved references avoid
machine-specific source paths. Effective config/calibration/manifest and normalized
selected poses make runs reproducible; timing and run UUID naturally differ.

## Internal metrics and limits

`drift_comparison.json` explicitly labels its results **internal reconstruction
metrics, NOT benchmark accuracy**. It contains correspondence fitness/RMSE under
initial and final graph poses, edge transform residuals, pose correction
magnitudes, frame counts, graph counts, and geometry diagnostics per branch.
The trajectory plot shows XY camera paths. No start/end closure is presumed
merely because it is the end of a capture. Side-by-side plans preserve the
individual exports and mark a missing plan as unavailable.

The synthetic drift test checks measurable internal residual improvement, not
real-world accuracy or guaranteed measurement improvement. Corrected area can
still be worse than raw area in a weakly constrained scene. Single planes,
repeated walls, poor overlap, device drift outside the search radius, moving
objects, calibration error, glass, mirrors, sparse depth, and heavy occlusion
remain important limitations. A single unsegmented multi-room cloud is not
automatically stitched or partitioned. Registration is sequential and bounded;
Open3D numeric results can vary slightly across versions/platforms.

Future application adapters implement `LidarCaptureAdapter`, supplying canonical
manifest/calibration/frame objects. `process_lidar(..., adapter=...)` accepts such
an adapter; the shipped default is `CanonicalRGBDAdapter`. No undocumented
Record3D/iOS format is guessed.
