# Video reconstruction pipeline

Video mode converts a walkthrough into one metric point cloud, then hands that
cloud to the existing geometry engine and floor-plan renderer. It does not contain
a second wall detector, measurement calculator, or renderer.

```text
MP4 / MOV
    ↓
FFprobe metadata + FFmpeg upright frame extraction
    ↓
Deterministic quality filtering and chronological keyframes
    ↓
CPU COLMAP sequential SfM
    ↓
Arbitrary-scale camera-to-world poses + sparse points
              +
Depth Anything V2 Small indoor metric depth
    ↓
Robust multi-frame scale recovery
    ↓
Metric RGB-D keyframe clouds
    ↓
Shared ICP / pose graph (optional) + voxel fusion
    ↓
Structural Z-up orientation
    ↓
Existing geometry engine → existing renderer → unified JSON
```

## Requirements

Video mode supports `.mp4` and `.mov`. On macOS, install the external tools and
the optional Python dependencies, then download the pinned model explicitly:

```bash
brew install ffmpeg colmap
python -m pip install -e '.[dev,video]'
python scripts/download_models.py --video
```

COLMAP runs with CPU SIFT extraction and matching. CUDA is neither required nor
selected. Depth inference selects Apple MPS when PyTorch reports it available and
retries once on CPU if an MPS operation fails. The model is
`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`, pinned to revision
`327deb803d09fac46b05f31a1ccc78a8470c7f6f`. Weights stay under `MODEL_DIR` and
are ignored by Git.

## Video decoding and keyframes

FFprobe supplies width, height, average FPS, duration, frame count, codec, and
container rotation. FFmpeg autorotates before extracting frames, samples at 3 FPS
by default, bounds width to 960 pixels, and stops at 180 candidate frames.

Candidate frames remain chronological. The selector records variance-of-Laplacian
sharpness, mean grayscale brightness, and a 64×48 mean absolute difference from
the last accepted frame. It rejects dark frames, severe blur, near duplicates,
frames inside the minimum time gap, and frames above the 60-keyframe budget. The
defaults favor track continuity; tune them with a `VideoConfig` JSON when needed.
Selected files are hard-linked rather than duplicated.

## SfM and coordinates

COLMAP uses one `PINHOLE` camera, sequential matching with overlap 8, CPU features
and matching, and incremental mapping. Supplied pinhole intrinsics can be scaled
isotropically to extracted frames; otherwise COLMAP estimates calibration. The
parser converts COLMAP world-to-camera quaternions and translations exactly once
to the internal optical camera-to-world convention:

```text
camera X right, camera Y down, camera Z forward
```

SfM world orientation and scale remain arbitrary at this stage. The code never
assumes that COLMAP Z is vertical or that a COLMAP unit is one meter.

## Metric depth and scale recovery

Metric depth runs only for selected, registered keyframes and writes aligned
`float32` meter maps as `video/depth/<frame>.npy`. Values outside the configured
0.2–15 m interval are excluded from fusion.

For each sparse COLMAP observation, the pipeline transforms its 3D point into the
observing camera, projects it with that frame's calibrated intrinsics, rejects
large reprojection error and image-edge samples, then compares SfM optical depth
with predicted metric depth:

```text
scale sample = predicted metric depth / SfM optical depth
```

Per-frame medians use MAD outlier rejection. Frame estimates inconsistent with the
cross-frame median are removed. The global scale is the median of accepted frame
medians so one feature-rich surface cannot dominate it. At least 100 accepted
correspondences across three consistent frames are required by default. The scale
multiplies camera translations and sparse point coordinates; rotations are never
scaled.

`scale_confidence_quality` reports internal support/dispersion only. It is not a
measurement confidence interval or real-world accuracy claim. When scale cannot
be resolved, `result.json` records `METRIC_SCALE_UNRESOLVED`, retains relative SfM
artifacts, and emits no metric cloud, floor plan, or measurements.

## Fusion, registration, and orientation

Depth is sampled every three pixels by default. Each keyframe cloud is voxelized,
bounded in point count, and given normals. Optional refinement reuses the LiDAR
module's point-to-plane neighbor ICP, conservative spatial loop candidates, edge
acceptance checks, and Levenberg-Marquardt pose-graph optimization. Rejected ICP
never replaces the COLMAP estimate; a weak COLMAP prior preserves graph
connectivity. Disable this phase with `--no-registration-refinement`.

Fusion transforms clouds by their metric poses, merges incrementally, and voxel
downsamples after each addition to cap memory. It writes an SfM-oriented metric
cloud first. Upright camera directions establish a gravity-sign prior, which must
also be supported by a large structural floor plane. The resulting transform
rotates the floor normal to +Z and places the floor near Z=0; it does not invent a
yaw direction.

## Configuration defaults

| Option | Default |
| --- | ---: |
| `extraction_fps` | 3 |
| `max_extracted_frames` | 180 |
| `max_keyframes` / `min_keyframes` | 60 / 6 |
| `blur_threshold` | 30 |
| `brightness_threshold` | 12 |
| `duplicate_threshold` | 2 |
| `minimum_time_gap` | 0.25 s |
| `sequential_overlap` | 8 |
| `colmap_max_features` | 4096 |
| `depth_input_size` | 518 |
| `depth_min_m` / `depth_max_m` | 0.2 / 15 |
| `scale_min_correspondences` | 100 |
| `scale_min_frames` | 3 |
| `scale_max_relative_mad` | 0.25 |
| `point_cloud_pixel_stride` | 3 |
| `enable_icp_refinement` | true |
| `enable_loop_closure` | true |

Registration, point budgets, voxel sizes, and geometry thresholds use the nested
`LidarConfig` because those algorithms are acquisition-independent.

## Run and outputs

```bash
python run.py --tier video --input ./inputs/walkthrough.mp4
python run.py --tier video --input ./inputs/walkthrough.mov \
  --extraction-fps 2 --max-keyframes 40 --no-registration-refinement
python scripts/test_video_pipeline.py --input ./inputs/walkthrough.mp4
```

An advanced configuration can be supplied with `--video-config config.json`.
Effective configuration and tool versions are saved in `processing_config.json`.

```text
outputs/<capture_id>/
├── result.json
├── floorplan.png
├── floorplan.svg
├── measurements.csv
├── processing_config.json
├── video/
│   ├── metadata.json
│   ├── keyframe_selection.json
│   ├── frames/ and keyframes/
│   ├── depth/*.npy
│   ├── sfm/
│   ├── scale_estimation.json
│   ├── initial_poses.json
│   ├── optimized_poses.json
│   ├── canonical_poses.json
│   ├── fused_metric_sfm.ply
│   └── fused_video.ply
└── diagnostics/video/
    ├── registrations.json
    ├── pose_graph.json
    └── trajectory.png
```

## Limitations

This first local pipeline may fail on motion blur, low light, blank walls, mirrors,
glass, repetitive texture, rapid motion, weak overlap, moving people, long paths,
or open scenes with few features. Learned monocular metric depth may be biased or
inconsistent. COLMAP may register only one connected subset. Structural
orientation requires a visible floor-like plane. The downstream geometry engine
still targets one room and may struggle with occlusion, sparse walls, curves, and
unsegmented multi-room captures.

Video mode does not implement photo reconstruction, damage analysis, opening
detection, room stitching, benchmark ground truth, cloud processing, a UI, or a
mobile application.
