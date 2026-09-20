# Property Scanner

A local-first Python foundation for property scanning from room photos,
walkthrough video, and exported LiDAR data. The long-term goal is a stitched 2D
floor plan, room dimensions, wall lengths, floor area, ceiling height, openings,
visible damage, repair/scope items, confidence intervals, and structured JSON.

**Current status: metric photo reconstruction with evidence-based single-floor room stitching, local metric video reconstruction, canonical LiDAR/RGB-D
reconstruction with optional drift correction, unified JSON, point-cloud geometry,
dimensioned PNG/SVG plans, and conservative metric door/window/opening detection.** Damage AI remains
unimplemented. Photo rooms, video, and canonical LiDAR captures process when
their external dependencies and required inputs are available. A development command can
measure an already reconstructed metric single-room `.ply`/`.pcd` cloud.
Validated result objects can be saved/loaded as JSON and the schema exported.

## Architecture

```text
Photos ──┐
Video ───┼─> Input adapters → NormalizedCapture
LiDAR ───┘
    → Reconstruction → Geometry → Multi-room stitching
    → Doors / windows → Damage → Measurements → Confidence
    → Unified result model → JSON + 2D floor plan
```

All inputs share `PropertyScanPipeline`. Only adapters inspect the input tier.
The shared pipeline dispatches photo, video, and canonical RGB-D reconstruction.
Geometry has a
working standalone `process_point_cloud()` entry point; rendering consumes
supplied room/property geometry through `render_room()` and `render_property()`.
See [docs/architecture.md](docs/architecture.md) for boundaries and future work.

## Setup

Use Python 3.11. Foundation, rendering, geometry, and LiDAR do not require an
NVIDIA GPU. Video depth uses Apple MPS when available and otherwise uses CPU;
COLMAP is explicitly configured for CPU feature extraction and matching.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` installs this project in editable mode with pytest and
jsonschema (used only to test the generated contract).
For runtime dependencies only, use `python -m pip install -e .`.
Dependencies are declared once in `pyproject.toml`: Pydantic, pydantic-settings,
and python-dotenv. The CLI uses standard-library argparse.

## Run

Supply your own files; the repository contains only empty input placeholders.
From the project directory, with the virtual environment activated:

```bash
python run.py --help
python run.py --tier photo --input ./inputs/example
python run.py --tier video --input ./inputs/example.mp4
python run.py --tier lidar --input ./inputs/lidar_capture
```

The installed `property-scan` command takes the same arguments.

| Mode | Current validation | Future acquisition work |
| --- | --- | --- |
| Photo | Property directory with room folders; each room supplies 2–8 selected JPG/JPEG/PNG/HEIC images; verified transition views support stitching | Semantic doors/windows and multi-storey layout |
| Video | Decodable MP4 or MOV; FFprobe metadata, bounded extraction, deterministic keyframes | App-specific intrinsic/IMU metadata adapters |
| LiDAR | Canonical manifest: calibrated registered RGB-D, explicit depth units and rigid poses; legacy preparation accepts a nonempty directory | App-specific export adapters and registration/remapping |

Photo processing decodes and validates each room independently. Video processing
validates and decodes its container. Canonical LiDAR processing
additionally decodes frames and validates calibration, poses, registration
declarations, depth values and dimensions as described below. Input media are
referenced in place; reconstruction artifacts are written separately.

A legacy noncanonical LiDAR preparation command creates an empty `outputs/<capture_id>/` directory, prints
`Status: prepared_not_processed` and an explicit processing-not-implemented
message, then exits with code 0. This means preparation succeeded, not that a
property scan completed. Invalid input/configuration exits with code 2 and a
readable error. Repeated runs get new UUIDs and separate directories.

## Configuration

Settings load on demand, with precedence: explicit Python settings arguments,
environment variables, `.env`, then defaults. `.env` is read from the current
working directory. Unknown entries are ignored.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Nonempty environment label |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, or CRITICAL; case-insensitive |
| `MODEL_DIR` | `./models` | Reserved local model storage |
| `OUTPUT_DIR` | `./outputs` | Capture output root |
| `INPUT_DIR` | `./inputs` | Input root setting for future acquisition tools |

Configured directory paths expand `~` and resolve relative to the current
working directory. The required CLI `--input` is an explicit path relative to
the working directory (or absolute); it is **not** prefixed with `INPUT_DIR`.
Loading settings does not create directories. Preparation creates only the
output root and capture directory. Existing files cannot be used as configured
directories. Inputs, outputs, model files, and `.env` are git-ignored.

## Tests

```bash
pytest
python scripts/download_models.py
```

Tests cover imports, adapters, validation, settings, JSON contracts, synthetic
geometry/LiDAR/video reconstruction, robust video scale recovery, and rendering.
Synthetic fixtures test algorithms and do not establish real-world accuracy. The
model download script makes an explicit network request only with `--video`,
`--photo`, or `--openings`.

## Limitations and planned phases

1. **Foundation (current):** package/configuration, adapters, shared stage
   interfaces, preparation CLI, logging, errors, startup tests, and a versioned
   unified result contract with JSON serialization and schema validation tests.
2. **Acquisition (partially implemented):** canonical RGB-D, video keyframes, and
   photo room discovery/EXIF normalization work. App-specific capture adapters remain future work.
3. **Reconstruction (partially implemented):** local CPU RGB-D fusion, shared
   ICP/pose graphs, CPU COLMAP SfM, metric depth, and robust photo/video scale recovery work.
4. **Geometry (partially implemented):** metric point-cloud geometry and rigid,
   evidence-based single-floor photo room stitching and shared metric opening detection work.
5. **Analysis:** visible damage, repair/scope items, measurements, and validated
   confidence intervals.
6. **Outputs (partially implemented):** JSON serialization and dimensioned PNG/SVG
   rendering work for supplied geometry, video, and canonical LiDAR. Benchmarking,
   and a UI remain future work.

Multi-storey stitching, Streamlit, benchmarking, database, authentication, mobile
capture, and Docker are not included. Model weights are not committed or downloaded
implicitly. Video users explicitly download one pinned indoor metric-depth model.

## Canonical LiDAR / RGB-D pipeline

Use the documented generic export format with registered RGB/depth, real pinhole
intrinsics, explicit depth units, and declared device pose/coordinate conventions.
No app-specific export layout is guessed. See [docs/lidar_pipeline.md](docs/lidar_pipeline.md).

```bash
python -m pip install -e '.[dev,lidar]'
python scripts/generate_synthetic_lidar_capture.py
python run.py --tier lidar --input ./inputs/synthetic_lidar --drift-correction on
python run.py --tier lidar --input ./inputs/synthetic_lidar --drift-correction off
```

The generator creates synthetic algorithm-test data and refuses to overwrite an
existing capture; use `--output` for a fresh destination. ON (the canonical default)
preserves device-pose fusion, runs validated ICP/loop closures and global pose-graph
optimization, and exports both branches. OFF never runs correction. Each command
uses a new output UUID and retains the source capture UUID in metadata.

Outputs include `result.json`, raw/corrected clouds and poses under `lidar/`, mode
floor plans and internal metrics under `ablation/`, and effective configuration.
The selected plan is also saved as `floorplan.png`/`.svg`. Missing geometry leaves
reconstruction artifacts intact and records processing errors instead of fabricating
a plan. Internal residual improvements are **not benchmark accuracy**. Check
`processing_info.errors`; exit 0 can include a partial downstream result.

For backward compatibility, LiDAR folders without a canonical manifest and without
LiDAR processing flags retain preparation-only behavior. Explicit
`--drift-correction` requires the canonical manifest.

## Photo reconstruction

Organize a property as immediate room folders. Folder names become human-readable
room labels; no semantic room classifier runs.

```text
inputs/property_01/
├── living_room/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
└── bedroom/
    ├── 01.heic
    ├── 02.jpg
    └── 03.png
```

Install the shared local reconstruction tools, HEIC decoder, and pinned metric
depth checkpoint, then process rooms sequentially:

```bash
brew install ffmpeg colmap
python -m pip install -e '.[dev,photo]'
python scripts/download_models.py --photo
python run.py --tier photo --input ./inputs/property_01
```

Each room is decoded with EXIF orientation, quality-ranked to 2–8 useful photos,
reconstructed with exhaustive CPU COLMAP matching, aligned to metric depth, fused,
and passed to the shared geometry engine and renderer. Failed rooms are reported
without removing successful room outputs. For multiple successful rooms, verified
cross-room transition views drive a rigid shared-property layout. Capture doorway
views in both neighboring room folders. Missing evidence leaves rooms local and
does not produce an unsupported whole-property plan. Successful stitching adds
root `floorplan.png` and `floorplan.svg`; per-room plans remain available. See
[docs/photo_pipeline.md](docs/photo_pipeline.md) and
[docs/multi_room_stitching.md](docs/multi_room_stitching.md).

Rerun stitching from cached room artifacts without COLMAP or depth inference:

```bash
python scripts/test_stitching.py --input ./outputs/<capture_id>
```

## Video reconstruction

Video mode requires local FFmpeg/FFprobe, CPU-capable COLMAP, and the pinned Depth
Anything V2 Small indoor metric checkpoint:

```bash
brew install ffmpeg colmap
python -m pip install -e '.[dev,video]'
python scripts/download_models.py --video
python run.py --tier video --input ./inputs/walkthrough.mp4
```

MP4 and MOV are supported. Candidate frames are autorotated, sampled, and filtered
for darkness, severe blur, and near duplication. CPU COLMAP estimates an
arbitrary-scale sparse reconstruction. Metric depth is inferred once per selected
registered keyframe on Apple MPS or CPU, and sparse SfM observations robustly align
that reconstruction to meters. Camera translations and sparse points receive the
scale; rotations do not.

The pipeline then reuses the LiDAR ICP/pose-graph and fusion components, the shared
geometry engine, renderer, and `PropertyScanResult`. If metric scale cannot be
resolved, it retains relative SfM diagnostics and emits no measurements or floor
plan. See [docs/video_pipeline.md](docs/video_pipeline.md) for configuration,
coordinate conventions, outputs, failure behavior, and limitations.

Useful overrides are `--extraction-fps`, `--max-keyframes`,
`--no-registration-refinement`, and `--video-config`. A convenience smoke command
uses developer-supplied footage and never downloads sample media:

```bash
python scripts/test_video_pipeline.py --input ./inputs/walkthrough.mp4
```

## Core geometry engine

Install the optional CPU geometry dependencies and run the development command:

```bash
python -m pip install -e '.[dev,geometry]'
python -m tests.synthetic_geometry --output inputs/synthetic_room.ply
python scripts/test_geometry.py --input inputs/synthetic_room.ply --diagnostics
pytest
```

The generated cloud is explicitly synthetic test data. For your own metric cloud,
replace the input path. The engine detects floor/ceiling/wall planes, intersects
bounded wall lines, and reports a supported room polygon, wall lengths, floor
area, and ceiling height. Missing ceiling or polygon evidence stays unavailable.
Debug PNG and cloud exports are optional and remain separate from the dimensioned
floor-plan renderer below.

See [docs/geometry_engine.md](docs/geometry_engine.md) for coordinates, all
configuration defaults, diagnostic definitions, failure behavior, and limitations.
Open3D/NumPy/Shapely/matplotlib are installed by the geometry extra, not required
for the original preparation CLI. Geometry tests require that extra.

## Dimensioned floor plans

Render typed geometry with local Matplotlib/Shapely/NumPy; Open3D is not required:

```bash
python -m pip install -e '.[dev,rendering]'
python scripts/render_sample_floorplan.py
```

The explicitly synthetic sample writes `outputs/sample_floorplan/floorplan.png`
and `floorplan.svg`. To render a geometry-engine result:

```bash
python scripts/render_floorplan.py --input outputs/<capture_id>/diagnostics/geometry/geometry.json --output-dir outputs/<capture_id>
```

Features include equal metric proportions, supplied wall dimensions, room/area/
ceiling labels, known door/window/passage gaps, scale bar, legend, and optional
supplied confidence bounds, overall extents, and north direction. No measurements,
door swings, or room positions are inferred. Missing ceiling height is shown as
unavailable. See [docs/floorplan_renderer.md](docs/floorplan_renderer.md) for API,
styling, validation, and limitations. Full tests including geometry require
`python -m pip install -e '.[dev,geometry,rendering]'`.

## Door, window, and opening detection

One shared downstream module uses the pinned SegFormer-B0 ADE20K checkpoint,
metric depth, camera poses, and existing wall geometry. It can populate doors,
windows, unknown openings, and open passages with metric width, supported height,
sill height, and position along a wall. Detections are conservative and are not
guaranteed for every capture.

```bash
python -m pip install -e '.[dev,openings]'
python scripts/download_models.py --openings
python scripts/detect_openings.py --capture ./outputs/<capture_id>
python scripts/test_openings_synthetic.py
```

Normal photo, video, and canonical LiDAR commands attempt the stage automatically.
Without cached weights they retain reconstruction outputs and report a warning.
See [docs/opening_detection.md](docs/opening_detection.md).

## Unified JSON contract

All three input modes share `PropertyScanResult` schema version `1.0.0`.
See [docs/data_model.md](docs/data_model.md) for fields, units, IDs/references,
partial results, serialization utilities, and validation rules.

Regenerate the schema directly from the Pydantic root model:

```bash
python scripts/export_schema.py
```

This writes `schemas/property_scan_result.schema.json`. Do not edit that generated
file manually. Cross-field and reference validation requires the Pydantic model;
JSON Schema alone cannot enforce those relationships.

`tests/fixtures/sample_property_result.json` is **entirely synthetic schema test
data**, not a real property scan or AI result. It is the only sample result.
The CLI does not load this fixture or generate an example scan.
