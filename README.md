# Property Scanner

A local-first Python foundation for property scanning from room photos,
walkthrough video, and exported LiDAR data. The long-term goal is a stitched 2D
floor plan, room dimensions, wall lengths, floor area, ceiling height, openings,
visible damage, repair/scope items, confidence intervals, and structured JSON.

**Current status: foundation, unified data contract, core point-cloud geometry,
and a dimensioned PNG/SVG floor-plan renderer. Reconstruction and AI processing
are not implemented yet.**
The capture CLI remains preparation-only. A separate development command can
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
The capture pipeline stops at unimplemented reconstruction. Geometry has a
working standalone `process_point_cloud()` entry point; rendering consumes
supplied room/property geometry through `render_room()` and `render_property()`.
See [docs/architecture.md](docs/architecture.md) for boundaries and future work.

## Setup

Use Python 3.11. No NVIDIA GPU, Xcode compilation, or heavy AI dependencies are
required by this foundation. CPU is sufficient; no hardware backend is selected.

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
| Photo | Directory with 2–8 immediate image files: jpg, jpeg, png, webp, heic, heif; extensions are case-insensitive | Decode, quality checks, room grouping |
| Video | File with mp4, mov, m4v, avi, or mkv extension | Decode and select walkthrough frames |
| LiDAR | Directory with at least one non-hidden file, including nested files | Validate exported RGB-D, depth, poses, intrinsics, units, and alignment |

Unrelated files in photo directories are ignored. Supported extensions are a
path-validation policy, not a promise that a codec is installed. Files are opened
only to check readability; content, empty/corrupt media, calibration, and LiDAR
completeness are **not** validated. Inputs are referenced in place, never copied.

A valid command creates an empty `outputs/<capture_id>/` directory, prints
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

Tests cover imports, all adapters, file/type validation, settings, unique output
allocation, explicit unimplemented stages, the CLI via subprocesses, data model
constraints, references, JSON round trips, and generated JSON Schema. They
use temporary path fixtures, not real reconstruction datasets. The download
script only explains that model downloading is not implemented; it makes no
network requests.

## Limitations and planned phases

1. **Foundation (current):** package/configuration, adapters, shared stage
   interfaces, preparation CLI, logging, errors, startup tests, and a versioned
   unified result contract with JSON serialization and schema validation tests.
2. **Acquisition:** actual decoding, media quality checks, multi-room manifests,
   and a documented LiDAR interchange format.
3. **Reconstruction:** real scene reconstruction and sensor-specific backends
   behind shared interfaces, with local CPU/MPS feasibility evaluated first.
4. **Geometry (partially implemented):** single-room metric point-cloud geometry
   now works; reconstruction, scale recovery, stitching, and openings remain future work.
5. **Analysis:** visible damage, repair/scope items, measurements, and validated
   confidence intervals.
6. **Outputs (partially implemented):** JSON serialization and dimensioned PNG/SVG
   rendering work for supplied geometry. Capture-pipeline integration, benchmarking,
   and a UI remain future work.

No COLMAP, depth estimation, LiDAR fusion, capture reconstruction, AI models,
Streamlit, benchmarking, database, authentication, mobile app, or Docker is
included. There are no model weights and no automatic model downloads. Large
scientific/AI dependencies will be added only when a concrete stage needs them.

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
