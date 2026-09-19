# Architecture

## Three adapters, one application

Photos, video, and LiDAR differ at acquisition. Photos are a set of images,
video needs frame selection and timestamps, and LiDAR exports may supply RGB,
depth, poses, calibration, and metric scale. `BaseInputAdapter` exposes
`validate()`, `load()`, and `prepare()` so each format owns its input rules.
Both `load()` and `prepare()` validate before returning a `NormalizedCapture`.
The adapter factory is the only tier dispatch point in this foundation.

`NormalizedCapture` contains a UUID, tier, source path, UTC ingestion time,
JSON-compatible metadata, and source-file references. Its `prepared_files`
currently means discovered source references, not transformed assets. No media
is decoded or copied. The metadata explicitly records the limited validation.

All adapters converge on `PropertyScanPipeline`. This prevents three separate
versions of measurements, confidence calculations, export, and rendering.
Sensor-specific reconstruction backends may be needed later, but they must
produce a common scene representation before geometry runs.

## Current execution boundary

`PropertyScanPipeline.prepare(adapter)` normalizes input and creates a unique
capture directory. It returns `PreparationResult`, whose only possible status
is `prepared_not_processed`. This is a lifecycle receipt, **not** a unified
property scan result. It can serialize for programmatic use, but the CLI writes
no JSON or processing output.

`PropertyScanPipeline.process(prepared)` defines the shared stage order:
reconstruction, geometry, stitching, openings, damage, measurements, confidence,
and rendering. It currently raises `NotImplementedError` at reconstruction.
Every capture stage still raises through its `run()` interface. The geometry
module now also provides a standalone `process_point_cloud()` entry point for
already reconstructed metric clouds; see [geometry_engine.md](geometry_engine.md).
It returns a typed `RoomGeometryResult` and can convert known measurements to
the existing `Room` schema. No empty geometry,
zero-valued measurements, or arbitrary confidence scores stand in for real work.

The CLI calls only `prepare()`. Exit code 0 reports successful preparation;
normal input/configuration errors report code 2 without a traceback. Python
callers receive application exceptions. There is no filesystem mutation on
package import or settings load.

## Module responsibilities

| Boundary | Owns | Must not own |
| --- | --- | --- |
| Acquisition (`inputs/`) | Discovery, format validation, future decoding/frame selection and calibration loading | Floor areas, damage, or rendering |
| Reconstruction (`reconstruction/`) | Future observations/poses/depth to a common scene, including scale provenance | Repair scope or display formatting |
| Geometry (`geometry/`) | Implemented single-room point-cloud planes, wall intersections, polygons and metric measurements | Capture codecs, reconstruction, stitching |
| Stitching (`stitching/`) | Future inter-room transforms and property coordinate alignment | Independent copies of measurement logic |
| Semantic analysis (`openings/`, `damage/`) | Future doors/windows/openings, visible damage regions, repair/scope suggestions | Invented dimensions or unsupported hidden damage |
| Measurement (`measurements/`) | Future physical quantities derived from scaled geometry | UI formatting or sensor-specific decoding |
| Confidence (`confidence/`) | Future uncertainty and provenance based on actual evidence | Arbitrary fixed confidence values |
| Rendering (`rendering/`) | Future 2D presentation of the unified result | Estimating missing geometry |

`schemas/common.py` owns capture-preparation and receipt contracts; the other
`schemas/` modules own the versioned unified output contract. `core/`
contains application errors, readable application-scoped logging, and output
allocation. `config/` loads validated environment settings. The CLI remains a
thin entry point and does not contain processing logic.

## Replaceable stages and future output contracts

`ProcessingStage` is a small structural protocol with `run(ScanContext)`.
`ScanContext` carries the normalized capture, output directory, and references
to future real artifact files. Its optional `result: PropertyScanResult` holds
only a real assembled result, and defaults to `None`. Keeping references rather than tensors or point
clouds at orchestration boundaries avoids eagerly loading large data into RAM.
There is no service layer, database, or heavyweight dependency injection system.

Implement or replace a stage in the central stage list while preserving its
boundary. Dependencies belong to the implementing module and should be loaded
only when needed. Introduce typed, versioned scene and geometry artifacts with
the first implementation; the current artifact dictionary is an extension point,
not a finalized interchange format. Future stages must define what they require
and produce and raise `ProcessingError` for operational failures. A new backend
must not add sensor-specific branches to measurement or rendering code.

Before introducing geometry, agree on coordinate frames, transforms, metric
units, scale observability, and evidence provenance. Photo/video captures may
lack absolute scale; unavailable dimensions must remain unavailable. Multi-room
capture grouping and manifests are deferred; current photo validation describes
one room only.

The versioned `PropertyScanResult` and JSON utilities are now implemented; see
[data_model.md](data_model.md). The model covers rooms, wall measurements,
surfaces, ceiling heights, openings, property connections, visible damage,
rule-backed concealed-damage flags, scope quantities, warnings, and provenance.
Missing measurements stay null and interval coverage is separate from a quality
score. `PropertyScanPipeline.process()` is typed to return this same model for
every tier, but still raises at reconstruction. Even if all stage placeholders
are bypassed, missing result assembly raises rather than returning fake data.

Future stages will assemble the result after confidence estimation and before
rendering, which will consume that same model rather than calculating its own
dimensions. JSON export currently serializes only supplied, validated objects;
it does not perform analysis. A preparation receipt remains a separate type and
must never be mistaken for a property scan result.

This approach keeps input formats isolated while sharing all downstream rules.
Future backends can be swapped independently without duplicating a complete app
for each acquisition mode. CPU and Apple MPS implementations can be evaluated
per stage within a local-first, 16 GB memory budget. Geometry uses optional,
lazily loaded CPU Open3D, NumPy, Shapely, and matplotlib; nothing requires a GPU
or loads model weights.
