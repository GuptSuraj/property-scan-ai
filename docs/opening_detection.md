# Door, window, and opening detection

Opening detection is one shared downstream stage for photo, video, and LiDAR.
It consumes walls already produced by `GeometryEngine`; it never redetects walls
or changes room polygons, wall lengths, areas, or ceiling heights.

```text
RGB frame
    ↓
SegFormer semantic mask
    ↓
Metric depth + intrinsics + camera pose
    ↓
3D samples → known wall → wall-local coordinates
    ↓
Multi-view fusion → metric Opening
```

## Semantic model and device

The implementation uses
[`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
at revision `b9175de73a0a34f7843135853d27629aa6987b2f`. It extracts ADE20K
door, window/windowpane, and wall probabilities behind the
`SemanticSurfaceDetector` interface. Weights are never committed or downloaded
implicitly:

```bash
python scripts/download_models.py --openings
```

Inference uses Apple MPS when available and retries on CPU after an MPS runtime
failure. The model loads once per processing call and processes selected,
calibrated frames only. Probability maps are released frame by frame.

## Metric candidate processing

Thresholded maps receive conservative morphology and connected-component
extraction. Tiny components are discarded. Mask pixels are sampled, invalid
depth is removed, and valid pixels are back-projected using real pinhole
calibration and the canonical camera-to-property pose.

Association requires enough points close to an existing wall plane and inside
its endpoint tolerance. Image location alone never assigns a wall. In wall-local
coordinates, `u=0` is the deterministic wall start and `v=0` is the floor. The
2nd and 98th percentiles provide robust bounds. Width, height, sill height, and
`position_along_wall` are metric; there is no pixel-width fallback.

Candidates on the same wall with nearby centers form one physical opening.
Measurements use medians and store MAD dispersion. Excessive width dispersion
rejects the cluster; unreliable height remains unavailable. Conflicting types
become `unknown`. Final quality combines semantics, wall distance, and metric
sample support. It is not calibrated probability and creates no confidence
intervals.

Broad size bounds, floor contact, wall bounds, minimum metric samples, and
multi-view support suppress phantom detections. An optional 4 cm wall occupancy
grid supplements semantics. A bounded supported gap can become only `unknown` or
`open_passage`, never an automatically guessed door.

## Connections, rendering, and outputs

Accepted detections populate the existing `Opening` model and wall/room
references. An opening may refine an existing Prompt 8 room connection when its
midpoint lies on the other room boundary. It cannot invent adjacency. Exterior
openings keep one room ID.

The existing renderer cuts metric wall gaps and draws door jambs, window double
lines, passages, or unknown markers. It never guesses a hinge or swing. Optional
diagnostic labels show type and width.

Normal commands attempt detection after geometry/stitching. Missing weights leave
reconstruction successful with `OPENING_MODEL_UNAVAILABLE`. Rerun cached evidence:

```bash
python scripts/detect_openings.py --capture ./outputs/<capture_id>
```

Outputs include `openings/openings.json`, `candidates.json`,
`detection_summary.json`, `wall_occupancy.json`, semantic masks, candidate
overlays, wall projections, a labeled diagnostic plan, updated root PNG/SVG, and
updated unified JSON.

Defaults include score 0.55, 150-pixel components, depth stride 4, 25 metric
points, 0.18 m wall distance, 2nd/98th percentiles, two-view support (or strong
single-view quality), 0.25 m merge distance, door width 0.45–2.5 m, window width
0.3–5.0 m, and 0.4 m minimum height. All values live in `OpeningConfig`.

The detector may struggle with closed doors resembling walls, glass, mirrors,
curtains, dark or reflective surfaces, furniture occlusion, open doors hiding
frames, missing glass depth, incomplete clouds, learned photo/video depth errors,
and partially visible openings. Real captures and ground truth remain necessary
for threshold tuning and accuracy evaluation.
