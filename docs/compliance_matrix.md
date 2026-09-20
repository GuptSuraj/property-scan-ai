# Compliance matrix

| Requirement | Implementation | File/module | Artifact | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Photo input | Per-room discovery and reconstruction | `reconstruction/photo` | `photo/` | Complete | Requires COLMAP and depth weights |
| Video input | MP4/MOV keyframes and metric reconstruction | `reconstruction/video` | `video/` | Complete | Requires FFmpeg, COLMAP, depth weights |
| LiDAR input | Record3D `.r3d` adapter and canonical registered RGB-D | `reconstruction/lidar` | `lidar/` | Complete | Stock capture route plus generic format |
| Room/wall dimensions | Shared point-cloud geometry | `geometry` | JSON/CSV | Complete | Real metric geometry required |
| Floor area/ceiling height | Polygon and plane measurements | `geometry` | JSON/CSV | Complete | Null when unresolved |
| Whole-property plan | Evidence-based room stitching | `stitching` | PNG/SVG | Partial | Single floor; evidence may be insufficient |
| Openings | SegFormer + metric geometry | `openings` | `openings/` | Partial | Requires weights; not benchmark validated |
| Visible damage | YOLOE + metric surface projection | `damage` | `damage/` | Partial | Optional model; not benchmark validated |
| Concealed-risk flags | Deterministic explained rules | `damage.detector` | result JSON | Complete | Risk flags, not confirmed hidden damage |
| Repair scope | Deterministic unpriced actions | `damage.detector` | result JSON | Complete | No costing |
| Confidence framework | Profile/evidence precedence | `confidence` | JSON/CSV | Requires benchmark validation | Uncalibrated fallbacks are labeled explicitly |
| LiDAR drift ON/OFF | Raw and optimized fusion | `reconstruction/lidar` | `ablation/` | Complete | Internal metrics are not accuracy |
| Unified JSON | Pydantic contract | `schemas` | `result.json` | Complete | Schema validated |
| PNG/SVG plan | Shared renderer | `rendering` | floorplan files | Complete | Requires valid geometry |
| One-command processing | Shared argparse CLI | `cli.py` | capture directory | Complete | External tools/models required by tier |
| Streamlit UI | Upload/path, processing, results/downloads | `app.py` | browser | Complete | Minimal local UI |
| Benchmark accuracy | External manual benchmark | — | — | Not implemented | Deliberately deferred |
