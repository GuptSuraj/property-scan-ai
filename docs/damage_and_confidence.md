# Damage, repair scope, and confidence

Selected calibrated RGB/depth frames from Photo, Video, and LiDAR use one downstream damage module. The optional local backend is Ultralytics YOLOE-11s prompt-free segmentation; its vocabulary is conservatively mapped to cracks, water or moisture staining, mold, holes, peeling paint, and general surface damage. Prompt-free mode avoids the separate 572 MB text encoder. It uses Apple MPS when available and CPU otherwise. Missing weights produce `DAMAGE_MODEL_UNAVAILABLE`; they never produce synthetic detections.

```text
RGB mask + metric depth + intrinsics + pose
  → property-space points → known wall/floor/ceiling
  → surface-local polygon → robust metric area/length
  → multi-view fusion → DamageRegion
  → deterministic concealed-risk flags and unpriced scope
```

Concealed-damage flags are explicitly rule based. They report the rule, evidence damage ID, room, and surface, and indicate possible hidden risk rather than confirmed hidden damage. Scope line items use deterministic mappings and contain no pricing.

Measurement uncertainty records one of `benchmark_calibrated`, `multi_view_dispersion`, `geometry_residual`, `tier_prior_uncalibrated`, or `unavailable`. Benchmark profiles take precedence and are only used when a profile file exists. Build a profile from later human benchmark data with:

```bash
python scripts/build_confidence_profile.py --benchmark benchmark_results.csv --tier lidar
```

Without a profile, multi-view MAD or geometric residuals can provide an
evidence-based interval. The standalone confidence class disables tier priors
by default. Unified CLI/UI finalization enables them so every exported
measurement has an interval, labels the method `tier_prior_uncalibrated`, and
adds an `UNCALIBRATED_CONFIDENCE_INTERVALS` warning. A tier-prior interval is
not a benchmark accuracy claim.

Current open-vocabulary damage detection is not benchmark validated. Fine cracks, reflective or patterned finishes, shadows, concealed conditions, and inaccurate estimated depth remain difficult.
