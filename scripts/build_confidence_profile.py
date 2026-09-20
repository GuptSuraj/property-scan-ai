"""Build an explicit benchmark calibration profile from human ground-truth CSV."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--tier", choices=("photo", "video", "lidar"), required=True)
    parser.add_argument("--output", type=Path, default=Path("confidence_profiles"))
    parser.add_argument("--confidence-level", type=float, default=.95)
    args = parser.parse_args()
    errors = defaultdict(list); rows = 0
    with args.benchmark.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("tier") not in (None, "", args.tier): continue
            try: errors[row["measurement_type"]].append(abs(float(row["measured_value"])-float(row["ground_truth_value"]))); rows += 1
            except (KeyError, ValueError): continue
    if not errors: parser.error("No usable measured_value/ground_truth_value rows found for this tier")
    args.output.mkdir(parents=True, exist_ok=True)
    profile = {"tier": args.tier, "confidence_level": args.confidence_level,
        "absolute_error_by_type": {kind: float(np.quantile(values, args.confidence_level)) for kind, values in errors.items()},
        "benchmark_rows": rows}
    path = args.output/f"{args.tier}.json"; path.write_text(json.dumps(profile, indent=2)+"\n")
    print(f"Wrote benchmark calibration profile: {path}"); return 0


if __name__ == "__main__": raise SystemExit(main())
