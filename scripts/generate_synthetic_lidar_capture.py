"""Generate an explicitly synthetic, small registered RGB-D capture."""
import argparse
from pathlib import Path
from property_scanner.reconstruction.lidar.synthetic import generate_capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("inputs/synthetic_lidar"))
    parser.add_argument("--no-drift", action="store_true")
    args = parser.parse_args()
    try:
        path = generate_capture(args.output, drift=not args.no_drift)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Synthetic test capture created: {path}")
    print("This is algorithm test data, not real-world LiDAR accuracy evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
