"""Run photo reconstruction for one room through the standard property CLI."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from property_scanner.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory containing 2–8 overlapping room photos")
    parser.add_argument("--photo-config", type=Path)
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Room directory not found: {args.input}")
    command = ["--tier", "photo", "--input", str(args.input)]
    if args.photo_config:
        command += ["--photo-config", str(args.photo_config)]
    return cli_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
