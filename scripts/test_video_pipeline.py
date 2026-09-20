"""Run the real video CLI against a developer-supplied walkthrough."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from property_scanner.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("inputs/sample_walkthrough.mp4"))
    parser.add_argument("--no-registration-refinement", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(
            f"Video not found: {args.input}. Place your own MP4/MOV there; "
            "the project does not download copyrighted sample footage."
        )
    command = ["--tier", "video", "--input", str(args.input)]
    if args.no_registration_refinement:
        command.append("--no-registration-refinement")
    return cli_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
