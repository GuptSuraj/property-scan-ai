"""Export the official result schema from the installed editable project."""

import argparse
from pathlib import Path

from property_scanner.schemas.serialization import export_json_schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "schemas" / "property_scan_result.schema.json",
    )
    args = parser.parse_args()
    print(f"Schema exported: {export_json_schema(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
