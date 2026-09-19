"""Deterministic UTF-8 JSON and official schema export from the Pydantic model."""

import json
from pathlib import Path

from property_scanner.schemas.result import PropertyScanResult


def validate_result_json(data: str | bytes) -> PropertyScanResult:
    """Parse JSON and apply all Pydantic constraints, including graph checks."""
    return PropertyScanResult.model_validate_json(data)


def result_to_json(result: PropertyScanResult, *, pretty: bool = True) -> str:
    """Serialize with sorted keys, preserving list order, nulls, and Unicode.

    Revalidation catches invalid edits made to mutable nested objects after
    construction. No times, identifiers, or measurement values are generated.
    """
    validated = PropertyScanResult.model_validate(result.model_dump(mode="python"))
    return json.dumps(
        validated.model_dump(mode="json"),
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def save_result(result: PropertyScanResult, path: Path, *, pretty: bool = True) -> Path:
    """Validate before writing; create parent directories and overwrite the file.

    Filesystem errors and Pydantic ValidationError propagate to the caller.
    """
    payload = result_to_json(result, pretty=pretty)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def load_result(path: Path) -> PropertyScanResult:
    """Read a result with no source-media access or path rewriting."""
    return validate_result_json(Path(path).read_text(encoding="utf-8"))


def export_json_schema(path: Path) -> Path:
    """Generate the canonical JSON Schema; never maintain a second schema by hand."""
    schema = PropertyScanResult.model_json_schema(mode="validation")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
