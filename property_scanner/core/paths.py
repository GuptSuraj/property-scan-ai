"""Filesystem operations kept outside scientific processing modules."""

from pathlib import Path
from uuid import UUID

from property_scanner.core.exceptions import ProcessingError


def create_capture_output(output_root: Path, capture_id: UUID) -> Path:
    """Create a unique capture directory; never reuse a previous run's output."""
    output_path = output_root / str(capture_id)
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ProcessingError(f"Cannot create output directory {output_path}: {exc}") from exc
    return output_path
