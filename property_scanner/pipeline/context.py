"""Small shared stage boundary with file references, not large in-memory data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from property_scanner.schemas.common import NormalizedCapture


@dataclass
class ScanContext:
    """Working state for future stages; artifacts must refer to real outputs.

    Domain-specific artifact schemas will be defined with their implementations.
    This context is internal state, not a serializable measurement result.
    """

    capture: NormalizedCapture
    output_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)


class ProcessingStage(Protocol):
    """A replaceable stage that consumes and returns the shared working state."""

    def run(self, context: ScanContext) -> ScanContext:
        """Process actual data or raise an explicit unsupported-stage error."""
        ...
