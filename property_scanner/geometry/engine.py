"""Acquisition-independent single-room point-cloud geometry entry point."""

from pathlib import Path
from property_scanner.geometry.config import GeometryConfig
from property_scanner.geometry.models import RoomGeometryResult

from property_scanner.pipeline.context import ScanContext


class GeometryEngine:
    """Process reconstructed metric clouds without loading capture pipelines."""

    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()

    def run(self, context: ScanContext) -> ScanContext:
        """Require an implementation instead of returning fabricated artifacts."""
        raise NotImplementedError("Reconstructed scene artifact integration is not implemented yet; use process_point_cloud().")

    def process_point_cloud(self, path: Path, *, diagnostics_dir: Path | None = None) -> RoomGeometryResult:
        """Process metric XYZ data; optionally export diagnostics to the supplied directory."""
        try:
            from property_scanner.geometry.processing import process
        except ImportError as exc:
            from property_scanner.core.exceptions import ConfigurationError
            raise ConfigurationError("Install geometry dependencies: pip install -e '.[geometry]'") from exc
        return process(Path(path), self.config, diagnostics_dir)
