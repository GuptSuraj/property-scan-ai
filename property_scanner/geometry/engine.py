"""Room geometry extraction contract; no implementation is provided yet."""

from property_scanner.pipeline.context import ScanContext


class GeometryEngine:
    """Replace this stage independently when real processing is introduced."""

    def run(self, context: ScanContext) -> ScanContext:
        """Require an implementation instead of returning fabricated artifacts."""
        raise NotImplementedError("Room geometry extraction is not implemented yet.")
