"""Multi-room stitching contract; no implementation is provided yet."""

from property_scanner.pipeline.context import ScanContext


class StitchingEngine:
    """Replace this stage independently when real processing is introduced."""

    def run(self, context: ScanContext) -> ScanContext:
        """Require an implementation instead of returning fabricated artifacts."""
        raise NotImplementedError("Multi-room stitching is not implemented yet.")
