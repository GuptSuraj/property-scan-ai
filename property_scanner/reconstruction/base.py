"""Scene reconstruction contract; no implementation is provided yet."""

from property_scanner.pipeline.context import ScanContext


class SceneReconstructor:
    """Replace this stage independently when real processing is introduced."""

    def run(self, context: ScanContext) -> ScanContext:
        """Require an implementation instead of returning fabricated artifacts."""
        raise NotImplementedError("Scene reconstruction is not implemented yet.")
