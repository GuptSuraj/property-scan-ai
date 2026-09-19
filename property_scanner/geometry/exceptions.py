"""Expected geometry errors for CLI and library callers."""
from property_scanner.core.exceptions import ProcessingError, InvalidInputError


class PointCloudLoadError(InvalidInputError):
    """Point cloud is missing, invalid, unreadable, or unsupported."""


class InsufficientGeometryError(ProcessingError):
    """Too little supported structural geometry is available."""


class FloorNotDetectedError(InsufficientGeometryError):
    """No supported low horizontal plane was found."""
