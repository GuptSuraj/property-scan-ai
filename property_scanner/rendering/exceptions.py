"""Expected renderer errors for API and command-line callers."""
from property_scanner.core.exceptions import ProcessingError


class FloorPlanRenderError(ProcessingError):
    """A supplied plan could not be rendered or exported."""


class InvalidGeometryError(FloorPlanRenderError):
    """Supplied geometry cannot form a trustworthy drawing."""
