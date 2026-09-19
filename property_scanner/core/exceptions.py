"""Application errors suitable for presentation by CLI or future UI layers."""


class PropertyScannerError(Exception):
    """Base class for expected application failures."""


class InvalidInputError(PropertyScannerError):
    """An input is missing, unreadable, or of an unsupported basic type."""


class UnsupportedTierError(PropertyScannerError):
    """The requested input mode has no registered adapter."""


class ConfigurationError(PropertyScannerError):
    """Settings cannot be loaded or contain invalid values."""


class ProcessingError(PropertyScannerError):
    """Pipeline preparation or a future processing stage failed."""
