"""Readable logging scoped to this application, without resetting root logging."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Install one application handler; repeated calls do not duplicate logs."""
    logger = logging.getLogger("property_scanner")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, "_property_scanner_handler", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    handler._property_scanner_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
