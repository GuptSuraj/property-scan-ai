"""Basic path and extension validation for walkthrough video."""

from pathlib import Path

from property_scanner.core.exceptions import InvalidInputError
from property_scanner.inputs.base import BaseInputAdapter
from property_scanner.schemas.common import InputTier


class VideoInputAdapter(BaseInputAdapter):
    """Reference an officially supported walkthrough-video container."""

    tier = InputTier.VIDEO
    extensions = frozenset({".mp4", ".mov"})

    def validate(self) -> None:
        self._require_exists()
        if not self.source_path.is_file():
            raise InvalidInputError("Video input must be a file.")
        if self.source_path.suffix.lower() not in self.extensions:
            raise InvalidInputError(
                f"Unsupported video extension; expected one of {', '.join(sorted(self.extensions))}."
            )

    def _discover_files(self) -> tuple[Path, ...]:
        return (self.source_path,)
