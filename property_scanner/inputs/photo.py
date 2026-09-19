"""Basic validation for a single room's photo directory."""

from pathlib import Path

from property_scanner.core.exceptions import InvalidInputError
from property_scanner.inputs.base import BaseInputAdapter
from property_scanner.schemas.common import InputTier


class PhotoInputAdapter(BaseInputAdapter):
    """Accept 2–8 supported image files directly inside a room directory."""

    tier = InputTier.PHOTO
    extensions = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})

    def validate(self) -> None:
        self._require_exists()
        if not self.source_path.is_dir():
            raise InvalidInputError("Photo input must be a directory containing 2–8 room photos.")
        count = len(self._discover_files())
        if not 2 <= count <= 8:
            raise InvalidInputError(
                f"Photo input requires 2–8 supported photos in one room directory; found {count}."
            )

    def _discover_files(self) -> tuple[Path, ...]:
        return tuple(sorted(
            path for path in self.source_path.iterdir()
            if path.is_file() and path.suffix.lower() in self.extensions
        ))
