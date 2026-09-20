"""Path-level validation for a photo property or legacy single-room folder."""

from pathlib import Path

from property_scanner.core.exceptions import InvalidInputError
from property_scanner.inputs.base import BaseInputAdapter
from property_scanner.schemas.common import InputTier


class PhotoInputAdapter(BaseInputAdapter):
    """Discover immediate room folders while preserving the legacy flat layout."""

    tier = InputTier.PHOTO
    extensions = frozenset({".jpg", ".jpeg", ".png", ".heic"})

    def validate(self) -> None:
        self._require_exists()
        if not self.source_path.is_dir():
            raise InvalidInputError("Photo input must be a property directory containing room folders.")
        direct = self._supported(self.source_path)
        if direct and not 2 <= len(direct) <= 8:
            raise InvalidInputError(
                f"Legacy single-room photo input requires 2–8 supported photos; found {len(direct)}."
            )
        rooms = [path for path in self.source_path.iterdir() if path.is_dir() and not path.name.startswith(".")]
        if not direct and not rooms:
            raise InvalidInputError("Photo property must contain room folders; legacy flat input requires 2–8 photos.")

    def _supported(self, directory: Path) -> tuple[Path, ...]:
        return tuple(sorted(path for path in directory.iterdir()
                            if path.is_file() and path.suffix.lower() in self.extensions))

    def _discover_files(self) -> tuple[Path, ...]:
        direct = self._supported(self.source_path)
        if direct:
            return direct
        return tuple(path for room in sorted(p for p in self.source_path.iterdir()
                                             if p.is_dir() and not p.name.startswith("."))
                     for path in self._supported(room))
