"""Container validation only for a future LiDAR export adapter."""

from pathlib import Path

from property_scanner.core.exceptions import InvalidInputError
from property_scanner.inputs.base import BaseInputAdapter
from property_scanner.schemas.common import InputTier


class LidarInputAdapter(BaseInputAdapter):
    """Accept a nonempty directory without assuming a vendor export schema."""

    tier = InputTier.LIDAR

    def validate(self) -> None:
        self._require_exists()
        if not self.source_path.is_dir():
            raise InvalidInputError("LiDAR input must be an exported capture directory.")
        if not self._discover_files():
            raise InvalidInputError("LiDAR capture directory must contain at least one visible file.")

    def _discover_files(self) -> tuple[Path, ...]:
        return tuple(sorted(
            path for path in self.source_path.rglob("*")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(self.source_path).parts)
        ))
