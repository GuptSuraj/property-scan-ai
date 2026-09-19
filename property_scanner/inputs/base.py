"""Common acquisition lifecycle used by all input modes."""

from abc import ABC, abstractmethod
from pathlib import Path

from property_scanner.core.exceptions import InvalidInputError
from property_scanner.schemas.common import InputTier, NormalizedCapture


class BaseInputAdapter(ABC):
    """Validate source references and normalize them without decoding data."""

    tier: InputTier

    def __init__(self, source_path: Path) -> None:
        self.source_path = Path(source_path).expanduser().absolute()

    def load(self) -> NormalizedCapture:
        """Discover and validate input, returning a lightweight capture."""
        try:
            self.validate()
            files = self._discover_files()
            for file in files:
                with file.open("rb"):
                    pass
            return NormalizedCapture(
                tier=self.tier,
                source_path=self.source_path.resolve(),
                prepared_files=tuple(file.resolve() for file in files),
                metadata={
                    "validation_level": "paths_and_extensions_only",
                    "file_count": len(files),
                    "media_decoded": False,
                },
            )
        except (OSError, RuntimeError) as exc:
            raise InvalidInputError(f"Cannot read input {self.source_path}: {exc}") from exc

    @abstractmethod
    def validate(self) -> None:
        """Reject invalid paths and basic input types with InvalidInputError."""

    def prepare(self) -> NormalizedCapture:
        """Return source references; no copying, conversion, or AI work occurs."""
        return self.load()

    def _require_exists(self) -> None:
        if not self.source_path.exists():
            raise InvalidInputError(f"Input path does not exist: {self.source_path}")

    @abstractmethod
    def _discover_files(self) -> tuple[Path, ...]:
        """Return deterministic references to the source files."""
