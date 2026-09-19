"""Foundation contracts. No computed geometry or measurement defaults exist."""

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class InputTier(StrEnum):
    """Input acquisition modes, not separate processing pipelines."""

    PHOTO = "photo"
    VIDEO = "video"
    LIDAR = "lidar"


class NormalizedCapture(BaseModel):
    """References to validated source files; preparation does not decode media.

    created_at is ingestion time, not the original recording time. Files are
    referenced in place and must remain available for future processing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    capture_id: UUID = Field(default_factory=uuid4)
    tier: InputTier
    source_path: Path
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    prepared_files: tuple[Path, ...] = ()


class PreparationResult(BaseModel):
    """A lifecycle receipt, explicitly not a property scan result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capture: NormalizedCapture
    output_dir: Path
    status: Literal["prepared_not_processed"] = "prepared_not_processed"
    message: str = "Input prepared. Reconstruction and AI processing are not implemented yet."
