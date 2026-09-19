"""Acquisition provenance, shared by every input mode."""

from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue

from property_scanner.schemas.base import ContractModel, NonEmptyText
from property_scanner.schemas.common import InputTier


class SourceType(StrEnum):
    DIRECTORY = "directory"
    FILE = "file"
    URI = "uri"
    UNKNOWN = "unknown"


class CaptureMetadata(ContractModel):
    """Actual provenance only; timestamps and device details are never invented."""

    capture_id: UUID = Field(description="Identifier linking this result to its normalized capture.")
    tier: InputTier
    source_type: SourceType = SourceType.UNKNOWN
    source_reference: NonEmptyText | None = Field(default=None, description="Portable path or URI; not resolved or opened during validation.")
    device_name: NonEmptyText | None = None
    device_model: NonEmptyText | None = None
    operating_system: NonEmptyText | None = None
    capture_timestamp: AwareDatetime | None = None
    processing_timestamp: AwareDatetime | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
