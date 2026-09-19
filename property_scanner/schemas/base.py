"""Shared validation policy and reusable scalar types for the output contract."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegativeNumber = Annotated[FiniteNumber, Field(ge=0)]
ConfidenceScore = Annotated[FiniteNumber, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    """Reject typos; use explicit metadata for extensions within a version."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


def require_unique(values: list[str], label: str) -> None:
    """Reject duplicate identifiers without constraining collection order."""
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")
