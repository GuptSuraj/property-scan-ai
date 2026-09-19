"""Adapter selection: the only tier dispatch point in the foundation."""

from pathlib import Path

from property_scanner.core.exceptions import UnsupportedTierError
from property_scanner.inputs.base import BaseInputAdapter
from property_scanner.inputs.lidar import LidarInputAdapter
from property_scanner.inputs.photo import PhotoInputAdapter
from property_scanner.inputs.video import VideoInputAdapter
from property_scanner.schemas.common import InputTier

ADAPTERS: dict[InputTier, type[BaseInputAdapter]] = {
    InputTier.PHOTO: PhotoInputAdapter,
    InputTier.VIDEO: VideoInputAdapter,
    InputTier.LIDAR: LidarInputAdapter,
}


def select_adapter(tier: str | InputTier, source_path: Path) -> BaseInputAdapter:
    """Resolve a tier, leaving source validation to the selected adapter."""
    try:
        input_tier = InputTier(tier)
    except ValueError as exc:
        raise UnsupportedTierError(
            f"Unsupported tier {tier!r}; choose photo, video, or lidar."
        ) from exc
    return ADAPTERS[input_tier](source_path)
