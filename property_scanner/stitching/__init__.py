"""Rigid, evidence-based multi-room property stitching."""

from property_scanner.stitching.engine import MultiRoomStitcher, StitchingResult
from property_scanner.stitching.models import RoomConnectionEvidence, StitchingConfig

__all__ = ["MultiRoomStitcher", "StitchingResult", "RoomConnectionEvidence", "StitchingConfig"]
