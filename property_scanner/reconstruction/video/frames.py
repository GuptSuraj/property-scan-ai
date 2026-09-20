"""FFprobe metadata and bounded FFmpeg extraction with deterministic quality selection."""
from fractions import Fraction
import json
import logging
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from property_scanner.core.exceptions import InvalidInputError, ConfigurationError, ProcessingError
from property_scanner.reconstruction.video.models import VideoConfig, VideoMetadata, SelectedFrame
from property_scanner.reconstruction.lidar.diagnostics import write_json

logger = logging.getLogger(__name__)


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ConfigurationError(f"{name} is required. On macOS: brew install ffmpeg colmap")
    return path


def probe_video(path: Path) -> VideoMetadata:
    if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov"}:
        raise InvalidInputError("Video input must be an existing .mp4 or .mov file")
    try:
        process = subprocess.run([executable("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
        if process.returncode:
            raise ValueError("ffprobe could not decode the video container")
        payload = json.loads(process.stdout)
        stream = payload["streams"][0]
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
        duration = float(stream.get("duration", payload.get("format", {}).get("duration", 0)))
        rotation = float(stream.get("tags", {}).get("rotate", 0))
        for side in stream.get("side_data_list", []):
            rotation = float(side.get("rotation", rotation))
        width, height = int(stream["width"]), int(stream["height"])
        if int(round(rotation)) % 180 == 90:
            width, height = height, width
        if min(width, height, fps, duration) <= 0 or not np.isfinite([fps, duration]).all():
            raise ValueError("invalid duration, resolution, or FPS")
        count = int(stream.get("nb_frames", "0")) if stream.get("nb_frames", "0").isdigit() else 0
        return VideoMetadata(width=width, height=height, fps=fps, duration=duration,
                             frame_count=count or round(duration*fps), codec=stream.get("codec_name", "unknown"), rotation_degrees=rotation)
    except (ValueError, KeyError, IndexError, ZeroDivisionError, OSError, subprocess.TimeoutExpired) as exc:
        raise InvalidInputError(f"Unreadable/invalid video: {exc}") from exc


def extract_keyframes(path: Path, directory: Path, config: VideoConfig) -> tuple[VideoMetadata, list[SelectedFrame]]:
    metadata = probe_video(path)
    frames_dir, selected_dir = directory/"frames", directory/"keyframes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)
    # A retried run must never mix old extraction artifacts with the new video.
    for stale in (*frames_dir.glob("*.jpg"), *selected_dir.glob("*.jpg")):
        stale.unlink()
    logger.info("Video duration: %.2f seconds", metadata.duration)
    # FFmpeg autorotation runs before filtering; calibration must refer to the upright frames.
    filters = f"fps={config.extraction_fps},scale='min({config.frame_max_width},iw)':-2"
    with (directory/"ffmpeg.log").open("w") as log:
        try:
            result = subprocess.run([executable("ffmpeg"), "-nostdin", "-y", "-v", "error", "-i", str(path), "-vf", filters,
                "-frames:v", str(config.max_extracted_frames), "-q:v", "2", "-start_number", "0", str(frames_dir/"%06d.jpg")], stdout=log, stderr=log, timeout=config.command_timeout)
        except subprocess.TimeoutExpired as exc:
            raise ProcessingError("Video extraction timed out; inspect video/ffmpeg.log") from exc
    if result.returncode:
        raise InvalidInputError("Video decoding failed; inspect video/ffmpeg.log")
    candidates = sorted(frames_dir.glob("*.jpg"))
    counts = {"frames_total": metadata.frame_count, "frames_extracted": len(candidates), "frames_rejected_blur": 0,
              "frames_rejected_dark": 0, "frames_rejected_duplicate": 0, "frames_rejected_time": 0, "frames_not_selected_budget": 0, "frames_used": 0}
    selected, records, previous = [], [], None
    for index, frame_path in enumerate(candidates):
        rgb = cv2.imread(str(frame_path))
        if rgb is None:
            raise InvalidInputError(f"Extracted frame cannot be decoded: {frame_path.name}")
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        sharpness, brightness = float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())
        small = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(float)
        change = float(np.mean(np.abs(small-previous))) if previous is not None else None
        timestamp = index/config.extraction_fps
        reason = None
        if brightness < config.brightness_threshold: reason = "dark"
        elif sharpness < config.blur_threshold: reason = "blur"
        elif selected and timestamp-selected[-1].timestamp < config.minimum_time_gap: reason = "time"
        elif change is not None and change < config.duplicate_threshold: reason = "duplicate"
        elif len(selected) >= config.max_keyframes: reason = "budget"
        record = SelectedFrame(frame_id=index, timestamp=timestamp, filename=frame_path.name, blur_score=sharpness, brightness=brightness, visual_change=change)
        records.append({**record.model_dump(), "accepted": reason is None, "rejection_reason": reason})
        if reason:
            counts["frames_not_selected_budget" if reason == "budget" else f"frames_rejected_{reason}"] += 1
            continue
        selected.append(record)
        previous = small
        # Same filesystem: no second full-size image copy.
        (selected_dir/frame_path.name).hardlink_to(frame_path)
    counts["frames_used"] = len(selected)
    write_json(directory/"metadata.json", metadata.model_dump())
    write_json(directory/"keyframe_selection.json", {"counts": counts, "frames": records})
    logger.info("Extracted %s frames; selected %s keyframes", len(candidates), len(selected))
    if len(selected) < config.min_keyframes:
        raise ProcessingError("Too few usable video keyframes; inspect keyframe_selection.json or reduce filtering")
    return metadata, selected
