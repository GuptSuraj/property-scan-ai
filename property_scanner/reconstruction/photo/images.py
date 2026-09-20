"""Room discovery, EXIF normalization, quality scoring, and bounded selection."""

from __future__ import annotations

from fractions import Fraction
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from property_scanner.core.exceptions import InvalidInputError, ProcessingError
from property_scanner.reconstruction.lidar.diagnostics import write_json
from property_scanner.reconstruction.photo.models import (
    ImageQualityRecord,
    PhotoConfig,
    PreparedRoomImages,
    RoomSource,
)

SUPPORTED = frozenset({".jpg", ".jpeg", ".png", ".heic"})


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9_.-]+", "_", value.lower()).strip("_.-")
    return result or "room"


def room_label(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip().title() or "Room"


def discover_rooms(root: Path) -> list[RoomSource]:
    """Discover immediate room directories; a flat folder remains one legacy room."""
    root = Path(root)
    direct = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED)
    if direct:
        return [RoomSource(room_id=_slug(root.name), label=room_label(root.name), directory=root, images=direct)]
    rooms = []
    identifiers = set()
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        identifier = _slug(directory.name)
        if identifier in identifiers:
            raise InvalidInputError(f"Room folder names produce duplicate identifier: {identifier}")
        identifiers.add(identifier)
        images = sorted(path for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in SUPPORTED)
        rooms.append(RoomSource(room_id=identifier, label=room_label(directory.name),
                                directory=directory, images=images))
    if not rooms:
        raise InvalidInputError("No room folders found in photo property")
    return rooms


def _register_heic() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError as exc:
        raise InvalidInputError("HEIC support requires the photo extra: pip install -e '.[photo]'") from exc
    register_heif_opener()


def _json_scalar(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, (tuple, list)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, Fraction) or hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _exif(image: Image.Image) -> dict:
    raw = image.getexif()
    wanted = {
        "Make": "camera_make", "Model": "camera_model", "FocalLength": "focal_length",
        "FocalLengthIn35mmFilm": "focal_length_35mm", "Orientation": "orientation",
        "DateTimeOriginal": "timestamp", "DateTime": "timestamp",
    }
    result = {}
    for key, value in raw.items():
        name = ExifTags.TAGS.get(key, str(key))
        if name in wanted and wanted[name] not in result:
            result[wanted[name]] = _json_scalar(value)
    return result


def _inspect(path: Path, config: PhotoConfig) -> tuple[ImageQualityRecord, np.ndarray | None]:
    try:
        if path.suffix.lower() == ".heic":
            _register_heic()
        with Image.open(path) as source:
            source.load()
            metadata = _exif(source)
            image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        if min(width, height) < config.min_image_dimension:
            raise ValueError(f"resolution {width}x{height} is below the minimum")
        rgb = np.asarray(image)
        if rgb.size == 0 or not np.any(rgb):
            raise ValueError("image contains no non-zero pixel data")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness, contrast = float(gray.mean()), float(gray.std())
        features = len(cv2.ORB_create(nfeatures=1000).detect(gray, None))
        score = blur**0.5 + contrast * 2 + min(features, 1000) / 10 - abs(brightness - 127.5) / 10
        reason = None
        if brightness < config.brightness_min:
            reason = "low_light"
        elif brightness > config.brightness_max:
            reason = "overexposed"
        elif blur < config.blur_threshold:
            reason = "blur"
        elif contrast < config.contrast_min:
            reason = "low_contrast"
        record = ImageQualityRecord(source_filename=path.name, readable=True, width=width, height=height,
            blur_score=blur, brightness=brightness, contrast=contrast, feature_count=features,
            quality_score=score, rejection_reason=reason, exif=metadata)
        thumbnail = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        return record, thumbnail
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return ImageQualityRecord(source_filename=path.name, readable=False,
                                  rejection_reason=f"invalid: {exc}"), None


def _same_camera(records: list[ImageQualityRecord]) -> bool:
    signatures = []
    for record in records:
        make, model = record.exif.get("camera_make"), record.exif.get("camera_model")
        if not make or not model:
            return False
        signatures.append((make, model, record.width, record.height,
                           record.exif.get("focal_length"), record.exif.get("focal_length_35mm")))
    return bool(signatures) and len(set(signatures)) == 1


def prepare_room_images(room: RoomSource, output: Path, config: PhotoConfig) -> PreparedRoomImages:
    selected_directory = output / "selected_images"
    selected_directory.mkdir(parents=True, exist_ok=True)
    for stale in selected_directory.glob("*.jpg"):
        stale.unlink()
    inspected = [_inspect(path, config) for path in room.images]
    records = [record for record, _ in inspected]
    usable = [(index, record, thumbnail) for index, (record, thumbnail) in enumerate(inspected)
              if record.readable and record.rejection_reason is None and thumbnail is not None]
    if len(usable) < config.min_images_per_room:
        write_json(output / "image_quality.json", {"room_id": room.room_id,
            "counts": {"images_total": len(records), "images_valid": len(usable),
                       "images_rejected": len(records)-len(usable)},
            "images": [record.model_dump(mode="json") for record in records]})
        raise ProcessingError(f"Room {room.room_id} has {len(usable)} usable photos; need at least {config.min_images_per_room}")

    ranked = sorted(usable, key=lambda item: (-float(item[1].quality_score or 0), item[1].source_filename))
    chosen = []
    duplicate = []
    for item in ranked:
        if any(float(np.mean(np.abs(item[2]-other[2]))) < config.duplicate_threshold for other in chosen):
            duplicate.append(item)
        elif len(chosen) < config.max_images_per_room:
            chosen.append(item)
        else:
            item[1].rejection_reason = "selection_budget"
    while len(chosen) < config.min_images_per_room and duplicate:
        chosen.append(duplicate.pop(0))
    for item in duplicate:
        item[1].rejection_reason = "near_duplicate"
    chosen.sort(key=lambda item: item[1].source_filename)

    selected = []
    for selected_index, (source_index, record, _) in enumerate(chosen):
        source_path = room.images[source_index]
        if source_path.suffix.lower() == ".heic":
            _register_heic()
        with Image.open(source_path) as source:
            normalized = ImageOps.exif_transpose(source).convert("RGB")
            exif = source.getexif()
            orientation_key = next((key for key, name in ExifTags.TAGS.items() if name == "Orientation"), None)
            if orientation_key is not None:
                exif[orientation_key] = 1
            if normalized.width > config.normalized_max_width:
                height = round(normalized.height * config.normalized_max_width / normalized.width)
                normalized = normalized.resize((config.normalized_max_width, height), Image.Resampling.LANCZOS)
            filename = f"{selected_index:06d}.jpg"
            normalized.save(selected_directory / filename, quality=95, exif=exif.tobytes())
        record.selected = True
        record.selected_filename = filename
        record.rejection_reason = None
        selected.append(record)

    warning_codes = []
    if len(selected) == 2:
        warning_codes.append("LOW_PHOTO_COUNT")
    if any(record.rejection_reason == "blur" for record in records):
        warning_codes.append("EXCESSIVE_BLUR")
    if any(record.rejection_reason == "low_light" for record in records):
        warning_codes.append("LOW_LIGHT")
    if duplicate:
        warning_codes.append("REDUNDANT_PHOTOS")
    valid = sum(record.readable for record in records)
    prepared = PreparedRoomImages(room=room, selected_directory=selected_directory, selected=selected,
        records=records, images_total=len(records), images_valid=valid,
        images_rejected=len(records)-len(selected), same_camera_supported=_same_camera(selected),
        warning_codes=warning_codes)
    write_json(output / "image_quality.json", {"room_id": room.room_id,
        "counts": {"images_total": prepared.images_total, "images_valid": prepared.images_valid,
                   "images_rejected": prepared.images_rejected, "images_selected": len(selected)},
        "same_camera_supported": prepared.same_camera_supported,
        "images": [record.model_dump(mode="json") for record in records]})
    return prepared
