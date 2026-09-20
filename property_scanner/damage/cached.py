"""Run damage analysis from calibrated frame caches shared with openings."""
from pathlib import Path
from property_scanner.core.exceptions import PropertyScannerError
from property_scanner.damage.detector import DamageDetector
from property_scanner.damage.models import DAMAGE_MODEL_FILE, DamageConfig, DamageFrame
from property_scanner.damage.vision import DamageVisionModel, UltralyticsYOLEDamageModel
from property_scanner.openings.cached import load_cached_frames
from property_scanner.reconstruction.lidar.diagnostics import write_json
from property_scanner.schemas.result import PropertyScanResult, ResultWarning


def process_cached_damage(output: Path, result: PropertyScanResult, model_dir: Path,
                          config=None, vision_model: DamageVisionModel | None=None):
    config = config or DamageConfig(); output = Path(output)
    if not config.enabled or not result.property.rooms: return None
    frames = [DamageFrame.model_validate(frame.model_dump(exclude={"metadata"})) for frame in load_cached_frames(output, result)]
    if not frames:
        result.warnings.append(ResultWarning(code="DAMAGE_DEPTH_UNAVAILABLE", message="Damage analysis skipped because no calibrated RGB-depth keyframes are cached.")); return None
    if vision_model is None and not (Path(model_dir)/DAMAGE_MODEL_FILE).is_file():
        result.warnings.append(ResultWarning(code="DAMAGE_MODEL_UNAVAILABLE", message="Damage analysis skipped; run python scripts/download_models.py --damage.")); return None
    detector = DamageDetector(config, vision_model or UltralyticsYOLEDamageModel(model_dir, config))
    geometry, regions, flags, scope, warnings = detector.detect(result.property, frames, diagnostics_dir=output/"diagnostics/damage")
    result.property, result.damages = geometry, regions; result.concealed_damage_flags, result.scope_line_items = flags, scope
    result.warnings.extend(warnings)
    if result.processing_info:
        if "damage_detection" not in result.processing_info.modules_used: result.processing_info.modules_used.append("damage_detection")
        result.processing_info.model_versions[config.model_file] = "local checkpoint"
        result.processing_info.metadata["damage"] = {"device": detector.vision_model.device, "frames_processed": len(frames), "regions": len(regions)}
    write_json(output/"damage/damages.json", [item.model_dump(mode="json") for item in regions])
    write_json(output/"damage/scope.json", [item.model_dump(mode="json") for item in scope])
    write_json(output/"damage/detection_summary.json", {"frames_processed": len(frames), "regions": len(regions), "concealed_flags": len(flags), "scope_items": len(scope)})
    return regions


def try_process_cached_damage(output, result, model_dir, config=None, vision_model=None):
    try: return process_cached_damage(output, result, model_dir, config, vision_model)
    except (PropertyScannerError, OSError, RuntimeError, ValueError, ImportError) as exc:
        result.warnings.append(ResultWarning(code="DAMAGE_DETECTION_FAILED", message=str(exc))); return None
