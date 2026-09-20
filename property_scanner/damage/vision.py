"""Optional local YOLOE damage segmentation with MPS-to-CPU fallback."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from property_scanner.core.exceptions import ConfigurationError, ProcessingError
from property_scanner.damage.models import DamageConfig, DamagePrediction
from property_scanner.schemas.damage import DamageType


class DamageVisionModel(ABC):
    device: str = "unspecified"

    @abstractmethod
    def predict(self, image: Image.Image) -> list[DamagePrediction]:
        """Return image-aligned damage masks without metric claims."""


def _damage_type(label: str) -> DamageType:
    """Conservatively map only explicit damage vocabulary labels.

    The prompt-free checkpoint contains thousands of labels (including items
    such as ``water bottle`` and ``molding``), so substring matching would
    create avoidable phantom damage.
    """
    label = " ".join(label.lower().replace("_", " ").split())
    aliases = {
        "crack": DamageType.CRACK,
        "wall crack": DamageType.CRACK,
        "crack in wall": DamageType.CRACK,
        "water damage": DamageType.WATER_DAMAGE,
        "water stain": DamageType.MOISTURE_STAIN,
        "moisture stain": DamageType.MOISTURE_STAIN,
        "moisture": DamageType.MOISTURE_STAIN,
        "stain": DamageType.MOISTURE_STAIN,
        "mold": DamageType.MOLD,
        "hole": DamageType.HOLE,
        "hole in wall": DamageType.HOLE,
        "peel": DamageType.PEELING_PAINT,
        "peeling paint": DamageType.PEELING_PAINT,
        "damage": DamageType.SURFACE_DAMAGE,
        "damaged wall": DamageType.SURFACE_DAMAGE,
        "surface damage": DamageType.SURFACE_DAMAGE,
    }
    return aliases.get(label, DamageType.UNKNOWN)


class UltralyticsYOLEDamageModel(DamageVisionModel):
    """YOLOE text-prompt segmentation loaded only from the configured model cache."""

    def __init__(self, model_dir: Path, config: DamageConfig) -> None:
        try:
            import torch
            from ultralytics import YOLOE
        except ImportError as exc:
            raise ConfigurationError("Install damage dependencies: pip install -e '.[damage]'") from exc
        path = Path(model_dir) / config.model_file
        if not path.is_file():
            raise ConfigurationError("Damage model unavailable; run python scripts/download_models.py --damage")
        self.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        try:
            self.model = YOLOE(path, verbose=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ConfigurationError(f"Cannot load cached damage model: {exc}") from exc
        self.config = config
        names = self.model.names.items() if isinstance(self.model.names, dict) else enumerate(self.model.names)
        self.damage_classes = {int(index): _damage_type(str(label)) for index, label in names
                               if _damage_type(str(label)) != DamageType.UNKNOWN}
        if not self.damage_classes:
            raise ConfigurationError("Cached damage model has no supported damage vocabulary")

    def predict(self, image: Image.Image) -> list[DamagePrediction]:
        try:
            result = self.model.predict(np.asarray(image.convert("RGB")), device=self.device,
                                        conf=self.config.semantic_min_score,
                                        classes=sorted(self.damage_classes), verbose=False)[0]
            if result.masks is None or result.boxes is None:
                return []
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            output = []
            for mask, class_id, score in zip(masks, classes, scores):
                resized = np.asarray(Image.fromarray(mask.astype(np.float32)).resize(
                    image.size, Image.Resampling.BILINEAR)) >= 0.5
                damage_type = self.damage_classes.get(int(class_id), DamageType.UNKNOWN)
                if damage_type != DamageType.UNKNOWN:
                    output.append(DamagePrediction(damage_type, resized, float(score)))
            return output
        except (RuntimeError, ValueError, IndexError) as exc:
            raise ProcessingError(f"Damage segmentation failed: {exc}") from exc
