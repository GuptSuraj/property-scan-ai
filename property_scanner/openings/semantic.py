"""One locally cached SegFormer abstraction with MPS then CPU execution."""

from abc import ABC, abstractmethod
import logging
import warnings
from pathlib import Path
import numpy as np
from PIL import Image

from property_scanner.core.exceptions import ConfigurationError, ProcessingError
from property_scanner.openings.models import OpeningConfig, SEMANTIC_CACHE_NAME


class SemanticSurfaceDetector(ABC):
    device: str = "unspecified"

    @abstractmethod
    def predict(self, image: Image.Image) -> dict[str, np.ndarray]:
        """Return image-aligned probability maps for door, window, and wall."""


class SegFormerSurfaceDetector(SemanticSurfaceDetector):
    def __init__(self, model_dir: Path, config: OpeningConfig):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        self.torch, self.config = torch, config
        path = Path(model_dir) / SEMANTIC_CACHE_NAME
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*reduce_labels.*")
                self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
            self.model = SegformerForSemanticSegmentation.from_pretrained(path, local_files_only=True)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                "Opening segmentation weights are missing. Run python scripts/download_models.py --openings"
            ) from exc
        self.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.model.eval().to(self.device)
        labels = {int(index): str(label).lower() for index, label in self.model.config.id2label.items()}
        self.class_ids = {
            "door": [index for index, label in labels.items() if label == "door"],
            "window": [index for index, label in labels.items() if "window" in label],
            "wall": [index for index, label in labels.items() if label == "wall"],
        }
        if not self.class_ids["door"] or not self.class_ids["window"]:
            raise ConfigurationError("The cached segmentation model lacks ADE20K door/window labels")
        logging.getLogger(__name__).info("Opening semantic inference device: %s", self.device)

    def _infer(self, image: Image.Image) -> dict[str, np.ndarray]:
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt",
            size={"height": self.config.semantic_input_size, "width": self.config.semantic_input_size})
        with torch.inference_mode():
            logits = self.model(**{key: value.to(self.device) for key, value in inputs.items()}).logits
            logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width),
                mode="bilinear", align_corners=False)
            probabilities = logits.softmax(dim=1)[0].detach().cpu().numpy()
        return {name: probabilities[indices].sum(axis=0).astype(np.float32)
                if indices else np.zeros((image.height, image.width), dtype=np.float32)
                for name, indices in self.class_ids.items()}

    def predict(self, image: Image.Image) -> dict[str, np.ndarray]:
        try:
            return self._infer(image)
        except RuntimeError as exc:
            if self.device != "mps":
                raise ProcessingError(f"Opening segmentation failed: {exc}") from exc
            logging.getLogger(__name__).warning("Opening segmentation MPS failure; retrying on CPU")
            self.device = "cpu"
            self.model.to("cpu")
            self.torch.mps.empty_cache()
            return self._infer(image)
