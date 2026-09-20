"""One cached metric indoor model, loaded once; MPS with CPU fallback."""
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import numpy as np
from PIL import Image
from property_scanner.core.exceptions import ConfigurationError, ProcessingError
from .models import VideoConfig

CACHE_NAME = 'depth_anything_v2_metric_indoor_small'


def inference_device(torch_module: object) -> str:
    """Choose Apple Metal when it is usable, with a portable CPU fallback."""
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def normalize_metric_config(model_config: object) -> object:
    """Map the pinned checkpoint's legacy metric key to current Transformers."""
    if getattr(model_config, 'depth_estimation', None) == 'metric':
        model_config.depth_estimation_type = 'metric'
    return model_config


class MetricDepthEstimator(ABC):
    device: str = 'unspecified'

    @abstractmethod
    def predict(self, image: Image.Image) -> np.ndarray:
        """Return image-aligned float32 optical Z depth in meters, not inverse depth."""


class DepthAnythingMetric(MetricDepthEstimator):
    def __init__(self, model_dir: Path, config: VideoConfig):
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.config = torch, config
        path=model_dir/CACHE_NAME
        try:
            self.processor=AutoImageProcessor.from_pretrained(path,local_files_only=True,use_fast=False)
            model_config=AutoConfig.from_pretrained(path,local_files_only=True)
            # This pinned model revision used the legacy key
            # ``depth_estimation=metric``. Normalize it before constructing a
            # recent Transformers model so the head uses sigmoid metric depth.
            model_config=normalize_metric_config(model_config)
            self.model=AutoModelForDepthEstimation.from_pretrained(path,config=model_config,local_files_only=True)
        except (OSError, ValueError) as exc:
            raise ConfigurationError('Metric depth weights missing/invalid. Run python scripts/download_models.py --video') from exc
        # The pinned checkpoint predates the Transformers rename from
        # ``depth_estimation`` to ``depth_estimation_type``. Prefer the value
        # stored by that checkpoint instead of the newer class default.
        depth_kind = getattr(self.model.config, 'depth_estimation_type', None)
        if depth_kind != 'metric' or not getattr(self.model.config, 'max_depth', None):
            raise ConfigurationError('Video requires metric indoor weights; relative depth is not supported')
        self.device=inference_device(torch)
        self.model.eval().to(self.device)
        logging.getLogger(__name__).info('Metric depth inference device: %s',self.device)

    def predict(self,image: Image.Image) -> np.ndarray:
        inputs=self.processor(images=image,return_tensors='pt',size={'height':self.config.depth_input_size,'width':self.config.depth_input_size})
        def infer():
            with self.torch.inference_mode():
                output=self.model(**{k:v.to(self.device) for k,v in inputs.items()})
                depth=self.processor.post_process_depth_estimation(output,target_sizes=[(image.height,image.width)])[0]['predicted_depth']
                return depth.detach().cpu().numpy().astype(np.float32)
        try: return infer()
        except RuntimeError as exc:
            if self.device != 'mps': raise ProcessingError(f'Metric depth inference failed: {exc}') from exc
            logging.getLogger(__name__).warning('MPS inference unavailable for this operation; retrying on CPU')
            self.device='cpu'; self.model.to('cpu'); self.torch.mps.empty_cache()
            return infer()
