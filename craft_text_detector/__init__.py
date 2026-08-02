from __future__ import absolute_import


from enum import Enum
class Device(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'
    MPS = 'mps'


from .craft_utils import load_craftnet_model, load_refinenet_model
from .image_utils import read_image
from .predict import get_prediction
from .file_utils import export_detected_regions, export_extra_results
from .torch_utils import empty_cuda_cache
from .craft import Craft


__version__ = "0.4.4"

__all__ = [
    "read_image",
    "load_craftnet_model",
    "load_refinenet_model",
    "get_prediction",
    "export_detected_regions",
    "export_extra_results",
    "empty_cuda_cache",
    "Craft",
    "Device",
]
