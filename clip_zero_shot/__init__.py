"""High-accuracy zero-shot CLIP inference utilities.

This package provides modular, inference-time techniques for improving
zero-shot classification accuracy using CLIP-style models.
"""

from .clip_loader import load_clip
from .prompts import get_default_prompts, expand_prompts
from .inference import ZeroShotPipeline
from .utils import softmax_with_temp
from .detector import detect_and_classify

__all__ = [
    "load_clip",
    "get_default_prompts",
    "expand_prompts",
    "ZeroShotPipeline",
    "softmax_with_temp",
    "detect_and_classify",
]
