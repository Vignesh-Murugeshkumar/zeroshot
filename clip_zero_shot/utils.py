from typing import List, Sequence, Tuple, Dict
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import math


def softmax_with_temp(scores: Sequence[float], T: float = 1.0) -> List[float]:
    """Compute softmax with temperature scaling.

    Args:
        scores: iterable of logits/scores
        T: temperature > 0
    Returns:
        probabilities summing to 1.0
    """
    scores = np.asarray(scores, dtype=np.float64)
    if T <= 0:
        raise ValueError("Temperature T must be > 0")
    scaled = scores / T
    # Numerical stability
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    probs = exp / exp.sum()
    return probs.tolist()


def tta_transforms(image: Image.Image) -> List[Image.Image]:
    """Return a list of augmented images for Test-Time Augmentation (TTA).

    Includes: original, horizontal flip, small rotations, center crop, color jitter variants.
    """
    imgs = []
    img = image.convert("RGB")
    imgs.append(img)

    # Horizontal flip
    imgs.append(ImageOps.mirror(img))

    # Small rotations
    for angle in (-10, 10):
        imgs.append(img.rotate(angle, resample=Image.BILINEAR, expand=False))

    # Center crop 90%
    w, h = img.size
    crop_w, crop_h = int(w * 0.9), int(h * 0.9)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    imgs.append(img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h)))

    # Color jitter slight variations
    enhancers = [ImageEnhance.Color, ImageEnhance.Brightness, ImageEnhance.Contrast]
    for i, E in enumerate(enhancers):
        enhancer = E(img)
        imgs.append(enhancer.enhance(0.9))
        imgs.append(enhancer.enhance(1.1))

    # Deduplicate by size and bytes is expensive; assume small list
    return imgs


def aggregate_scores_mean(scores_list: List[List[float]]) -> List[float]:
    """Aggregate per-prompt or per-augmentation scores by mean.

    Args:
        scores_list: list of score vectors (N_prompts or N_classes)
    Returns:
        aggregated score vector
    """
    arr = np.asarray(scores_list, dtype=np.float64)
    return np.mean(arr, axis=0).tolist()


def aggregate_scores_trimmed_mean(scores_list: List[List[float]], trim_frac: float = 0.1) -> List[float]:
    """Aggregates scores by trimmed mean (remove top/bottom trim_frac fraction before averaging).

    Args:
        scores_list: list of score vectors
        trim_frac: fraction to trim from each side (0 <= trim_frac < 0.5)
    """
    arr = np.asarray(scores_list, dtype=np.float64)
    n = arr.shape[0]
    k = int(math.floor(n * trim_frac))
    if k == 0:
        return aggregate_scores_mean(scores_list)

    # sort along 0 axis for each column
    sorted_arr = np.sort(arr, axis=0)
    trimmed = sorted_arr[k:n - k, :]
    return np.mean(trimmed, axis=0).tolist()
