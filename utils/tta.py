"""
Test-Time Augmentation (TTA) for CLIP-based zero-shot classification.

Applies multiple image transformations to test samples:
- Geometric: flips, rotations, crops
- Photometric: color, brightness, contrast adjustments
- Scale variations: zoom, perspective

Addresses robustness to:
- Camera angles and perspectives
- Lighting variations
- Partial occlusions
- Object scale variations
"""

from typing import List, Tuple
from PIL import Image, ImageOps, ImageEnhance
import random
import math


def _get_base_transforms(image: Image.Image) -> List[Image.Image]:
    """Return base transforms without augmentation-specific randomization."""
    img = image.convert("RGB")
    transforms = [img]

    # Horizontal flip
    transforms.append(ImageOps.mirror(img))

    # Small rotations
    for angle in (-10, -5, 5, 10):
        rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        transforms.append(rotated)

    return transforms


def _get_crop_transforms(image: Image.Image, num_crops: int = 3) -> List[Image.Image]:
    """Apply center crops at varying scales."""
    img = image.convert("RGB")
    w, h = img.size
    crops = []

    crop_scales = [0.85, 0.90, 0.95][:num_crops]
    for scale in crop_scales:
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        cropped = cropped.resize((w, h), Image.Resampling.LANCZOS)
        crops.append(cropped)

    return crops


def _get_color_transforms(image: Image.Image) -> List[Image.Image]:
    """Apply subtle color/brightness/contrast variations."""
    img = image.convert("RGB")
    transforms = []

    # Brightness variations
    for factor in [0.85, 0.90, 1.10, 1.15]:
        enhancer = ImageEnhance.Brightness(img)
        transforms.append(enhancer.enhance(factor))

    # Contrast variations
    for factor in [0.85, 0.90, 1.10, 1.15]:
        enhancer = ImageEnhance.Contrast(img)
        transforms.append(enhancer.enhance(factor))

    # Color saturation
    for factor in [0.80, 0.90, 1.10]:
        enhancer = ImageEnhance.Color(img)
        transforms.append(enhancer.enhance(factor))

    return transforms


def get_tta_transforms(image: Image.Image, num_augmentations: int = 5) -> List[Image.Image]:
    """
    Generate test-time augmented images.

    Applies various transformations to improve robustness by averaging
    predictions across multiple views of the same image.

    Args:
        image: input PIL Image (RGB)
        num_augmentations: total number of augmented images to return
                          (includes original)

    Returns:
        list of PIL Images with augmentations applied

    Example:
        >>> img = Image.open("waste.jpg")
        >>> augmented = get_tta_transforms(img, num_augmentations=8)
        >>> # All images have same size as input
        >>> assert all(a.size == img.size for a in augmented)
    """
    img = image.convert("RGB")
    augmented = [img]  # Include original

    if num_augmentations <= 1:
        return augmented

    # Draw from different categories
    base = _get_base_transforms(img)
    crops = _get_crop_transforms(img, num_crops=2)
    colors = _get_color_transforms(img)

    # Pool all available transforms
    pool = base + crops + colors

    # Randomly select remaining augmentations
    remaining = num_augmentations - 1
    selected = random.sample(pool, min(remaining, len(pool)))
    augmented.extend(selected)

    # Ensure we return exactly num_augmentations images
    if len(augmented) < num_augmentations:
        augmented.extend(random.choices(pool, k=num_augmentations - len(augmented)))

    return augmented[:num_augmentations]


def get_tta_transforms_deterministic(image: Image.Image, num_augmentations: int = 5) -> List[Image.Image]:
    """
    Generate deterministic test-time augmentations (reproducible for research).

    Args:
        image: input PIL Image (RGB)
        num_augmentations: total number of augmented images to return

    Returns:
        list of PIL Images with deterministic transformations
    """
    img = image.convert("RGB")
    augmented = [img]  # Include original

    if num_augmentations <= 1:
        return augmented

    w, h = img.size

    # 1. Horizontal flip
    if len(augmented) < num_augmentations:
        augmented.append(ImageOps.mirror(img))

    # 2. Vertical flip
    if len(augmented) < num_augmentations:
        augmented.append(ImageOps.flip(img))

    # 3. Small rotation (+5 degrees)
    if len(augmented) < num_augmentations:
        augmented.append(img.rotate(5, resample=Image.Resampling.BILINEAR))

    # 4. Small rotation (-5 degrees)
    if len(augmented) < num_augmentations:
        augmented.append(img.rotate(-5, resample=Image.Resampling.BILINEAR))

    # 5. Center crop 90%
    if len(augmented) < num_augmentations:
        crop_w, crop_h = int(w * 0.90), int(h * 0.90)
        left, top = (w - crop_w) // 2, (h - crop_h) // 2
        cropped = img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h))
        augmented.append(cropped)

    # 6. Brightness +10%
    if len(augmented) < num_augmentations:
        enhancer = ImageEnhance.Brightness(img)
        augmented.append(enhancer.enhance(1.1))

    # 7. Brightness -10%
    if len(augmented) < num_augmentations:
        enhancer = ImageEnhance.Brightness(img)
        augmented.append(enhancer.enhance(0.9))

    # 8. Contrast +10%
    if len(augmented) < num_augmentations:
        enhancer = ImageEnhance.Contrast(img)
        augmented.append(enhancer.enhance(1.1))

    return augmented[:num_augmentations]
