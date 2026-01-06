"""
Research-Grade CLIP Classifier with Prompt Ensembling and Test-Time Augmentation.

Implements:
- Prompt ensembling with mean/trimmed-mean aggregation
- Test-Time Augmentation (TTA) with multiple image transforms
- Temperature scaling for calibration
- Configurable batch inference
- Performance profiling (inference time tracking)
"""

from __future__ import annotations

from typing import Dict, List, Union, Optional
from dataclasses import dataclass, field
from collections import Counter
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import time

# ============================================================================
# Configuration Classes
# ============================================================================
@dataclass
class ClipConfig:
    """Configuration for CLIP model and inference."""
    model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
    use_fp16: bool = True
    text_batch: int = 64
    image_batch: int = 32
    temperature: float = 0.1
    aggregation_method: str = "mean"  # "mean" or "trimmed_mean"
    trimmed_mean_fraction: float = 0.1


@dataclass
class ClassificationResult:
    """Result of image classification."""
    ranked: List[tuple]  # [(class_name, score), ...]
    all_scores: Dict[str, float]
    logits_raw: torch.Tensor
    inference_time_ms: float = 0.0
    num_prompts_used: int = 0
    num_augmentations_used: int = 1


# ============================================================================
# CLIP Waste Classifier
# ============================================================================
class ClipWasteClassifier:
    """
    Zero-shot CLIP classifier with prompt ensembling and TTA support.

    Features:
    - Caches text embeddings for all prompts
    - Supports prompt-level ensemble aggregation (mean/trimmed-mean)
    - Supports test-time augmentation (TTA)
    - Tracks inference time and prompts used
    - Optional fp16 inference on GPU

    Example:
        >>> from prompts.waste_prompts import build_prompt_bank, PromptSetConfig
        >>> cfg_prompt = PromptSetConfig(size="medium")
        >>> prompt_bank = build_prompt_bank(config=cfg_prompt)
        >>> clf_cfg = ClipConfig(device="cuda", use_fp16=True)
        >>> clf = ClipWasteClassifier(prompt_bank, config=clf_cfg)
        >>> result = clf.classify_image(image, use_tta=True, tta_augmentations=5)
        >>> print(result.ranked[0])  # Top prediction
    """

    def __init__(self, prompt_bank: Dict[str, List[str]], config: Optional[ClipConfig] = None):
        self.config = config or ClipConfig()
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # Load model
        self.model = CLIPModel.from_pretrained(self.config.model_name)
        if self.config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)

        # Store prompt bank and cache embeddings
        self.prompt_bank = prompt_bank
        self._owners: List[str] = []  # Maps embedding idx to class name
        self._text_feats: torch.Tensor  # Cached text embeddings
        self._cache_text_embeddings()

        # Statistics
        self._inference_times: List[float] = []

    def _cache_text_embeddings(self) -> None:
        """Precompute and cache text embeddings for all prompts."""
        prompts: List[str] = []
        owners: List[str] = []

        for cls, plist in self.prompt_bank.items():
            prompts.extend(plist)
            owners.extend([cls] * len(plist))

        self._owners = owners

        # Batch encode prompts
        feats = []
        for i in range(0, len(prompts), self.config.text_batch):
            batch = prompts[i : i + self.config.text_batch]
            ti = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                tf = self.model.get_text_features(**ti)
                tf = F.normalize(tf, p=2, dim=-1)

            feats.append(tf)

        self._text_feats = torch.cat(feats, dim=0)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a single image to normalized embedding."""
        ii = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            imf = self.model.get_image_features(**ii)
            imf = F.normalize(imf, p=2, dim=-1)

        return imf

    def _aggregate_prompt_scores(
        self, logits: torch.Tensor
    ) -> Dict[str, float]:
        """
        Aggregate per-prompt similarity scores to per-class scores.

        Args:
            logits: shape (N_prompts,) - raw similarity scores per prompt

        Returns:
            dict: class_name -> aggregated_score
        """
        class_scores: Dict[str, float] = {}
        counts = Counter(self._owners)

        # Sum scores per class
        for logit, cls in zip(logits.tolist(), self._owners):
            class_scores[cls] = class_scores.get(cls, 0.0) + logit

        # Aggregate (mean or trimmed mean)
        if self.config.aggregation_method == "trimmed_mean":
            # Group logits by class, compute trimmed mean
            class_logits: Dict[str, List[float]] = {}
            for logit, cls in zip(logits.tolist(), self._owners):
                if cls not in class_logits:
                    class_logits[cls] = []
                class_logits[cls].append(logit)

            class_scores = {}
            for cls, cls_logits_list in class_logits.items():
                trimmed = self._trimmed_mean(cls_logits_list, self.config.trimmed_mean_fraction)
                class_scores[cls] = trimmed
        else:
            # Standard mean
            for cls in class_scores:
                class_scores[cls] /= max(1, counts[cls])

        return class_scores

    @staticmethod
    def _trimmed_mean(values: List[float], trim_fraction: float) -> float:
        """Compute trimmed mean by removing top/bottom trim_fraction."""
        if trim_fraction <= 0 or len(values) < 2:
            return sum(values) / len(values)

        import math
        values_sorted = sorted(values)
        n = len(values)
        k = int(math.floor(n * trim_fraction))
        trimmed = values_sorted[k : n - k]
        return sum(trimmed) / len(trimmed) if trimmed else sum(values) / n

    def classify_image(
        self,
        image: Union[str, Image.Image],
        top_k: int = 3,
        use_tta: bool = False,
        tta_augmentations: int = 5,
    ) -> ClassificationResult:
        """
        Classify a single image with optional TTA.

        Args:
            image: PIL Image or path to image
            top_k: number of top predictions to return
            use_tta: whether to use test-time augmentation
            tta_augmentations: number of augmentations if use_tta=True

        Returns:
            ClassificationResult with predictions and metadata
        """
        start_time = time.time()

        # Load image
        pil = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

        # Classify with optional TTA
        if use_tta:
            result_scores = self._classify_with_tta(pil, tta_augmentations)
        else:
            # Single forward pass
            imf = self._encode_image(pil)
            logits = (imf @ self._text_feats.T) / self.config.temperature
            result_scores = self._aggregate_prompt_scores(logits[0])

        # Rank predictions
        ranked = sorted(result_scores.items(), key=lambda x: x[1], reverse=True)

        # Package result
        elapsed_ms = (time.time() - start_time) * 1000
        result = ClassificationResult(
            ranked=ranked[:top_k],
            all_scores=result_scores,
            logits_raw=torch.tensor(list(result_scores.values())),
            inference_time_ms=elapsed_ms,
            num_prompts_used=len(self._owners),
            num_augmentations_used=tta_augmentations if use_tta else 1,
        )

        self._inference_times.append(elapsed_ms)
        return result

    def _classify_with_tta(self, image: Image.Image, num_augmentations: int) -> Dict[str, float]:
        """
        Classify by averaging predictions across augmented images.

        Args:
            image: PIL Image
            num_augmentations: number of augmentations to apply

        Returns:
            dict: aggregated class scores
        """
        from ..utils.tta import get_tta_transforms

        augmented_images = get_tta_transforms(image, num_augmentations)

        all_scores: List[Dict[str, float]] = []

        for aug_img in augmented_images:
            imf = self._encode_image(aug_img)
            logits = (imf @ self._text_feats.T) / self.config.temperature
            scores = self._aggregate_prompt_scores(logits[0])
            all_scores.append(scores)

        # Average across augmentations
        aggregated = {}
        for cls in all_scores[0].keys():
            scores_for_cls = [s[cls] for s in all_scores]
            aggregated[cls] = sum(scores_for_cls) / len(scores_for_cls)

        return aggregated

    def classify_batch(
        self,
        images: List[Image.Image],
        top_k: int = 3,
        use_tta: bool = False,
        tta_augmentations: int = 5,
    ) -> List[ClassificationResult]:
        """
        Classify multiple images.

        Args:
            images: list of PIL Images
            top_k: number of top predictions per image
            use_tta: whether to use test-time augmentation
            tta_augmentations: number of augmentations if use_tta=True

        Returns:
            list of ClassificationResult objects
        """
        return [self.classify_image(img, top_k, use_tta, tta_augmentations) for img in images]

    def rebuild_with_new_prompts(self, new_prompt_bank: Dict[str, List[str]]) -> None:
        """Swap in a new prompt bank and recache text embeddings."""
        self.prompt_bank = new_prompt_bank
        self._cache_text_embeddings()

    def get_stats(self) -> Dict[str, float]:
        """Return inference statistics."""
        if not self._inference_times:
            return {"mean_inference_time_ms": 0.0, "num_inferences": 0}

        import statistics
        return {
            "mean_inference_time_ms": statistics.mean(self._inference_times),
            "median_inference_time_ms": statistics.median(self._inference_times),
            "min_inference_time_ms": min(self._inference_times),
            "max_inference_time_ms": max(self._inference_times),
            "num_inferences": len(self._inference_times),
        }
