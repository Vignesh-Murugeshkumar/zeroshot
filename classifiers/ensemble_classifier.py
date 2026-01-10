"""
Multi-Model CLIP Ensemble for Maximum Zero-Shot Accuracy.

This module implements model ensembling across multiple CLIP variants:
- openai/clip-vit-base-patch32 (baseline, fast)
- openai/clip-vit-large-patch14 (higher capacity)
- laion/CLIP-ViT-L-14-336px (OpenCLIP, higher resolution)

Ensembling Strategy:
- Each model produces normalized similarity scores per class
- Scores are aggregated using configurable methods:
    * mean: simple average across models
    * weighted: weighted average with learned/tuned weights
    * max: take maximum score per class
    * vote: majority voting on top-1 predictions

Performance Benefits:
- Reduces model-specific biases
- Captures different visual features (resolution, training data)
- Typically improves accuracy by 2-5% over single model
- Essential for reaching ~97% on controlled datasets

Trade-offs:
- 2-3x slower inference (multiple forward passes)
- 2-3x higher memory usage
- Requires more GPU memory or CPU fallback
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from classifiers.clip_classifier import ClipWasteClassifier, ClipConfig, ClassificationResult


@dataclass
class EnsembleConfig:
    """Configuration for multi-model ensembling."""
    model_names: List[str]
    device: Optional[str] = None
    use_fp16: bool = True
    aggregation_method: Literal["mean", "weighted", "max", "vote"] = "mean"
    model_weights: Optional[List[float]] = None  # For weighted aggregation
    temperature: float = 0.1


class MultiModelEnsemble:
    """
    Ensemble classifier that combines multiple CLIP models.
    
    Example:
        >>> from prompts.waste_prompts import build_prompt_bank, PromptSetConfig
        >>> cfg_prompt = PromptSetConfig(size="large")
        >>> prompt_bank = build_prompt_bank(config=cfg_prompt)
        >>> 
        >>> ensemble_cfg = EnsembleConfig(
        ...     model_names=[
        ...         "openai/clip-vit-base-patch32",
        ...         "openai/clip-vit-large-patch14"
        ...     ],
        ...     aggregation_method="mean"
        ... )
        >>> 
        >>> ensemble = MultiModelEnsemble(prompt_bank, config=ensemble_cfg)
        >>> result = ensemble.classify_image(image, use_tta=True)
        >>> print(f"Top prediction: {result.ranked[0]}")
    """
    
    def __init__(
        self, 
        prompt_bank: Dict[str, List[str]], 
        config: Optional[EnsembleConfig] = None
    ):
        self.config = config or EnsembleConfig(
            model_names=["openai/clip-vit-base-patch32"]
        )
        
        # Initialize model weights
        if self.config.model_weights is None:
            # Equal weights by default
            self.config.model_weights = [1.0] * len(self.config.model_names)
        
        # Normalize weights
        total_weight = sum(self.config.model_weights)
        self.config.model_weights = [w / total_weight for w in self.config.model_weights]
        
        # Create individual classifiers for each model
        self.classifiers: List[ClipWasteClassifier] = []
        
        print(f"Initializing ensemble with {len(self.config.model_names)} models...")
        for idx, model_name in enumerate(self.config.model_names):
            print(f"  [{idx+1}/{len(self.config.model_names)}] Loading {model_name}...")
            
            clip_config = ClipConfig(
                model_name=model_name,
                device=self.config.device,
                use_fp16=self.config.use_fp16,
                temperature=self.config.temperature,
            )
            
            classifier = ClipWasteClassifier(prompt_bank, config=clip_config)
            self.classifiers.append(classifier)
        
        print("Ensemble initialization complete.")
        self.class_names = list(prompt_bank.keys())
    
    def classify_image(
        self,
        image: Image.Image,
        top_k: int = 3,
        use_tta: bool = False,
        tta_augmentations: int = 5,
    ) -> ClassificationResult:
        """
        Classify image using model ensemble.
        
        Args:
            image: PIL Image
            top_k: number of top predictions to return
            use_tta: whether to use test-time augmentation
            tta_augmentations: number of augmentations if use_tta=True
        
        Returns:
            ClassificationResult with ensembled predictions
        """
        start_time = time.time()
        
        # Get predictions from each model
        all_scores: List[Dict[str, float]] = []
        all_top1_predictions: List[str] = []
        
        for classifier in self.classifiers:
            result = classifier.classify_image(
                image,
                top_k=len(self.class_names),
                use_tta=use_tta,
                tta_augmentations=tta_augmentations
            )
            all_scores.append(result.all_scores)
            all_top1_predictions.append(result.ranked[0][0])
        
        # Aggregate scores based on method
        if self.config.aggregation_method == "mean":
            final_scores = self._aggregate_mean(all_scores)
        
        elif self.config.aggregation_method == "weighted":
            final_scores = self._aggregate_weighted(all_scores)
        
        elif self.config.aggregation_method == "max":
            final_scores = self._aggregate_max(all_scores)
        
        elif self.config.aggregation_method == "vote":
            final_scores = self._aggregate_vote(all_scores, all_top1_predictions)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
        
        # Rank predictions
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Package result
        elapsed_ms = (time.time() - start_time) * 1000
        result = ClassificationResult(
            ranked=ranked[:top_k],
            all_scores=final_scores,
            logits_raw=torch.tensor(list(final_scores.values())),
            inference_time_ms=elapsed_ms,
            num_prompts_used=self.classifiers[0]._text_feats.shape[0],
            num_augmentations_used=tta_augmentations if use_tta else 1,
        )
        
        return result
    
    def _aggregate_mean(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Simple average across all models."""
        aggregated = {}
        for cls in self.class_names:
            scores = [model_scores[cls] for model_scores in all_scores]
            aggregated[cls] = sum(scores) / len(scores)
        return aggregated
    
    def _aggregate_weighted(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Weighted average using model_weights."""
        aggregated = {}
        for cls in self.class_names:
            weighted_sum = 0.0
            for model_scores, weight in zip(all_scores, self.config.model_weights):
                weighted_sum += model_scores[cls] * weight
            aggregated[cls] = weighted_sum
        return aggregated
    
    def _aggregate_max(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Take maximum score per class across models."""
        aggregated = {}
        for cls in self.class_names:
            scores = [model_scores[cls] for model_scores in all_scores]
            aggregated[cls] = max(scores)
        return aggregated
    
    def _aggregate_vote(
        self, 
        all_scores: List[Dict[str, float]], 
        all_top1_predictions: List[str]
    ) -> Dict[str, float]:
        """
        Voting-based aggregation.
        
        For the class with most votes, boost its averaged score.
        This handles cases where models disagree.
        """
        from collections import Counter
        
        # Count votes
        vote_counts = Counter(all_top1_predictions)
        most_voted_class = vote_counts.most_common(1)[0][0]
        
        # Average scores but boost the winner
        aggregated = self._aggregate_mean(all_scores)
        
        # Boost most voted class slightly
        max_score = max(aggregated.values())
        aggregated[most_voted_class] = max(aggregated[most_voted_class], max_score * 1.1)
        
        return aggregated
    
    def classify_batch(
        self,
        images: List[Image.Image],
        top_k: int = 3,
        use_tta: bool = False,
        tta_augmentations: int = 5,
    ) -> List[ClassificationResult]:
        """Classify multiple images with ensemble."""
        return [
            self.classify_image(img, top_k, use_tta, tta_augmentations) 
            for img in images
        ]
    
    def get_model_info(self) -> Dict:
        """Return information about the ensemble."""
        return {
            "num_models": len(self.classifiers),
            "model_names": self.config.model_names,
            "model_weights": self.config.model_weights,
            "aggregation_method": self.config.aggregation_method,
            "temperature": self.config.temperature,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
        }
