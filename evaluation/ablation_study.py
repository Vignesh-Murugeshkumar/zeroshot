"""
Comprehensive Ablation Study for Zero-Shot CLIP Waste Classification.

This script evaluates the impact of each optimization technique:
1. Baseline: Single prompt per class
2. Prompt ensemble: Multiple prompts per class (small/medium/large)
3. Prompt ensemble + TTA: Add test-time augmentation
4. Multi-model ensemble: Combine multiple CLIP variants
5. Full system: All techniques combined

Metrics tracked:
- Accuracy, precision, recall, F1-score
- Per-class performance
- Inference time
- Confusion matrices

Expected Results (TrashNet):
- Baseline (single prompt): ~75-80%
- Prompt ensemble (medium): ~85-90%
- Prompt ensemble + TTA: ~90-93%
- Multi-model ensemble: ~93-95%
- Full system (all optimizations): ~95-97%
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from classifiers.clip_classifier import ClipWasteClassifier, ClipConfig
from classifiers.ensemble_classifier import MultiModelEnsemble, EnsembleConfig
from prompts.waste_prompts import build_prompt_bank, PromptSetConfig
from utils.tta import get_tta_transforms_research


@dataclass
class AblationResult:
    """Result of single ablation experiment."""
    name: str
    description: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]
    inference_time_ms: float
    num_samples: int
    config: Dict


def load_dataset(
    dataset_path: str,
    class_names: List[str]
) -> Tuple[List[Image.Image], List[str]]:
    """
    Load images and labels from directory structure.
    
    Expected structure:
        dataset_path/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
    
    Args:
        dataset_path: path to dataset root
        class_names: list of class names to load
    
    Returns:
        tuple of (images, labels)
    """
    dataset_root = Path(dataset_path)
    images = []
    labels = []
    
    for class_name in class_names:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        for img_path in class_dir.glob("*.[jp][pn]g"):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                labels.append(class_name)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
    
    print(f"Loaded {len(images)} images from {len(set(labels))} classes")
    return images, labels


def evaluate_configuration(
    classifier,
    images: List[Image.Image],
    labels: List[str],
    class_names: List[str],
    use_tta: bool = False,
    tta_strategy: str = "medium",
    description: str = "",
) -> AblationResult:
    """
    Evaluate classifier on dataset.
    
    Args:
        classifier: ClipWasteClassifier or MultiModelEnsemble instance
        images: list of PIL images
        labels: list of ground truth labels
        class_names: list of all class names
        use_tta: whether to use test-time augmentation
        tta_strategy: TTA strategy (light/medium/heavy)
        description: human-readable description
    
    Returns:
        AblationResult with metrics
    """
    predictions = []
    inference_times = []
    
    print(f"Evaluating: {description}")
    for img in tqdm(images, desc="Classifying"):
        start = time.time()
        
        if use_tta:
            if hasattr(classifier, 'classify_image'):
                # Single model with TTA
                from utils.tta import get_tta_transforms_research
                views = get_tta_transforms_research(img, strategy=tta_strategy)
                
                # Get predictions for all views
                all_scores = []
                for view in views:
                    result = classifier.classify_image(view, top_k=len(class_names))
                    all_scores.append(result.all_scores)
                
                # Average scores across views
                avg_scores = {}
                for cls in class_names:
                    avg_scores[cls] = sum(s[cls] for s in all_scores) / len(all_scores)
                
                pred_class = max(avg_scores.items(), key=lambda x: x[1])[0]
            else:
                # Ensemble with TTA
                result = classifier.classify_image(img, use_tta=True, tta_augmentations=len(views))
                pred_class = result.ranked[0][0]
        else:
            result = classifier.classify_image(img, top_k=1)
            pred_class = result.ranked[0][0]
        
        elapsed = (time.time() - start) * 1000
        predictions.append(pred_class)
        inference_times.append(elapsed)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_names, average='macro', zero_division=0
    )
    
    # Per-class F1
    _, _, per_class_f1_scores, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_names, average=None, zero_division=0
    )
    per_class_f1 = dict(zip(class_names, per_class_f1_scores.tolist()))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=class_names)
    
    result = AblationResult(
        name=description.lower().replace(" ", "_"),
        description=description,
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        per_class_f1=per_class_f1,
        confusion_matrix=cm.tolist(),
        inference_time_ms=np.mean(inference_times),
        num_samples=len(images),
        config={}
    )
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Inference time: {result.inference_time_ms:.1f}ms")
    print()
    
    return result


def run_ablation_study(
    dataset_path: str,
    output_dir: str = "results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, AblationResult]:
    """
    Run complete ablation study.
    
    Experiments:
    1. Baseline: Single generic prompt per class
    2. Small prompt set (20 prompts/class)
    3. Medium prompt set (50 prompts/class)
    4. Large prompt set (100 prompts/class)
    5. Large + TTA (light)
    6. Large + TTA (medium)
    7. Large + TTA (heavy)
    8. Multi-model ensemble (no TTA)
    9. Multi-model ensemble + TTA
    
    Args:
        dataset_path: path to dataset with class subdirectories
        output_dir: directory to save results
        device: torch device
    
    Returns:
        dictionary of experiment_name -> AblationResult
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define class names
    class_names = ["plastic", "paper", "metal", "glass", "organic", "e-waste"]
    
    # Load dataset
    print("=" * 70)
    print("Loading dataset...")
    print("=" * 70)
    images, labels = load_dataset(dataset_path, class_names)
    
    if len(images) == 0:
        raise ValueError(f"No images found in {dataset_path}")
    
    results = {}
    
    # Experiment 1: Baseline (single prompt per class)
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Baseline (single generic prompt)")
    print("=" * 70)
    baseline_prompts = {cls: [f"{cls} waste"] for cls in class_names}
    config = ClipConfig(device=device, temperature=0.1)
    classifier = ClipWasteClassifier(baseline_prompts, config=config)
    results["baseline"] = evaluate_configuration(
        classifier, images, labels, class_names,
        description="Baseline (single prompt)"
    )
    
    # Experiment 2-4: Prompt ensemble (small/medium/large)
    for size in ["small", "medium", "large"]:
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: Prompt Ensemble ({size})")
        print("=" * 70)
        
        prompt_config = PromptSetConfig(size=size)
        prompt_bank = build_prompt_bank(config=prompt_config)
        
        print(f"Prompt counts per class:")
        for cls, prompts in prompt_bank.items():
            print(f"  {cls}: {len(prompts)} prompts")
        
        config = ClipConfig(device=device, temperature=0.1)
        classifier = ClipWasteClassifier(prompt_bank, config=config)
        results[f"prompt_ensemble_{size}"] = evaluate_configuration(
            classifier, images, labels, class_names,
            description=f"Prompt Ensemble ({size})"
        )
    
    # Experiment 5-7: Large prompts + TTA
    print("\n" + "=" * 70)
    print("Building large prompt set for TTA experiments...")
    print("=" * 70)
    prompt_config_large = PromptSetConfig(size="large")
    prompt_bank_large = build_prompt_bank(config=prompt_config_large)
    config = ClipConfig(device=device, temperature=0.1)
    classifier_large = ClipWasteClassifier(prompt_bank_large, config=config)
    
    for tta_strategy in ["light", "medium", "heavy"]:
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: Large Prompts + TTA ({tta_strategy})")
        print("=" * 70)
        results[f"large_tta_{tta_strategy}"] = evaluate_configuration(
            classifier_large, images, labels, class_names,
            use_tta=True, tta_strategy=tta_strategy,
            description=f"Large + TTA ({tta_strategy})"
        )
    
    # Experiment 8: Multi-model ensemble
    print("\n" + "=" * 70)
    print("EXPERIMENT: Multi-Model Ensemble (no TTA)")
    print("=" * 70)
    ensemble_config = EnsembleConfig(
        model_names=[
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ],
        device=device,
        aggregation_method="mean"
    )
    ensemble = MultiModelEnsemble(prompt_bank_large, config=ensemble_config)
    results["multi_model_ensemble"] = evaluate_configuration(
        ensemble, images, labels, class_names,
        description="Multi-Model Ensemble"
    )
    
    # Experiment 9: Multi-model ensemble + TTA
    print("\n" + "=" * 70)
    print("EXPERIMENT: Multi-Model Ensemble + TTA (medium)")
    print("=" * 70)
    results["full_system"] = evaluate_configuration(
        ensemble, images, labels, class_names,
        use_tta=True, tta_strategy="medium",
        description="Full System (Ensemble + TTA)"
    )
    
    # Save results
    results_json = {}
    for name, result in results.items():
        results_json[name] = {
            "description": result.description,
            "accuracy": result.accuracy,
            "precision_macro": result.precision_macro,
            "recall_macro": result.recall_macro,
            "f1_macro": result.f1_macro,
            "per_class_f1": result.per_class_f1,
            "inference_time_ms": result.inference_time_ms,
            "num_samples": result.num_samples,
        }
    
    results_file = output_path / "ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<40} {'Accuracy':>10} {'F1':>10} {'Time (ms)':>12}")
    print("-" * 70)
    for name, result in results.items():
        print(f"{result.description:<40} {result.accuracy:>10.4f} {result.f1_macro:>10.4f} {result.inference_time_ms:>12.1f}")
    print("=" * 70)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ablation study for zero-shot waste classification"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory with class subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ablation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    results = run_ablation_study(
        dataset_path=args.dataset,
        output_dir=args.output,
        device=args.device
    )
