

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import statistics

import torch
from PIL import Image

from classifiers.clip_classifier import ClipWasteClassifier, ClipConfig
from prompts.waste_prompts import build_prompt_bank, PromptSetConfig


# ============================================================================
# Performance Metrics
# ============================================================================
@dataclass
class LatencyMetrics:
    """Inference latency statistics."""
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    num_samples: int

    def __str__(self) -> str:
        return (
            f"Mean: {self.mean_ms:.2f}ms | Median: {self.median_ms:.2f}ms | "
            f"Min: {self.min_ms:.2f}ms | Max: {self.max_ms:.2f}ms | "
            f"Std: {self.std_ms:.2f}ms"
        )


@dataclass
class PromptScalabilityResult:
    """Result of prompt scalability analysis."""
    num_prompts: int
    accuracy: float
    latency_ms: float
    throughput_images_per_sec: float


@dataclass
class ModelComparisonResult:
    """Comparison between different CLIP models."""
    model_name: str
    accuracy: float
    latency_ms: float
    model_size_gb: float
    device: str
    use_fp16: bool

class PerformanceProfiler:

    @staticmethod
    def measure_latency(
        classifier: ClipWasteClassifier,
        images: List[Image.Image],
        warmup: int = 2,
    ) -> LatencyMetrics:
        timings = []

        # Warmup
        for _ in range(min(warmup, len(images))):
            result = classifier.classify_image(images[0], top_k=1)

        # Actual measurements
        for img in images:
            start = time.perf_counter()
            result = classifier.classify_image(img, top_k=1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

        return LatencyMetrics(
            mean_ms=statistics.mean(timings),
            median_ms=statistics.median(timings),
            min_ms=min(timings),
            max_ms=max(timings),
            std_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
            num_samples=len(timings),
        )

    @staticmethod
    def measure_batch_latency(
        classifier: ClipWasteClassifier,
        images: List[Image.Image],
        batch_size: int = 8,
    ) -> Tuple[float, float]:
        start = time.perf_counter()

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            classifier.classify_batch(batch, top_k=1)

        elapsed = time.perf_counter() - start
        throughput = len(images) / elapsed if elapsed > 0 else 0

        return elapsed, throughput

    @staticmethod
    def measure_memory_usage(classifier: ClipWasteClassifier) -> Dict:
        if classifier.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # Dummy inference to measure
            test_img = Image.new("RGB", (224, 224))
            classifier.classify_image(test_img)
            torch.cuda.synchronize()

            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
            return {
                "device": "GPU",
                "peak_memory_gb": peak_memory,
                "cuda_available": True,
            }
        else:
            return {
                "device": "CPU",
                "peak_memory_gb": None,
                "cuda_available": False,
            }


# ============================================================================
# Scalability Analysis
# ============================================================================
class ScalabilityAnalyzer:
    @staticmethod
    def analyze_prompt_scaling(
        images: List[Image.Image],
        labels: List[str],
        class_names: List[str],
    ) -> List[PromptScalabilityResult]:
        from evaluation.benchmark import ZeroShotEvaluator

        results = []

        for prompt_size in ["small", "medium", "large"]:
            print(f"\nAnalyzing prompt size: {prompt_size}")

            # Build classifier with different prompt set
            prompt_cfg = PromptSetConfig(size=prompt_size)
            prompt_bank = build_prompt_bank(config=prompt_cfg)
            clip_cfg = ClipConfig(device="cuda" if torch.cuda.is_available() else "cpu")
            classifier = ClipWasteClassifier(prompt_bank, config=clip_cfg)

            # Measure accuracy
            evaluator = ZeroShotEvaluator(classifier, class_names)
            metrics = evaluator.evaluate(images[:min(100, len(images))], labels[:min(100, len(labels))])
            accuracy = metrics.accuracy

            # Measure latency
            profiler = PerformanceProfiler()
            latency_metrics = profiler.measure_latency(
                classifier, images[:min(20, len(images))], warmup=1
            )

            total_prompts = sum(len(p) for p in prompt_bank.values())
            throughput = 1000 / latency_metrics.mean_ms  # images per second

            result = PromptScalabilityResult(
                num_prompts=total_prompts,
                accuracy=accuracy,
                latency_ms=latency_metrics.mean_ms,
                throughput_images_per_sec=throughput,
            )

            results.append(result)

        return results


# ============================================================================
# Model Comparison
# ============================================================================
class ModelComparator:
    """Compare different CLIP model variants."""

    @staticmethod
    def compare_models(
        images: List[Image.Image],
        labels: List[str],
        class_names: List[str],
        models: List[str] = None,
    ) -> List[ModelComparisonResult]:
        if models is None:
            models = [
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14",
            ]

        from evaluation.benchmark import ZeroShotEvaluator

        results = []

        for model_name in models:
            print(f"\nEvaluating model: {model_name}")

            try:
                prompt_bank = build_prompt_bank()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                use_fp16 = device == "cuda"

                clip_cfg = ClipConfig(
                    model_name=model_name,
                    device=device,
                    use_fp16=use_fp16,
                )
                classifier = ClipWasteClassifier(prompt_bank, config=clip_cfg)

                # Accuracy
                evaluator = ZeroShotEvaluator(classifier, class_names)
                metrics = evaluator.evaluate(images[:min(50, len(images))], labels[:min(50, len(labels))])

                # Latency
                profiler = PerformanceProfiler()
                latency_metrics = profiler.measure_latency(
                    classifier, images[:min(10, len(images))], warmup=1
                )

                # Model size (approximate)
                model_size_gb = 0.3 if "base" in model_name else 0.8

                result = ModelComparisonResult(
                    model_name=model_name,
                    accuracy=metrics.accuracy,
                    latency_ms=latency_metrics.mean_ms,
                    model_size_gb=model_size_gb,
                    device=device,
                    use_fp16=use_fp16,
                )

                results.append(result)

            except Exception as e:
                print(f"  Error loading {model_name}: {e}")

        return results


# ============================================================================
# Report Generation
# ============================================================================
def print_performance_report(
    latency: LatencyMetrics,
    model_name: str,
    num_prompts: int,
    device: str,
) -> None:
    """Print formatted performance report."""
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nModel:           {model_name}")
    print(f"Device:          {device}")
    print(f"Number of prompts: {num_prompts}")
    print(f"\nInference Latency:")
    print(f"  {latency}")
    print(f"\nThroughput:")
    print(f"  {1000 / latency.mean_ms:.2f} images/sec")
    print("=" * 80 + "\n")


def print_scalability_report(results: List[PromptScalabilityResult]) -> None:
    """Print prompt scalability analysis."""
    print("\n" + "=" * 80)
    print("PROMPT SCALABILITY ANALYSIS")
    print("=" * 80)
    print(f"\n{'Size':<10} {'Prompts':<10} {'Accuracy':<12} {'Latency (ms)':<15} {'Throughput':<15}")
    print("-" * 80)

    for result in results:
        print(
            f"{'':10} {result.num_prompts:<10} {result.accuracy:.4f}      "
            f"{result.latency_ms:.2f}          {result.throughput_images_per_sec:.2f}"
        )

    print("=" * 80 + "\n")
    print("Insight: Larger prompt sets improve accuracy but increase latency.")
    print("Select 'small' for real-time deployment, 'large' for offline analysis.\n")


def print_model_comparison(results: List[ModelComparisonResult]) -> None:
    """Print model comparison report."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"\n{'Model':<40} {'Accuracy':<12} {'Latency':<12} {'Size (GB)':<10}")
    print("-" * 80)

    for result in results:
        model_short = result.model_name.split("/")[-1]
        print(
            f"{model_short:<40} {result.accuracy:.4f}       "
            f"{result.latency_ms:.2f}ms      {result.model_size_gb:.2f}"
        )

    print("=" * 80 + "\n")
