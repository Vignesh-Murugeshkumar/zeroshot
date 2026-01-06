

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance
import numpy as np

from evaluation.benchmark import ZeroShotEvaluator, EvaluationMetrics


# ============================================================================
# Condition Simulators
# ============================================================================
class ConditionSimulator:
    """Simulate adverse real-world conditions on images."""

    @staticmethod
    def low_light(image: Image.Image, brightness_factor: float = 0.3) -> Image.Image:
        """Simulate low lighting by reducing brightness."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness_factor)

    @staticmethod
    def high_contrast(image: Image.Image, contrast_factor: float = 0.5) -> Image.Image:
        """Simulate harsh lighting with high contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)

    @staticmethod
    def color_cast(image: Image.Image, color_factor: float = 0.5) -> Image.Image:
        """Simulate color degradation."""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(color_factor)

    @staticmethod
    def blur(image: Image.Image, kernel_size: int = 5) -> Image.Image:
        """Simulate motion blur or focus issues."""
        return image.filter(
            __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(kernel_size)
        )

    @staticmethod
    def add_noise(image: Image.Image, noise_level: float = 0.05) -> Image.Image:
        """Add Gaussian noise to simulate sensor noise."""
        arr = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 255 * noise_level, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    @staticmethod
    def compress(image: Image.Image, quality: int = 50) -> Image.Image:
        """Simulate JPEG compression artifacts."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)


# ============================================================================
# Robustness Evaluation
# ============================================================================
@dataclass
class RobustnessResult:
    """Result of robustness evaluation for a single condition."""
    condition: str
    clean_accuracy: float
    degraded_accuracy: float
    degradation_drop: float
    degradation_percentage: float
    is_resilient: bool  # True if <10% accuracy drop
    evaluator_metrics: Dict  # Full metrics from evaluator


class RobustnessAnalyzer:
    """
    Analyze zero-shot classifier robustness under adverse conditions.

    Example:
        >>> clf = ClipWasteClassifier(...)
        >>> evaluator = ZeroShotEvaluator(clf, class_names)
        >>> analyzer = RobustnessAnalyzer(evaluator)
        >>> results = analyzer.evaluate_all_conditions(clean_images, labels)
    """

    def __init__(self, evaluator: ZeroShotEvaluator):
        self.evaluator = evaluator
        self.simulator = ConditionSimulator()

    def evaluate_condition(
        self,
        clean_images: List[Image.Image],
        labels: List[str],
        condition_name: str,
        degradation_fn,
    ) -> RobustnessResult:
        print(f"\nEvaluating condition: {condition_name}")
        print(f"  Clean baseline...")
        clean_metrics = self.evaluator.evaluate(clean_images, labels, use_tta=False)
        clean_acc = clean_metrics.accuracy

        # Apply degradation
        print(f"  Applying degradation: {condition_name}...")
        degraded_images = []
        for img in clean_images:
            try:
                degraded = degradation_fn(img)
                degraded_images.append(degraded)
            except Exception as e:
                print(f"    Warning: degradation failed - {e}, skipping image")
                degraded_images.append(img)  # Fallback to original

        print(f"  Degraded evaluation...")
        degraded_metrics = self.evaluator.evaluate(degraded_images, labels, use_tta=False)
        degraded_acc = degraded_metrics.accuracy

        degradation = clean_acc - degraded_acc
        degradation_pct = (degradation / clean_acc * 100) if clean_acc > 0 else 0
        is_resilient = degradation_pct < 10.0

        return RobustnessResult(
            condition=condition_name,
            clean_accuracy=clean_acc,
            degraded_accuracy=degraded_acc,
            degradation_drop=degradation,
            degradation_percentage=degradation_pct,
            is_resilient=is_resilient,
            evaluator_metrics={
                "clean": clean_metrics.to_dict(),
                "degraded": degraded_metrics.to_dict(),
            },
        )

    def evaluate_all_conditions(
        self,
        images: List[Image.Image],
        labels: List[str],
    ) -> Dict[str, RobustnessResult]:
        conditions = {
            "Low Lighting": lambda img: self.simulator.low_light(img, brightness_factor=0.4),
            "High Contrast": lambda img: self.simulator.high_contrast(img),
            "Color Degradation": lambda img: self.simulator.color_cast(img),
            "Blur (Out-of-Focus)": lambda img: self.simulator.blur(img, kernel_size=7),
            "Sensor Noise": lambda img: self.simulator.add_noise(img, noise_level=0.08),
            "JPEG Compression": lambda img: self.simulator.compress(img, quality=50),
        }

        results = {}
        for cond_name, degradation_fn in conditions.items():
            result = self.evaluate_condition(images, labels, cond_name, degradation_fn)
            results[cond_name] = result

        return results

    def print_robustness_report(self, results: Dict[str, RobustnessResult]) -> None:
        """Print formatted robustness evaluation report."""
        print("\n" + "=" * 80)
        print("ROBUSTNESS ANALYSIS REPORT")
        print("=" * 80 + "\n")

        print(f"{'Condition':<30} {'Clean':<10} {'Degraded':<10} {'Drop':<10} {'Status':<15}")
        print("-" * 80)

        total_degradation = []
        for condition, result in results.items():
            status = "✓ RESILIENT" if result.is_resilient else "✗ SENSITIVE"
            print(
                f"{condition:<30} {result.clean_accuracy:.4f}    "
                f"{result.degraded_accuracy:.4f}    "
                f"{result.degradation_drop:+.4f}   {status}"
            )
            total_degradation.append(result.degradation_drop)

        avg_degradation = sum(total_degradation) / len(total_degradation) if total_degradation else 0
        resilience_count = sum(1 for r in results.values() if r.is_resilient)

        print("-" * 80)
        print(f"\nSummary:")
        print(f"  Average degradation: {avg_degradation:.4f} ({avg_degradation / avg_degradation * 100:.1f}%)")
        print(f"  Resilient conditions: {resilience_count}/{len(results)}")

        if resilience_count >= len(results) * 0.67:
            print(f"  Overall Resilience: ✓ GOOD (handles diverse conditions well)")
        elif resilience_count >= len(results) * 0.33:
            print(f"  Overall Resilience: ◐ MODERATE (handles some adverse conditions)")
        else:
            print(f"  Overall Resilience: ✗ LIMITED (struggles with real-world conditions)")

        print("=" * 80 + "\n")


# ============================================================================
# Utilities
# ============================================================================
def export_robustness_analysis(
    results: Dict[str, RobustnessResult],
    output_path: Path | str,
) -> None:
    """Export robustness analysis to JSON."""
    import json

    export_data = {}
    for condition, result in results.items():
        export_data[condition] = {
            "clean_accuracy": result.clean_accuracy,
            "degraded_accuracy": result.degraded_accuracy,
            "degradation_drop": result.degradation_drop,
            "degradation_percentage": result.degradation_percentage,
            "is_resilient": result.is_resilient,
        }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Robustness analysis exported to {output_path}")
