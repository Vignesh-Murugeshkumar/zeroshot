#!/usr/bin/env python
"""
Example: Complete Zero-Shot Waste Classification Pipeline

Demonstrates:
1. Building CLIP classifier with advanced prompts
2. Running zero-shot evaluation on a dataset
3. Robustness analysis under adverse conditions
4. Performance profiling
5. Generating research-grade reports

Usage:
    python examples/complete_pipeline.py \\
        --data_root /path/to/dataset \\
        --output results/
"""

import argparse
from pathlib import Path

from classifiers.clip_classifier import ClipWasteClassifier, ClipConfig
from prompts.waste_prompts import build_prompt_bank, PromptSetConfig
from evaluation.benchmark import ZeroShotEvaluator, load_dataset_from_directory, print_baseline_comparison
from evaluation.robustness import RobustnessAnalyzer
from evaluation.performance import PerformanceProfiler, ScalabilityAnalyzer, PerformanceProfiler as PerformanceProfiler2


def main():
    parser = argparse.ArgumentParser(description="Complete zero-shot waste classification pipeline")
    parser.add_argument("--data_root", required=True, type=Path, help="Path to dataset root directory")
    parser.add_argument("--prompt_set", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ZERO-SHOT WASTE CLASSIFICATION PIPELINE")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Load Dataset
    # ========================================================================
    print("\n[STEP 1] Loading dataset...")
    images, labels, class_names = load_dataset_from_directory(args.data_root)
    print(f"✓ Loaded {len(images)} images from {len(class_names)} classes")

    # ========================================================================
    # STEP 2: Build CLIP Classifier with Advanced Prompts
    # ========================================================================
    print("\n[STEP 2] Building CLIP classifier with advanced prompts...")
    prompt_cfg = PromptSetConfig(size=args.prompt_set)
    prompt_bank = build_prompt_bank(config=prompt_cfg)
    
    clip_cfg = ClipConfig(
        model_name=args.model,
        device="cuda" if args.gpu else "cpu",
        use_fp16=args.gpu,
    )
    classifier = ClipWasteClassifier(prompt_bank, config=clip_cfg)
    
    total_prompts = sum(len(p) for p in prompt_bank.values())
    print(f"✓ Classifier ready with {total_prompts} prompts")
    print(f"  - Prompt set: {args.prompt_set}")
    print(f"  - Model: {args.model.split('/')[-1]}")
    print(f"  - Device: {'GPU' if args.gpu else 'CPU'}")

    # ========================================================================
    # STEP 3: Zero-Shot Evaluation
    # ========================================================================
    print("\n[STEP 3] Running zero-shot evaluation...")
    evaluator = ZeroShotEvaluator(classifier, class_names)
    metrics = evaluator.evaluate(images, labels, use_tta=args.use_tta)
    
    print("\n" + str(metrics))
    print_baseline_comparison(metrics.accuracy, dataset="TrashNet")

    # Save results
    import json
    metrics_path = args.output / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # ========================================================================
    # STEP 4: Robustness Analysis
    # ========================================================================
    print("\n[STEP 4] Analyzing robustness to adverse conditions...")
    analyzer = RobustnessAnalyzer(evaluator)
    robustness_results = analyzer.evaluate_all_conditions(images, labels)
    
    analyzer.print_robustness_report(robustness_results)

    # Save robustness results
    from evaluation.robustness import export_robustness_analysis
    robustness_path = args.output / "robustness.json"
    export_robustness_analysis(robustness_results, robustness_path)
    print(f"✓ Robustness analysis saved to {robustness_path}")

    # ========================================================================
    # STEP 5: Performance Profiling
    # ========================================================================
    print("\n[STEP 5] Profiling performance...")
    profiler = PerformanceProfiler()
    
    # Measure latency
    test_images = images[:min(20, len(images))]
    latency = profiler.measure_latency(classifier, test_images, warmup=2)
    print(f"\nInference Latency:")
    print(f"  {latency}")

    # Measure memory
    memory = profiler.measure_memory_usage(classifier)
    print(f"\nMemory Usage:")
    print(f"  Device: {memory['device']}")
    if memory['peak_memory_gb']:
        print(f"  Peak memory: {memory['peak_memory_gb']:.2f} GB")

    # ========================================================================
    # STEP 6: Scalability Analysis
    # ========================================================================
    print("\n[STEP 6] Analyzing prompt scalability...")
    scaling_analyzer = ScalabilityAnalyzer()
    scaling_results = scaling_analyzer.analyze_prompt_scaling(
        images[:min(100, len(images))], 
        labels[:min(100, len(labels))],
        class_names
    )

    from evaluation.performance import print_scalability_report
    print_scalability_report(scaling_results)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output.resolve()}")
    print(f"  - metrics.json: Classification metrics (accuracy, precision, recall, F1)")
    print(f"  - robustness.json: Robustness analysis (adversarial conditions)")
    print(f"\nKey Findings:")
    print(f"  - Zero-shot accuracy: {metrics.accuracy:.1%}")
    print(f"  - Inference latency: {latency.mean_ms:.1f}ms")
    print(f"  - Robustness: Evaluated on 6 conditions")
    print(f"\nNext steps:")
    print(f"  1. Review results in {args.output}")
    print(f"  2. Adjust prompt set size or model for your use case")
    print(f"  3. Run streamlit app for interactive classification: streamlit run app.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
