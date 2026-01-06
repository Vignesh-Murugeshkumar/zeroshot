"""
Advanced Prompt Engineering for Zero-Shot Waste Classification.

This module implements research-grade prompt design addressing real-world waste
classification challenges:
- Food-contaminated waste
- Dirty/wet conditions
- Cluttered backgrounds
- Mixed waste scenarios
- Varying lighting and scales

Supports configurable prompt sets (small/medium/large) for scalability analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# BASE TEMPLATES: Essential class descriptors
# ============================================================================
BASE_TEMPLATES: Dict[str, List[str]] = {
    "plastic": [
        "plastic waste",
        "plastic bottle",
        "plastic bag",
        "plastic packaging",
        "plastic container",
        "plastic film",
    ],
    "paper": [
        "paper waste",
        "cardboard",
        "newspaper",
        "cardboard box",
        "paper scraps",
        "paper envelope",
    ],
    "metal": [
        "metal waste",
        "aluminum can",
        "metal can",
        "steel waste",
        "tin can",
        "metal foil",
        "shiny metal surface",
        "reflective metal object",
        "metallic material",
        "polished metal",
        "metal lid",
        "chrome metal",
    ],
    "glass": [
        "glass waste",
        "glass bottle",
        "glass jar",
        "broken glass",
        "glass container",
        "glass fragment",
    ],
    "organic": [
        "organic waste",
        "food waste",
        "food scraps",
        "vegetable waste",
        "fruit waste",
        "compostable waste",
    ],
    "e-waste": [
        "electronic waste",
        "circuit board",
        "broken electronics",
        "electronic cable",
        "electronic component",
        "tech waste",
        "computer part",
        "circuit board with chips",
        "silicon chip",
        "motherboard",
        "broken phone or laptop",
        "wired electronic device",
    ],
}

# ============================================================================
# REAL-WORLD CONDITION DESCRIPTORS
# ============================================================================
CONTAMINATION_DESCRIPTORS: List[str] = [
    "with food residue",
    "with dried food",
    "with sticky residue",
    "contaminated with food",
    "stained with food",
    "covered with food",
]

DIRT_DESCRIPTORS: List[str] = [
    "dirty and stained",
    "muddy and wet",
    "wet and soiled",
    "covered in dirt",
    "grimy surface",
    "dusty and dirty",
]

CLUTTER_DESCRIPTORS: List[str] = [
    "in a cluttered scene",
    "mixed with other trash",
    "among other waste",
    "in a messy pile",
    "in a jumbled heap",
    "surrounded by clutter",
]

CONTEXT_DESCRIPTORS: List[str] = [
    "on the ground",
    "on a sidewalk",
    "in a trash bin",
    "in a recycling bin",
    "overflowing from a bin",
    "on a pile",
]

LIGHTING_DESCRIPTORS: List[str] = [
    "in low light",
    "in shadows",
    "partially obscured",
    "dimly lit",
    "with shadows",
    "in poor lighting",
]

SCALE_DESCRIPTORS: List[str] = [
    "close-up view",
    "at a distance",
    "from above",
    "at ground level",
    "partially visible",
    "in perspective",
]

# Special descriptors to distinguish metal from electronics
METAL_SPECIFIC_DESCRIPTORS: List[str] = [
    "with metallic sheen",
    "smooth shiny surface",
    "reflective surface",
    "solid metal material",
    "polished finish",
    "mirror-like reflection",
]

ELECTRONICS_SPECIFIC_DESCRIPTORS: List[str] = [
    "with circuit traces",
    "with silicon chips",
    "with wires",
    "with plastic housing",
    "with electronic components",
    "with circuit board",
]

# ============================================================================
# PROMPT SET CONFIGURATIONS
# ============================================================================
@dataclass
class PromptSetConfig:
    """Configuration for prompt set size and coverage."""
    size: Literal["small", "medium", "large"]
    include_contamination: bool = True
    include_dirt: bool = True
    include_clutter: bool = True
    include_context: bool = True
    include_lighting: bool = True
    include_scale: bool = True
    # Special descriptors for better class distinction
    include_metal_specific: bool = True
    include_electronics_specific: bool = True


def expand_class_prompts(
    base_prompts: List[str],
    descriptors_list: List[List[str]],
    max_prompts_per_class: Optional[int] = None,
) -> List[str]:
    """
    Expand base prompts with real-world descriptors.

    Args:
        base_prompts: base class prompts (e.g., "plastic waste")
        descriptors_list: list of descriptor lists to combine
        max_prompts_per_class: limit expansions per class

    Returns:
        expanded prompt list with duplicates removed
    """
    prompts: List[str] = []

    # Add base prompts with common templates
    for base in base_prompts:
        prompts.append(f"a photo of {base}")
        prompts.append(f"a close-up of {base}")
        prompts.append(f"{base} on the ground")
        prompts.append(f"discarded {base}")

    # Add combinations with descriptors
    for descriptors in descriptors_list:
        for desc in descriptors:
            for base in base_prompts:
                prompts.append(f"a photo of {base} {desc}")
                prompts.append(f"{base} {desc}")

    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for p in prompts:
        if p not in seen:
            out.append(p)
            seen.add(p)
            if max_prompts_per_class and len(out) >= max_prompts_per_class:
                break

    return out


def build_prompt_bank(
    base_templates: Optional[Dict[str, List[str]]] = None,
    config: Optional[PromptSetConfig] = None,
    extra_classes: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Build a research-grade prompt bank with configurable size and coverage.

    Args:
        base_templates: custom base templates (defaults to BASE_TEMPLATES)
        config: PromptSetConfig for size/coverage (defaults to medium config)
        extra_classes: additional classes to include

    Returns:
        Dictionary mapping class name -> list of prompts

    Example:
        >>> cfg = PromptSetConfig(size="large")
        >>> prompts = build_prompt_bank(config=cfg)
        >>> print(len(prompts["plastic"]))  # ~30-50 prompts per class
    """
    if base_templates is None:
        base_templates = BASE_TEMPLATES

    if config is None:
        config = PromptSetConfig(size="medium")

    # Select descriptors based on config and class
    descriptors_list: List[List[str]] = []
    if config.include_contamination:
        descriptors_list.append(CONTAMINATION_DESCRIPTORS)
    if config.include_dirt:
        descriptors_list.append(DIRT_DESCRIPTORS)
    if config.include_clutter:
        descriptors_list.append(CLUTTER_DESCRIPTORS)
    if config.include_context:
        descriptors_list.append(CONTEXT_DESCRIPTORS)
    if config.include_lighting:
        descriptors_list.append(LIGHTING_DESCRIPTORS)
    if config.include_scale:
        descriptors_list.append(SCALE_DESCRIPTORS)

    # Limit based on prompt set size
    max_per_class = {"small": 15, "medium": 35, "large": 60}.get(config.size, 35)

    prompt_bank: Dict[str, List[str]] = {}
    templates = dict(base_templates)
    if extra_classes:
        templates.update(extra_classes)

    for class_name, base_list in templates.items():
        # Add class-specific descriptors
        class_descriptors = list(descriptors_list)
        if class_name == "metal" and config.include_metal_specific:
            class_descriptors.append(METAL_SPECIFIC_DESCRIPTORS)
        elif class_name == "e-waste" and config.include_electronics_specific:
            class_descriptors.append(ELECTRONICS_SPECIFIC_DESCRIPTORS)
        
        expanded = expand_class_prompts(base_list, class_descriptors, max_per_class)
        prompt_bank[class_name] = expanded

    return prompt_bank


def get_prompt_set_info(config: PromptSetConfig) -> str:
    """Return a human-readable description of the prompt set configuration."""
    lines = [
        f"Prompt Set Size: {config.size}",
        f"Contamination descriptors: {config.include_contamination}",
        f"Dirt descriptors: {config.include_dirt}",
        f"Clutter descriptors: {config.include_clutter}",
        f"Context descriptors: {config.include_context}",
        f"Lighting descriptors: {config.include_lighting}",
        f"Scale descriptors: {config.include_scale}",
    ]
    return "\n".join(lines)
