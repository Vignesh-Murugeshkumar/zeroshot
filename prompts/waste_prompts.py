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
# HIERARCHICAL PROMPT TEMPLATES (Levels 1-4)
# ============================================================================

# Level 1: Generic Material Descriptors (minimal context)
LEVEL1_GENERIC: Dict[str, List[str]] = {
    "plastic": [
        "plastic",
        "plastic material",
        "plastic waste",
        "discarded plastic",
    ],
    "paper": [
        "paper",
        "paper material",
        "paper waste",
        "cardboard",
    ],
    "metal": [
        "metal",
        "metallic material",
        "metal waste",
        "shiny metal",
    ],
    "glass": [
        "glass",
        "glass material",
        "glass waste",
        "transparent glass",
    ],
    "organic": [
        "organic waste",
        "food waste",
        "compost",
        "biodegradable waste",
    ],
    "e-waste": [
        "electronics",
        "electronic waste",
        "circuit board",
        "electronic device",
    ],
}

# Level 2: Contextual Descriptors (adds photo/scene context)
LEVEL2_CONTEXTUAL: Dict[str, List[str]] = {
    "plastic": [
        "a photo of plastic waste",
        "a close-up of plastic material",
        "plastic waste in a photo",
        "discarded plastic in an image",
        "plastic garbage in a picture",
        "a photograph of plastic trash",
    ],
    "paper": [
        "a photo of paper waste",
        "a close-up of paper material",
        "cardboard in a photo",
        "discarded paper in an image",
        "paper garbage in a picture",
        "a photograph of paper trash",
    ],
    "metal": [
        "a photo of metal waste",
        "a close-up of metallic object",
        "metal can in a photo",
        "discarded metal in an image",
        "metallic garbage in a picture",
        "a photograph of metal trash",
    ],
    "glass": [
        "a photo of glass waste",
        "a close-up of glass object",
        "glass bottle in a photo",
        "discarded glass in an image",
        "glass garbage in a picture",
        "a photograph of glass trash",
    ],
    "organic": [
        "a photo of organic waste",
        "a close-up of food scraps",
        "organic waste in a photo",
        "discarded food in an image",
        "food garbage in a picture",
        "a photograph of food waste",
    ],
    "e-waste": [
        "a photo of electronic waste",
        "a close-up of circuit board",
        "electronics in a photo",
        "discarded electronics in an image",
        "electronic garbage in a picture",
        "a photograph of e-waste",
    ],
}

# Level 3: Object-Based Descriptors (specific waste items)
LEVEL3_OBJECT_BASED: Dict[str, List[str]] = {
    "plastic": [
        "plastic bottle",
        "plastic bag",
        "plastic packaging",
        "plastic container",
        "plastic wrapper",
        "plastic cup",
        "plastic straw",
        "plastic food container",
        "plastic water bottle",
        "plastic shopping bag",
        "plastic film wrap",
        "plastic takeout container",
        "clear plastic bottle",
        "crushed plastic bottle",
        "empty plastic container",
    ],
    "paper": [
        "cardboard box",
        "newspaper",
        "paper bag",
        "paper envelope",
        "paper scraps",
        "corrugated cardboard",
        "folded cardboard",
        "torn paper",
        "paper packaging",
        "paper receipt",
        "office paper",
        "shredded paper",
        "crumpled paper",
        "flattened cardboard box",
    ],
    "metal": [
        "aluminum can",
        "metal can",
        "tin can",
        "soda can",
        "beer can",
        "metal lid",
        "metal foil",
        "steel can",
        "crushed aluminum can",
        "shiny metal can",
        "reflective metal surface",
        "polished metal object",
        "chrome metal",
        "aluminum beverage can",
    ],
    "glass": [
        "glass bottle",
        "glass jar",
        "wine bottle",
        "beer bottle",
        "glass container",
        "broken glass",
        "glass fragment",
        "clear glass bottle",
        "green glass bottle",
        "brown glass bottle",
        "glass food jar",
        "empty glass bottle",
    ],
    "organic": [
        "food scraps",
        "fruit peel",
        "vegetable waste",
        "banana peel",
        "apple core",
        "orange peel",
        "leftovers",
        "spoiled food",
        "rotten fruit",
        "vegetable scraps",
        "eggshells",
        "coffee grounds",
        "tea bags",
    ],
    "e-waste": [
        "circuit board",
        "motherboard",
        "computer chip",
        "electronic cable",
        "wire",
        "broken phone",
        "old laptop part",
        "electronic component",
        "circuit board with chips",
        "silicon chip",
        "printed circuit board",
        "PCB with components",
        "electronic connector",
        "battery",
    ],
}

# Level 4: Contamination-Aware Descriptors (real-world conditions)
LEVEL4_CONTAMINATION: Dict[str, List[str]] = {
    "plastic": [
        "plastic bottle with food residue",
        "dirty plastic container",
        "plastic with sticky residue",
        "contaminated plastic packaging",
        "plastic bag with stains",
        "greasy plastic container",
        "plastic with dried food",
        "stained plastic wrapper",
    ],
    "paper": [
        "paper with food stains",
        "dirty cardboard box",
        "wet paper waste",
        "stained newspaper",
        "paper with grease marks",
        "contaminated cardboard",
        "soggy paper",
        "paper with oil stains",
    ],
    "metal": [
        "metal can with liquid residue",
        "dirty aluminum can",
        "rusty metal waste",
        "metal with corrosion",
        "tarnished metal surface",
        "metal can with sticky residue",
        "oxidized metal",
    ],
    "glass": [
        "glass bottle with label residue",
        "dirty glass jar",
        "glass with dried liquid",
        "contaminated glass container",
        "glass with sticky residue",
        "stained glass bottle",
    ],
    "organic": [
        "rotten food waste",
        "moldy organic matter",
        "decaying food scraps",
        "spoiled fruit waste",
        "decomposing vegetable matter",
        "organic waste with mold",
    ],
    "e-waste": [
        "dusty circuit board",
        "corroded electronic component",
        "dirty electronic waste",
        "circuit board with dust",
        "aged electronic component",
        "weathered electronics",
    ],
}

# Backward compatibility: Combined base templates
BASE_TEMPLATES: Dict[str, List[str]] = {
    cls: LEVEL1_GENERIC[cls] + LEVEL3_OBJECT_BASED[cls]
    for cls in LEVEL1_GENERIC.keys()
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
    size: Literal["minimal", "small", "medium", "large"]
    # Hierarchical levels
    include_level1_generic: bool = True
    include_level2_contextual: bool = True
    include_level3_object_based: bool = True
    include_level4_contamination: bool = True
    # Environmental descriptors
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
    Build a research-grade prompt bank with hierarchical structure and configurable size.

    Implements 4-level prompt hierarchy:
    - Level 1: Generic material descriptors
    - Level 2: Contextual (photo/scene framing)
    - Level 3: Object-based (specific items)
    - Level 4: Contamination-aware (real-world conditions)

    Args:
        base_templates: custom base templates (ignored if using hierarchical)
        config: PromptSetConfig for size/coverage (defaults to medium config)
        extra_classes: additional classes to include

    Returns:
        Dictionary mapping class name -> list of prompts

    Example:
        >>> cfg = PromptSetConfig(size="large", include_level4_contamination=True)
        >>> prompts = build_prompt_bank(config=cfg)
        >>> print(len(prompts["plastic"]))  # 40-80 prompts per class
        >>> 
        >>> # Different configurations
        >>> cfg_small = PromptSetConfig(size="small")
        >>> cfg_medium = PromptSetConfig(size="medium")
        >>> cfg_large = PromptSetConfig(
        ...     size="large",
        ...     include_level1_generic=True,
        ...     include_level2_contextual=True,
        ...     include_level3_object_based=True,
        ...     include_level4_contamination=True
        ... )
    """
    if config is None:
        config = PromptSetConfig(size="medium")

    # Collect prompts from hierarchical levels
    prompt_bank: Dict[str, List[str]] = {}
    
    # Get class list
    class_list = list(LEVEL1_GENERIC.keys())
    if extra_classes:
        class_list.extend(extra_classes.keys())
    
    for class_name in class_list:
        class_prompts: List[str] = []
        
        # Level 1: Generic
        if config.include_level1_generic and class_name in LEVEL1_GENERIC:
            class_prompts.extend(LEVEL1_GENERIC[class_name])
        
        # Level 2: Contextual
        if config.include_level2_contextual and class_name in LEVEL2_CONTEXTUAL:
            class_prompts.extend(LEVEL2_CONTEXTUAL[class_name])
        
        # Level 3: Object-based
        if config.include_level3_object_based and class_name in LEVEL3_OBJECT_BASED:
            class_prompts.extend(LEVEL3_OBJECT_BASED[class_name])
        
        # Level 4: Contamination
        if config.include_level4_contamination and class_name in LEVEL4_CONTAMINATION:
            class_prompts.extend(LEVEL4_CONTAMINATION[class_name])
        
        # Add extra classes if provided
        if extra_classes and class_name in extra_classes:
            class_prompts.extend(extra_classes[class_name])
        
        # Build expanded prompts with environmental descriptors
        base_for_expansion = class_prompts.copy()
        
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
        
        # Add class-specific descriptors
        if class_name == "metal" and config.include_metal_specific:
            descriptors_list.append(METAL_SPECIFIC_DESCRIPTORS)
        elif class_name == "e-waste" and config.include_electronics_specific:
            descriptors_list.append(ELECTRONICS_SPECIFIC_DESCRIPTORS)
        
        # Expand with descriptors
        if descriptors_list:
            expanded = expand_class_prompts(
                base_for_expansion[:10],  # Use subset for expansion to control size
                descriptors_list,
                max_prompts_per_class=None,
            )
            class_prompts.extend(expanded)
        
        # Limit based on prompt set size
        max_per_class = {"minimal": 10, "small": 20, "medium": 50, "large": 100}.get(config.size, 50)
        
        # Deduplicate and limit
        seen = set()
        final_prompts: List[str] = []
        for p in class_prompts:
            if p not in seen:
                final_prompts.append(p)
                seen.add(p)
                if len(final_prompts) >= max_per_class:
                    break
        
        prompt_bank[class_name] = final_prompts
    
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
