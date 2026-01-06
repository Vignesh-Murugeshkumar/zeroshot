"""
Research-Grade Zero-Shot Waste Classification using CLIP.

Streamlit UI for the pure CLIP-based waste classifier with:
- Prompt ensemble support (small/medium/large sets)
- Test-Time Augmentation (TTA)
- Real-time inference with performance metrics
- Robustness evaluation capabilities
- Model comparison and analysis

No object detection (YOLO) - fully zero-shot CLIP-based.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import io
import time
from typing import List

import streamlit as st
from PIL import Image
import torch

from classifiers.clip_classifier import ClipWasteClassifier, ClipConfig
from prompts.waste_prompts import build_prompt_bank, PromptSetConfig
from utils.tta import get_tta_transforms_deterministic


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Zero-Shot Waste Classification (CLIP)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🗑️ Zero-Shot Waste Classification")
st.markdown(
    """
    **Research-Grade CLIP-Based System**
    
    Classify waste images without training using advanced prompt engineering and test-time augmentation.
    No object detection required - purely zero-shot.
    """
)

# ============================================================================
# Sidebar Configuration
# ============================================================================
st.sidebar.header("⚙️ Settings")

prompt_set_size = st.sidebar.selectbox(
    "Prompt Set Size",
    ["small", "medium", "large"],
    index=1,
    help="Larger sets improve accuracy but increase latency",
)

model_choice = st.sidebar.selectbox(
    "CLIP Model",
    [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
    ],
)

use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=True)
use_fp16 = st.sidebar.checkbox("Use FP16 (faster, GPU only)", value=use_gpu and torch.cuda.is_available())

use_tta = st.sidebar.checkbox(
    "Test-Time Augmentation",
    value=False,
    help="Average predictions across augmented views (slower but more robust)",
)

if use_tta:
    tta_augmentations = st.sidebar.slider(
        "TTA Augmentations",
        min_value=3,
        max_value=12,
        value=5,
        help="More augmentations = better robustness but slower",
    )
else:
    tta_augmentations = 1

top_k = st.sidebar.slider(
    "Show Top-K Predictions",
    min_value=1,
    max_value=6,
    value=3,
)

st.sidebar.markdown("---")
st.sidebar.header("📊 System Information")

gpu_available = torch.cuda.is_available()
device_status = f"✓ GPU: {torch.cuda.get_device_name(0)}" if gpu_available else "CPU Only"
st.sidebar.write(f"**Device**: {device_status}")

# ============================================================================
# Model Loading (Cached)
# ============================================================================
@st.cache_resource
def load_classifier(
    model_name: str,
    prompt_size: str,
    use_gpu: bool,
    use_fp16: bool,
):
    """Load and cache the CLIP classifier."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Build prompt bank
    prompt_cfg = PromptSetConfig(size=prompt_size)
    prompt_bank = build_prompt_bank(config=prompt_cfg)

    # Build classifier
    clip_cfg = ClipConfig(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16 and device == "cuda",
    )
    classifier = ClipWasteClassifier(prompt_bank, config=clip_cfg)

    return classifier, prompt_bank


# Load models
with st.spinner("Loading CLIP model..."):
    classifier, prompt_bank = load_classifier(
        model_name=model_choice,
        prompt_size=prompt_set_size,
        use_gpu=use_gpu,
        use_fp16=use_fp16,
    )

st.sidebar.success(f"Model loaded with {sum(len(p) for p in prompt_bank.values())} prompts")

# ============================================================================
# Image Input
# ============================================================================
st.header("📸 Input")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    camera_img = st.camera_input("Or Take Photo")

# Get image
image = None
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
elif camera_img is not None:
    image = Image.open(io.BytesIO(camera_img.read())).convert("RGB")

if image is None:
    st.info("👆 Upload an image or take a photo to classify waste.")
    st.stop()

# ============================================================================
# Inference
# ============================================================================
st.header("🔍 Classification Results")

col_input, col_preview = st.columns(2)

with col_input:
    st.subheader("Input Image")
    st.image(image, use_column_width=True, caption="Uploaded image")

with col_preview:
    st.subheader("Augmentations (TTA)")
    if use_tta:
        augmentations = get_tta_transforms_deterministic(image, tta_augmentations)
        # Show first 4 augmentations as a grid
        aug_cols = st.columns(2)
        for i, aug_img in enumerate(augmentations[:4]):
            with aug_cols[i % 2]:
                st.image(aug_img, use_column_width=True, caption=f"Aug {i+1}")
    else:
        st.write("TTA disabled - using original image only")

# Run classification
with st.spinner(f"Classifying (TTA={use_tta}, Augmentations={tta_augmentations})..."):
    start_time = time.time()

    result = classifier.classify_image(
        image,
        top_k=top_k,
        use_tta=use_tta,
        tta_augmentations=tta_augmentations,
    )

    elapsed_ms = (time.time() - start_time) * 1000

# ============================================================================
# Results Display
# ============================================================================
st.subheader("🎯 Predictions")

# Top prediction highlighted
top_class, top_score = result.ranked[0]
col_main, col_metrics = st.columns([2, 1])

with col_main:
    st.success(f"**Top Prediction**: {top_class.upper()} ({top_score:.1%})")

with col_metrics:
    st.metric("Inference Time", f"{elapsed_ms:.0f}ms")

# All predictions as a bar chart
st.subheader("📊 Score Distribution")

pred_classes = [p[0] for p in result.ranked]
pred_scores = [p[1] for p in result.ranked]

chart_data = {
    "Class": pred_classes,
    "Confidence": pred_scores,
}

import pandas as pd

df = pd.DataFrame(chart_data).set_index("Class")
st.bar_chart(df, height=400)

# Detailed predictions table
st.subheader("📋 Detailed Results")

table_data = []
for i, (cls, score) in enumerate(result.ranked, 1):
    table_data.append({
        "Rank": i,
        "Class": cls,
        "Confidence": f"{score:.1%}",
        "Bar": "█" * int(score * 20),
    })

df_results = pd.DataFrame(table_data)
st.dataframe(df_results, use_container_width=True)

# ============================================================================
# System Information
# ============================================================================
st.markdown("---")
st.subheader("ℹ️ System Details")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Model", model_choice.split("/")[-1])

with col_info2:
    st.metric("Prompts", sum(len(p) for p in prompt_bank.values()))

with col_info3:
    device_name = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
    st.metric("Device", device_name)

if use_tta:
    st.info(
        f"ℹ️ **TTA Mode**: Predictions averaged across {tta_augmentations} augmented views "
        f"for improved robustness. Slower but more reliable on challenging images."
    )

# ============================================================================
# Footer
# ============================================================================
st.markdown(
    """
    ---
    **About this system**:
    - Zero-shot classification: no training required
    - CLIP-based with advanced prompt engineering
    - Supports test-time augmentation for robustness
    - Research-grade evaluation tools available
    
    [📚 Research Documentation](./research.md) | 
    [🔍 Evaluation](./eval.py)
    """
)

