"""
Zero-Shot Waste Classification using CLIP.

Simplified Streamlit demo for CLIP-based waste classification.

Features:
- Zero-shot classification (no training required)
- Prompt engineering for accurate predictions
- Image upload and camera input

Fixed Configuration:
- Model: ViT-B/32 (fast and balanced)
- Prompt set: medium (50 prompts per class)
- Device: CPU (universal compatibility)
- No test-time augmentation (optimized for speed)

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import io
import time

import streamlit as st
from PIL import Image

from classifiers.clip_classifier import ClipWasteClassifier, ClipConfig
from prompts.waste_prompts import build_prompt_bank, PromptSetConfig


# ============================================================================
# Environment Detection
# ============================================================================
import torch
import os

# Detect if running in Kaggle
IS_KAGGLE = os.path.exists('/kaggle/input')

# Check GPU availability
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_NAME = None
    GPU_COUNT = 0


# ============================================================================
# Fixed Configuration (No UI Controls)
# ============================================================================

FIXED_CONFIG = {
    "model_name": "openai/clip-vit-base-patch32",  # ViT-B/32 - fast and balanced
    "prompt_set_size": "minimal",  # Minimal prompts for speed (10-15 per class)
    "device": "cuda" if GPU_AVAILABLE else "cpu",  # Auto GPU detection (works on Kaggle)
    "use_fp16": GPU_AVAILABLE,  # FP16 on GPU for speed (2x faster)
    "use_tta": False,  # TTA disabled for speed
    "top_k": 1,  # Show only top prediction
}


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Zero-Shot Waste Classification",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("♻️ Waste Classification")
st.markdown(
    """
    **Zero-Shot CLIP-Based Classification**
    
    Upload a waste image to automatically identify its category using AI.
    No training required - powered by OpenAI's CLIP model.
    """
)

# Display device status
if IS_KAGGLE:
    if GPU_AVAILABLE:
        st.success(f"🚀 Running on Kaggle with **{GPU_NAME}** GPU (FP16 enabled)")
    else:
        st.warning("⚠️ Running on Kaggle CPU. Enable GPU: Settings → Accelerator → GPU (T4/P100)")
else:
    if GPU_AVAILABLE:
        st.success(f"🚀 GPU Detected: **{GPU_NAME}** ({GPU_COUNT} device{'s' if GPU_COUNT > 1 else ''})")
    else:
        st.info("💻 Running on CPU (slower but works everywhere)")

st.markdown("---")


# ============================================================================
# Model Loading (Cached)
# ============================================================================
@st.cache_resource
def load_classifier():
    """Load and cache the CLIP classifier with fixed configuration."""
    # Build prompt bank
    prompt_cfg = PromptSetConfig(size=FIXED_CONFIG["prompt_set_size"])
    prompt_bank = build_prompt_bank(config=prompt_cfg)

    # Build classifier
    clip_cfg = ClipConfig(
        model_name=FIXED_CONFIG["model_name"],
        device=FIXED_CONFIG["device"],
        use_fp16=FIXED_CONFIG["use_fp16"],
        temperature=0.1,  # Optimized for decisive predictions
    )
    classifier = ClipWasteClassifier(prompt_bank, config=clip_cfg)

    return classifier


# Load model once
with st.spinner("🔄 Loading AI model..."):
    classifier = load_classifier()

# ============================================================================
# Image Input
# ============================================================================
st.header("📸 Upload Image")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Choose a waste image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of waste item"
    )

with col2:
    camera_img = st.camera_input(
        "Or take a photo",
        help="Use your camera to capture waste"
    )

# Get image
image = None
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
elif camera_img is not None:
    image = Image.open(io.BytesIO(camera_img.read())).convert("RGB")

if image is None:
    st.info("👆 Please upload an image or take a photo to begin classification.")
    st.stop()


# ============================================================================
# Display Input
# ============================================================================
st.markdown("---")
st.header("🖼️ Your Image")
st.image(image, use_column_width=True, caption="Uploaded waste image")


# ============================================================================
# Inference
# ============================================================================
st.markdown("---")
st.header("🔍 Classification")

with st.spinner("🤖 Analyzing image..."):
    start_time = time.time()

    result = classifier.classify_image(
        image,
        top_k=FIXED_CONFIG["top_k"],
        use_tta=FIXED_CONFIG["use_tta"],
    )

    elapsed_ms = (time.time() - start_time) * 1000


# ============================================================================
# Results Display
# ============================================================================
# Top prediction (only show top-1)
top_class, top_score = result.ranked[0]

# Large, centered result
st.markdown("### 🎯 Result")
st.success(f"# **{top_class.upper()}**")
st.progress(min(top_score, 1.0))
st.caption(f"Confidence: {top_score:.1%} • Time: {elapsed_ms:.0f}ms")

# Waste category descriptions
category_info = {
    "plastic": "♻️ Recyclable plastic materials (bottles, containers, bags, packaging)",
    "paper": "📄 Recyclable paper products (cardboard, newspapers, documents)",
    "metal": "🔩 Recyclable metals (aluminum cans, steel cans, metal containers)",
    "glass": "🍾 Recyclable glass (bottles, jars, containers)",
    "organic": "🌱 Compostable organic waste (food scraps, plant matter)",
    "e-waste": "🔌 Electronic waste (circuit boards, cables, electronics)",
}

if top_class in category_info:
    st.info(category_info[top_class])


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")

with st.expander("ℹ️ About this system"):
    st.markdown(
        """
        **Zero-Shot Waste Classification**
        
        This system uses OpenAI's CLIP model to classify waste images without any training.
        
        **How it works:**
        - CLIP matches images to text descriptions
        - 50+ prompts per waste category for accuracy
        - No training data required (zero-shot learning)
        
        **Waste categories:**
        - ♻️ Plastic: bottles, bags, containers, packaging
        - 📄 Paper: cardboard, newspapers, documents
        - 🔩 Metal: aluminum/steel cans, metal containers
        - 🍾 Glass: bottles, jars, containers
        - 🌱 Organic: food scraps, plant matter
        - 🔌 E-waste: electronics, circuit boards, cables
        
        **Configuration:**
        - Model: CLIP ViT-B/32 (optimized for speed)
        - Device: GPU (CUDA) if available, otherwise CPU
        - Prompts: Minimal set (10-15 per class for faster inference)
        - FP16: Enabled on GPU for 2x speed boost
        
        **Note:** This is a demonstration system. For production use or advanced 
        configuration (TTA, model selection, GPU acceleration), see the full 
        research implementation in the repository.
        """
    )

st.caption("Powered by OpenAI CLIP • Zero-shot learning • No training required")


