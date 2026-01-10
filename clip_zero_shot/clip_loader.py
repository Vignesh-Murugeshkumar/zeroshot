from typing import Tuple, Optional
import torch
from transformers import CLIPModel, CLIPProcessor


def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Load a CLIP model and processor with automatic GPU detection.

    Args:
        model_name: Hugging Face model name (openai/clip or open_clip equivalent)
        device: optional device string (e.g., 'cuda:0' or 'cpu'). Auto-detects GPU if None.

    Returns:
        model, processor, device
    """
    # Auto-detect GPU if device not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # Load model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Move to device and set to eval mode (disables dropout, batch norm updates)
    model.to(torch_device)
    model.eval()

    return model, processor, torch_device
