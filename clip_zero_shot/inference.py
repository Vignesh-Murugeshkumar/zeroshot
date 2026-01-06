"""Minimal inference module - YOLO detection only."""

from typing import List, Dict, Optional, Union
from PIL import Image
import torch
from .clip_loader import load_clip
from .prompts import get_default_prompts


class ZeroShotPipeline:
    """Minimal zero-shot classification pipeline using YOLO detection only."""
    
    def __init__(self, model_names: Optional[List[str]] = None, prompts: Optional[Dict[str, List[str]]] = None):
        """Initialize pipeline.
        
        Args:
            model_names: list of CLIP model names (default: openai/clip-vit-base-patch32)
            prompts: optional custom prompts dict; if None, use defaults
        """
        self.model_names = model_names or ["openai/clip-vit-base-patch32"]
        self.prompts = prompts or get_default_prompts()
        self.models = []
        self.processors = []
        self.text_batch_size = 64
        
        # Load CLIP models
        for model_name in self.model_names:
            model, processor, device = load_clip(model_name)
            self.models.append(model)
            self.processors.append(processor)
            self.device = device
    
    def classify(
        self,
        image: Union[str, Image.Image],
        tta: bool = False,
        agg_method: str = "mean",
        temp: float = 0.1,
        top_k: int = 3,
    ) -> Dict:
        """Classify an image using CLIP zero-shot classification.
        
        Args:
            image: PIL Image or path to image
            tta: test-time augmentation (currently unused)
            agg_method: aggregation method (mean or trimmed_mean)
            temp: temperature for softmax
            top_k: number of top predictions to return
        
        Returns:
            dict with 'ranked_scores' (list of tuples) and 'raw_scores' (dict)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        # Get all prompts for all classes
        all_prompts = []
        class_names = []
        for cls_name, prompts_list in self.prompts.items():
            all_prompts.extend(prompts_list)
            class_names.extend([cls_name] * len(prompts_list))
        
        # Encode texts with the first model
        model = self.models[0]
        processor = self.processors[0]
        device = self.device
        
        # Encode image
        image_inputs = processor(image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Encode texts in batches
        text_features_list = []
        for i in range(0, len(all_prompts), self.text_batch_size):
            batch_prompts = all_prompts[i:i + self.text_batch_size]
            text_inputs = processor(text=batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                batch_features = model.get_text_features(**text_inputs)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            text_features_list.append(batch_features)
        
        text_features = torch.cat(text_features_list, dim=0)
        
        # Compute similarities
        logits = (image_features @ text_features.T) * 100  # temperature scaling
        probs = torch.softmax(logits, dim=-1)[0]
        
        # Aggregate by class
        class_scores = {}
        for cls_name in self.prompts.keys():
            class_scores[cls_name] = 0.0
        
        for i, cls_name in enumerate(class_names):
            class_scores[cls_name] += float(probs[i])
        
        # Normalize by number of prompts per class
        for cls_name in self.prompts.keys():
            num_prompts = len(self.prompts[cls_name])
            if num_prompts > 0:
                class_scores[cls_name] /= num_prompts
        
        # Sort by score
        sorted_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_scores = sorted_scores[:top_k]
        
        return {
            'ranked_scores': ranked_scores,
            'raw_scores': class_scores,
        }
