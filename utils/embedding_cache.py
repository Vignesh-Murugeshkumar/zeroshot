"""
Persistent Text Embedding Cache for CLIP Models.

This module provides disk-based caching of text embeddings to:
- Eliminate redundant encoding of the same prompts across runs
- Reduce startup time for evaluation and inference
- Enable faster experimentation with different configurations

Cache Key Generation:
- Hash of: model_name + prompt_text + normalization_flag
- Ensures cache hits only for identical configurations

Performance Benefits:
- First run: ~5-10s to encode 300 prompts
- Subsequent runs: <0.1s to load from disk
- Critical for large prompt banks (60-100 prompts per class)

Storage:
- Uses pickle for fast serialization
- Organized by model name in cache directory
- Automatic cache invalidation on model change
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class EmbeddingCache:
    """
    Disk-based cache for CLIP text embeddings.
    
    Example:
        >>> cache = EmbeddingCache(cache_dir=".cache/embeddings")
        >>> 
        >>> # Try to load from cache
        >>> embeddings = cache.load(
        ...     model_name="openai/clip-vit-base-patch32",
        ...     prompts=["plastic bottle", "glass jar", "metal can"]
        ... )
        >>> 
        >>> if embeddings is None:
        ...     # Cache miss - compute embeddings
        ...     embeddings = encode_prompts_with_model(prompts)
        ...     
        ...     # Save to cache
        ...     cache.save(
        ...         model_name="openai/clip-vit-base-patch32",
        ...         prompts=prompts,
        ...         embeddings=embeddings
        ...     )
    """
    
    def __init__(self, cache_dir: str = ".cache/clip_embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True
    
    def _generate_cache_key(
        self, 
        model_name: str, 
        prompts: List[str],
        normalized: bool = True
    ) -> str:
        """
        Generate unique cache key for model + prompts combination.
        
        Args:
            model_name: CLIP model identifier
            prompts: list of text prompts
            normalized: whether embeddings are L2-normalized
        
        Returns:
            cache key (hex string)
        """
        # Concatenate all prompts in order
        prompts_str = "||".join(prompts)
        
        # Include model name and normalization flag
        key_components = f"{model_name}|{prompts_str}|norm={normalized}"
        
        # Hash to get fixed-length key
        key_hash = hashlib.sha256(key_components.encode()).hexdigest()
        
        return key_hash
    
    def _get_cache_path(self, cache_key: str, model_name: str) -> Path:
        """Get file path for cache key."""
        # Sanitize model name for filesystem
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        
        # Organize by model name subdirectory
        model_dir = self.cache_dir / safe_model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        return model_dir / f"{cache_key}.pkl"
    
    def load(
        self, 
        model_name: str, 
        prompts: List[str],
        normalized: bool = True
    ) -> Optional[Tuple[torch.Tensor, List[str]]]:
        """
        Load cached embeddings for given model and prompts.
        
        Args:
            model_name: CLIP model identifier
            prompts: list of text prompts (order matters)
            normalized: whether embeddings should be L2-normalized
        
        Returns:
            Tuple of (embeddings_tensor, prompt_owners) or None if cache miss
        """
        if not self._enabled:
            return None
        
        cache_key = self._generate_cache_key(model_name, prompts, normalized)
        cache_path = self._get_cache_path(cache_key, model_name)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            
            # Validate cache structure
            if not isinstance(cached_data, dict):
                return None
            
            if "embeddings" not in cached_data or "prompts" not in cached_data:
                return None
            
            # Verify prompts match (safeguard against hash collision)
            if cached_data["prompts"] != prompts:
                return None
            
            embeddings = cached_data["embeddings"]
            prompt_owners = cached_data.get("prompt_owners", prompts)
            
            return embeddings, prompt_owners
        
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None
    
    def save(
        self, 
        model_name: str, 
        prompts: List[str],
        embeddings: torch.Tensor,
        prompt_owners: Optional[List[str]] = None,
        normalized: bool = True
    ) -> None:
        """
        Save embeddings to cache.
        
        Args:
            model_name: CLIP model identifier
            prompts: list of text prompts (order matters)
            embeddings: tensor of shape (N, embedding_dim)
            prompt_owners: optional list mapping embeddings to class names
            normalized: whether embeddings are L2-normalized
        """
        if not self._enabled:
            return
        
        cache_key = self._generate_cache_key(model_name, prompts, normalized)
        cache_path = self._get_cache_path(cache_key, model_name)
        
        # Move embeddings to CPU for storage
        embeddings_cpu = embeddings.cpu()
        
        cached_data = {
            "model_name": model_name,
            "prompts": prompts,
            "embeddings": embeddings_cpu,
            "prompt_owners": prompt_owners if prompt_owners else prompts,
            "normalized": normalized,
        }
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def clear(self, model_name: Optional[str] = None) -> int:
        """
        Clear cache files.
        
        Args:
            model_name: if provided, only clear cache for this model
                       if None, clear all cache files
        
        Returns:
            number of files deleted
        """
        count = 0
        
        if model_name:
            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            model_dir = self.cache_dir / safe_model_name
            
            if model_dir.exists():
                for cache_file in model_dir.glob("*.pkl"):
                    cache_file.unlink()
                    count += 1
        else:
            for cache_file in self.cache_dir.rglob("*.pkl"):
                cache_file.unlink()
                count += 1
        
        return count
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cache usage."""
        total_files = 0
        total_size_bytes = 0
        models = set()
        
        for cache_file in self.cache_dir.rglob("*.pkl"):
            total_files += 1
            total_size_bytes += cache_file.stat().st_size
            models.add(cache_file.parent.name)
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "num_models": len(models),
            "model_names": sorted(models),
            "cache_dir": str(self.cache_dir),
        }
    
    def disable(self) -> None:
        """Disable cache (will always return None on load, no-op on save)."""
        self._enabled = False
    
    def enable(self) -> None:
        """Enable cache."""
        self._enabled = True


# Global cache instance for convenience
_global_cache: Optional[EmbeddingCache] = None


def get_global_cache(cache_dir: str = ".cache/clip_embeddings") -> EmbeddingCache:
    """Get or create global cache instance."""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = EmbeddingCache(cache_dir=cache_dir)
    
    return _global_cache
