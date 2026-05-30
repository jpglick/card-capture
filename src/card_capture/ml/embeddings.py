"""ReID embedding generation for card instances."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from .models.dino_embedder import DinoEmbedder


# Global shared embedder to avoid reloading models per-call
_SHARED_EMBEDDER: Optional[DinoEmbedder] = None

def compute_reid_embedding_array(
    image_bgr: np.ndarray,
    variant: str = "vits14",
    device: Optional[str] = None
) -> np.ndarray:
    """Compute an L2-normalized visual embedding for an in-memory BGR ndarray.
    
    Used by the V5.5 in-process refine / store stages to avoid disk round-trips.
    """
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None or _SHARED_EMBEDDER.variant != variant:
        _SHARED_EMBEDDER = DinoEmbedder(variant=variant, device=device)
        
    # DinoEmbedder.embed_array expects RGB
    import cv2
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    emb = _SHARED_EMBEDDER.embed_array(image_rgb)
    return emb.squeeze(0).cpu().numpy().astype("float32")

def compute_reid_embedding(
    image: Union[np.ndarray, Image.Image, str, Path],
    variant: str = "vits14",
    device: Optional[str] = None
) -> np.ndarray:
    """Compute an L2-normalized visual embedding for a card crop.
    
    Delegates to compute_reid_embedding_array for ndarrays or loads from disk.
    
    Returns:
        1D float32 array (e.g. 384-dim for vits14).
    """
    if isinstance(image, (str, Path)):
        import cv2
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"compute_reid_embedding: {image!r} not readable")
        return compute_reid_embedding_array(img, variant=variant, device=device)
    
    if isinstance(image, np.ndarray):
        return compute_reid_embedding_array(image, variant=variant, device=device)
        
    # Handle PIL Image
    if isinstance(image, Image.Image):
        # PIL doesn't support 'BGR' mode; convert to RGB then flip channels
        rgb_array = np.array(image.convert("RGB"))
        import cv2
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return compute_reid_embedding_array(bgr_array, variant=variant, device=device)

    raise ValueError(f"Unsupported image type: {type(image)}")
