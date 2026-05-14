"""ReID embedding generation for card instances."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from .models.dino_embedder import DinoEmbedder


# Global shared embedder to avoid reloading models per-call
_SHARED_EMBEDDER: Optional[DinoEmbedder] = None

def compute_reid_embedding(
    image: Union[np.ndarray, Image.Image, str, Path],
    variant: str = "vits14",
    device: Optional[str] = None
) -> np.ndarray:
    """Compute an L2-normalized visual embedding for a card crop.
    
    Uses DINOv2 ViT backbone.
    
    Returns:
        1D float32 array (e.g. 384-dim for vits14).
    """
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None or _SHARED_EMBEDDER.variant != variant:
        _SHARED_EMBEDDER = DinoEmbedder(variant=variant, device=device)
        
    emb = _SHARED_EMBEDDER.embed_image(image)
    # embed_image returns [1, D] tensor on self.device
    return emb.squeeze(0).cpu().numpy().astype("float32")
