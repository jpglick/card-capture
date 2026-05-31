import numpy as np
import torch
import pytest
import cv2
from PIL import Image
from unittest.mock import MagicMock, patch
from card_capture.ml.models.dino_embedder import DinoEmbedder

def test_embed_array_parity(tmp_path):
    # 1. Setup
    img_array_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_path = str(tmp_path / "test_img.png")
    
    # Save as RGB for PIL parity
    Image.fromarray(img_array_rgb).save(img_path)
    
    # Also load via cv2 to simulate the new path-based delegate behavior
    # cv2.imread returns BGR
    img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img_array_bgr)

    # 2. Mock
    with patch("torch.hub.load") as mock_load:
        mock_model = MagicMock()
        # Return deterministic values
        mock_model.return_value = torch.ones(1, 384)
        mock_load.return_value = mock_model
        
        embedder = DinoEmbedder(variant="vits14", device="cpu")
        
        # 3. Verify embed_array exists and produces same result as embed_image used to
        # (Actually first run should fail because embed_array is missing)
        
        # Current behavior (before refactor):
        # embed_image(path) -> PIL.Image.open(path).convert("RGB") -> self.transform
        
        # New behavior (after refactor):
        # embed_image(path) -> cv2.imread(path) -> cv2.cvtColor(BGR2RGB) -> embed_array(array)
        
        # Let's see it fail
        assert hasattr(embedder, "embed_array"), "DinoEmbedder should have embed_array method"
        
        emb1 = embedder.embed_image(img_path)
        emb2 = embedder.embed_array(img_array_rgb)
        
        torch.testing.assert_close(emb1, emb2)
