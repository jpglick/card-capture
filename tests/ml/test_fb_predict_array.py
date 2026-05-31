import numpy as np
import torch
import pytest
import cv2
from PIL import Image
from unittest.mock import MagicMock, patch
from card_capture.ml.inference.fb_predict import FBPredictor

def test_predict_array_parity(tmp_path):
    # 1. Setup dummy checkpoint
    ckpt_path = tmp_path / "dummy_fb.pt"
    torch.save({"state_dict": {}}, ckpt_path)
    
    img_array_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_path = str(tmp_path / "test_img.png")
    cv2.imwrite(img_path, img_array_bgr)

    # 2. Mock
    with patch("card_capture.ml.inference.fb_predict.FBClassifier") as mock_cls, \
         patch("torch.load") as mock_load, \
         patch("card_capture.ml.inference.fb_predict.pick_device") as mock_pick:
        
        mock_pick.return_value = "cpu"
        mock_model = MagicMock()
        # Mock forward pass: logits [10.0, -10.0] -> index 0 (front)
        mock_model.return_value = torch.tensor([[10.0, -10.0]])
        mock_cls.return_value = mock_model
        
        predictor = FBPredictor(checkpoint_path=ckpt_path)
        
        assert hasattr(predictor, "predict_array"), "FBPredictor should have predict_array method"
        
        res1 = predictor.predict(img_path)
        img_array_rgb = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
        res2 = predictor.predict_array(img_array_rgb)
        
        assert res1 == res2
        assert res1[0] == "front"
