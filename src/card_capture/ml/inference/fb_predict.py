"""Inference wrapper for the Front/Back side classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import cv2
import numpy as np
from PIL import Image

from ..fb_classifier import FBClassifier, get_transforms
from ..scaffolding import pick_device


class FBPredictor:
    def __init__(self, checkpoint_path: str | Path | None = None):
        self.device = pick_device()
        self.model = FBClassifier(pretrained=(checkpoint_path is None))
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading F/B classifier from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
            else:
                self.model.load_state_dict(ckpt)
                
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_transforms()

    @torch.no_grad()
    def predict(self, image: np.ndarray | Image.Image | str | Path) -> Tuple[str, float]:
        """Predict if an image is the Front or Back of a card.
        
        Returns:
            Tuple of (label, confidence) where label is "front" or "back".
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.convert("RGB")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        
        conf, idx = torch.max(probs, dim=1)
        label = "front" if idx.item() == 0 else "back"
        
        return label, float(conf.item())
