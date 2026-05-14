"""Inference wrapper for the Front/Back side classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Union

import torch
import cv2
import numpy as np
from PIL import Image

from ..fb_classifier import FBClassifier, get_transforms
from ..scaffolding import pick_device
from ..errors import UntrainedModelError


class FBPredictor:
    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None):
        if not self.is_available(checkpoint_path):
            raise UntrainedModelError(
                f"FBPredictor requires a trained checkpoint; got "
                f"{checkpoint_path!r}. Use FBPredictor.is_available() "
                f"to check before instantiation."
            )

        self.device = pick_device()
        self.model = FBClassifier(pretrained=False)
        
        print(f"Loading F/B classifier from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)
                
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_transforms()

    @classmethod
    def is_available(cls, checkpoint_path: Optional[Union[str, Path]]) -> bool:
        """Return True if a trained checkpoint exists at the given path."""
        if checkpoint_path is None:
            return False
        p = Path(checkpoint_path)
        return p.exists() and p.is_file()

    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Tuple[str, float]:
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
