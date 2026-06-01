from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


def _build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model


def _resolve_device(prefer: str = "auto") -> torch.device:
    _VALID = {"auto", "cpu", "mps"}
    if prefer not in _VALID:
        raise ValueError(f"Invalid device {prefer!r}. Must be one of {_VALID}")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps":
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PresenceClassifier:
    """Binary classifier: returns P(card present) for an input frame/patch."""

    def __init__(self, weights_path: Path, device: str = "auto"):
        self.device = _resolve_device(device)
        self.model = _build_model().to(self.device)
        ckpt = torch.load(str(weights_path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _prep(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.tx(rgb)

    def score(self, frame_bgr: np.ndarray) -> float:
        """Return P(card present) in [0, 1]."""
        with torch.no_grad():
            x = self._prep(frame_bgr).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(x), dim=1)
            return float(probs[0, 1].item())

    def score_batch(self, frames_bgr: List[np.ndarray]) -> List[float]:
        """Return list of P(card present) scores, one per frame."""
        if not frames_bgr:
            return []
        with torch.no_grad():
            xs = torch.stack([self._prep(f) for f in frames_bgr]).to(self.device)
            probs = torch.softmax(self.model(xs), dim=1)
            return probs[:, 1].cpu().tolist()
