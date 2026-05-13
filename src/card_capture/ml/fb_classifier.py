"""Front/Back side classifier model architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class FBClassifier(torch.nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Use a lightweight ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Binary classification (front vs back)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_ftrs, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_transforms():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
