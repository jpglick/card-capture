"""DINOv2 embedder for card crops.

Uses Facebook's DINOv2 self-supervised models to produce rich visual
embeddings suitable for card identity matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

DINO_VARIANT = Literal["vits14", "vitb14", "vitl14", "vitg14"]


class DinoEmbedder:
    """Wrapper around DINOv2 hub models."""

    def __init__(
        self,
        variant: DINO_VARIANT = "vits14",
        device: str | None = None,
    ):
        self.variant = variant
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading DINOv2 {variant} on {self.device}...")
        self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{variant}")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Determine embedding dimension
        self.dim = 384 if variant == "vits14" else 768 if variant == "vitb14" else 1024 if variant == "vitl14" else 1536

    @torch.no_grad()
    def embed_image(self, image: Image.Image | Path | str) -> torch.Tensor:
        """Compute L2-normalized embedding for a single image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        
        # L2 Normalize for cosine similarity via inner product
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        return embedding / norm

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Batch-compute L2-normalized embeddings."""
        if not images:
            return torch.empty((0, self.dim), device=self.device)

        tensors = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        embeddings = self.model(tensors)
        
        norm = embeddings.norm(p=2, dim=1, keepdim=True)
        return embeddings / norm
