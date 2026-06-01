"""DINOv2 embedder for card crops.

Uses Facebook's DINOv2 self-supervised models to produce rich visual
embeddings suitable for card identity matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import numpy as np
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
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
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
    def embed_array(self, image_array: np.ndarray | Image.Image) -> torch.Tensor:
        """Compute L2-normalized embedding for a single numpy array (RGB) or PIL Image."""
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array)
        else:
            image = image_array

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        
        # L2 Normalize for cosine similarity via inner product
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        return embedding / norm

    @torch.no_grad()
    def embed_image(self, image: Image.Image | Path | str) -> torch.Tensor:
        """Compute L2-normalized embedding for a single image."""
        if isinstance(image, (str, Path)):
            import cv2
            image_array = cv2.imread(str(image))
            if image_array is None:
                raise FileNotFoundError(f"Could not read image at {image}")
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            return self.embed_array(image_array)

        if isinstance(image, Image.Image):
            return self.embed_array(np.array(image.convert("RGB")))
        
        # If it's already an array-like thing that self.transform can handle
        return self.embed_array(image)

    @torch.no_grad()
    def embed_tensors_batch(self, tensors: torch.Tensor) -> torch.Tensor:
        """Batch-compute embeddings from a (N,3,H,W) float32 [0,1] tensor.
        Optimized for zero-download pipelines where frames are already on GPU.
        Input must be RGB and already resized to 224x224.
        """
        if tensors.numel() == 0:
            return torch.empty((0, self.dim), device=self.device)
            
        # Manually apply normalization (ImageNet stats)
        # Note: input is [0,1], so mean/std are scaled accordingly
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        normalized = (tensors - mean) / std
        
        embeddings = self.model(normalized)
        norm = embeddings.norm(p=2, dim=1, keepdim=True)
        return embeddings / norm

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Batch-compute L2-normalized embeddings."""
        if not images:
            return torch.empty((0, self.dim), device=self.device)

        tensors = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        embeddings = self.model(tensors)
        
        norm = embeddings.norm(p=2, dim=1, keepdim=True)
        return embeddings / norm
