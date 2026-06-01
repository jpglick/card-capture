"""Train MobileNetV3-Small presence classifier.

Usage (via CLI):
    card-capture train presence --data data/presence_dataset --out models/presence_classifier.pt

Usage (direct):
    python -m card_capture.training.presence \
        --data data/presence_dataset --out models/presence_classifier.pt \
        --epochs 8 --batch-size 64
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class _PresenceDataset(Dataset):
    def __init__(self, root: Path, train: bool):
        self.samples: list[Tuple[Path, int]] = []
        for label, sub in enumerate(["negatives", "positives"]):  # negative=0, positive=1
            for path in sorted((root / sub).glob("*.jpg")):
                self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"no samples under {root}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            *([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ] if train else []),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"could not read {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), label


def _build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    data_dir: Path,
    out_path: Path,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.15,
) -> float:
    """Train the classifier and return best val_acc."""
    device = _device()
    print(f"device: {device}")

    ds = _PresenceDataset(data_dir, train=True)
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = _build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += int(y.size(0))
        acc = correct / total if total else 0.0
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "val_acc": acc}, out_path)
            print(f"  → saved {out_path} (val_acc={acc:.4f})")

    print(f"best val_acc={best_acc:.4f}")
    return best_acc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small presence classifier")
    parser.add_argument("--data", type=Path, required=True, help="Dataset root (contains positives/ and negatives/)")
    parser.add_argument("--out", type=Path, default=Path("models/presence_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args(argv)
    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, val_split=args.val_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
