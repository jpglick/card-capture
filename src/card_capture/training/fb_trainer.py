"""Train the Front/Back classifier (FBClassifier / ResNet-18) from fb_labels.

Uses the same FBClassifier architecture that FBPredictor loads at inference time,
so training and production inference are guaranteed to be compatible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from card_capture.data.connection import read_connection
from card_capture.data.sql_queries import TRAINING_FB_LABELED_ROWS

# 'uncertain' and 'no_card' are excluded — not useful for the FB task
_LABEL_MAP = {"front": 0, "back": 1}


class _FBDataset(Dataset):
    def __init__(self, rows: list[dict], tx):
        self.rows = rows
        self.tx = tx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(row["image_path"])
        if img is None:
            img = np.zeros((750, 1050, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tx(rgb)
        y = torch.tensor(_LABEL_MAP[row["label"]], dtype=torch.long)
        return x, y


def train_fb(
    db_path: Path,
    output_path: Path,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    progress_cb=None,
) -> dict:
    """Train and save the FB classifier. Returns eval metrics dict."""
    rows = _load_labeled_rows(db_path)
    if len(rows) < 10:
        raise ValueError(f"Need at least 10 labeled front/back samples, got {len(rows)}")

    val_rows = [r for r in rows if r["id"] % 5 == 0]
    train_rows = [r for r in rows if r["id"] % 5 != 0]

    tx_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tx_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = _FBDataset(train_rows, tx_train)
    val_ds = _FBDataset(val_rows, tx_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = _get_device()
    from card_capture.ml.fb_classifier import FBClassifier
    model = FBClassifier(pretrained=True)
    model = model.to(device)

    # Freeze backbone, only train the final fc layer first
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.fc.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.backbone.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        if epoch == epochs // 2 + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=lr * 0.1)

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

        acc = _evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if progress_cb:
            progress_cb({"epoch": epoch, "total_epochs": epochs, "val_accuracy": acc})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive = output_path.parent / "archive"
    archive.mkdir(exist_ok=True)
    if output_path.exists():
        import shutil, time
        shutil.copy(output_path, archive / f"fb_classifier_{int(time.time())}.pt")

    torch.save({"state_dict": best_state}, str(output_path))
    return {"accuracy": round(best_acc, 4), "val_samples": len(val_rows)}


def _load_labeled_rows(db_path: Path) -> list[dict]:
    with read_connection(db_path) as conn:
        rows = conn.execute(TRAINING_FB_LABELED_ROWS).fetchall()
    return [dict(r) for r in rows]


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
