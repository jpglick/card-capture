"""Training pipeline for the Front/Back side classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from card_capture.data.connection import read_connection
from card_capture.data.sql_queries import ML_TRAIN_FB_DATASET
from card_capture.ml.fb_classifier import FBClassifier, get_transforms


class FBDataset(Dataset):
    def __init__(self, db_path: Path, transform=None):
        self.db_path = db_path
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        
        with read_connection(db_path) as conn:
            # We need to join fb_labels with card_views to get the image_path
            rows = conn.execute(ML_TRAIN_FB_DATASET).fetchall()
            
            side_map = {'front': 0, 'back': 1}
            for r in rows:
                if Path(r['image_path']).exists():
                    self.samples.append((r['image_path'], side_map[r['side']]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def train_fb(*, db_path: Path, out_path: Path, epochs: int = 10, lr: float = 0.001):
    dataset = FBDataset(db_path, transform=get_transforms())
    if len(dataset) < 10:
        print("Not enough data to train. Need at least 10 samples.")
        return
        
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FBClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")
    
    # Register in DB
    from card_capture.ml.registry import register_model
    register_model(
        db_path=db_path,
        model_name="fb_classifier",
        training_set_hash=f"n={len(dataset)}", # simplified
        eval_metrics={"loss": total_loss/len(loader)},
        checkpoint_path=str(out_path)
    )
