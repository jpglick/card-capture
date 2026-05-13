"""Deterministic training-loop scaffold.

Provides:
- ``pick_device()``   — MPS > CUDA > CPU preference.
- ``set_seed(seed)``  — reproducible random state across Python, NumPy, PyTorch.
- ``train_one_epoch`` — generic one-epoch supervised training loop.
"""
import os
import random

import numpy as np
import torch


def pick_device() -> torch.device:
    """Return the best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, *, device: torch.device) -> float:
    """Run one full pass over *loader* and return the mean loss.

    Args:
        model:     Any ``torch.nn.Module``.
        loader:    Iterable of ``(x, y)`` batches.
        optimizer: Any ``torch.optim.Optimizer``.
        loss_fn:   Callable ``(logits, targets) -> scalar tensor``.
        device:    Target compute device.

    Returns:
        Sample-weighted mean loss for the epoch (scalar ``float``).
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    if n_samples == 0:
        return 0.0
    return total_loss / n_samples
