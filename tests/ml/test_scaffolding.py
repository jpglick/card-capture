"""Tests for src/card_capture/ml/scaffolding.py"""
import torch

from card_capture.ml.scaffolding import pick_device, set_seed, train_one_epoch


def test_pick_device_returns_torch_device():
    d = pick_device()
    assert isinstance(d, torch.device)


def test_set_seed_makes_init_reproducible():
    set_seed(42)
    a = torch.randn(8, 8)
    set_seed(42)
    b = torch.randn(8, 8)
    assert torch.equal(a, b)


def test_train_one_epoch_returns_loss():
    model = torch.nn.Linear(4, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = [(x[i : i + 4], y[i : i + 4]) for i in range(0, 16, 4)]
    avg_loss = train_one_epoch(model, loader, optim, loss_fn, device=torch.device("cpu"))
    assert avg_loss > 0
