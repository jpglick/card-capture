"""Tests for src/card_capture/ml/registry.py"""
import sqlite3

import pytest
from pathlib import Path

from card_capture.ml.registry import get_latest, list_models, register_model
from migrations.run_migrations import apply_migrations


def _fresh_db(tmp_path: Path) -> Path:
    db = tmp_path / "cards.sqlite"
    sqlite3.connect(db).close()
    apply_migrations(db)
    return db


def test_register_and_fetch(tmp_path: Path):
    db = _fresh_db(tmp_path)
    v_id = register_model(
        db_path=db,
        model_name="fb_classifier",
        training_set_hash="abc",
        eval_metrics={"val_acc": 0.91},
        checkpoint_path="models/fb_classifier/v1.pt",
    )
    assert isinstance(v_id, int) and v_id > 0

    latest = get_latest(db_path=db, model_name="fb_classifier")
    assert latest is not None
    assert latest.version_id == v_id
    assert latest.model_name == "fb_classifier"
    assert latest.eval_metrics["val_acc"] == pytest.approx(0.91)


def test_get_latest_returns_none_when_missing(tmp_path: Path):
    db = _fresh_db(tmp_path)
    assert get_latest(db_path=db, model_name="nonexistent") is None


def test_list_models_returns_all(tmp_path: Path):
    db = _fresh_db(tmp_path)
    register_model(db_path=db, model_name="m1", training_set_hash="h1", eval_metrics={}, checkpoint_path="p1")
    register_model(db_path=db, model_name="m2", training_set_hash="h2", eval_metrics={}, checkpoint_path="p2")
    models = list_models(db_path=db)
    names = {m.model_name for m in models}
    assert names == {"m1", "m2"}


def test_unique_training_set_per_name(tmp_path: Path):
    db = _fresh_db(tmp_path)
    register_model(
        db_path=db, model_name="fb", training_set_hash="h", eval_metrics={}, checkpoint_path="p"
    )
    with pytest.raises(Exception):
        register_model(
            db_path=db, model_name="fb", training_set_hash="h", eval_metrics={}, checkpoint_path="p2"
        )
