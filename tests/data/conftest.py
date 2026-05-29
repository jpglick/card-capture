"""Shared fixtures for repository tests.

`prod_db` returns a Path to a freshly-created SQLite database with every
production migration applied. Repository tests use this instead of inlining
their own toy schemas.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from migrations.run_migrations import apply_migrations


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "migrations"


def _apply_migrations(db_path: Path) -> None:
    apply_migrations(db_path)


@pytest.fixture
def prod_db(tmp_path: Path) -> Path:
    """A SQLite database with every production migration applied."""
    db = tmp_path / "cards.sqlite"
    _apply_migrations(db)
    return db
