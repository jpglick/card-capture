"""Metaflow is entirely removed in V5.5."""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_metaflow_not_imported():
    for root in ("src", "app", "harness"):
        for p in (REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(p.read_text(), filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        assert not name.name.startswith("metaflow"), f"{p} imports metaflow"
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("metaflow"), f"{p} imports metaflow"
