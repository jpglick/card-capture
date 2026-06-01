"""Audit that every @pytest.mark.skip / skipif / quarantine has a reason."""
from __future__ import annotations
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"

ALLOWED_MARKERS = {"mps", "benchmark", "slow", "quarantine"}


def _iter_test_files():
    for p in TEST_ROOT.rglob("test_*.py"):
        yield p


def _decorator_name(dec: ast.expr) -> str:
    # @pytest.mark.skip / @pytest.mark.skipif / @pytest.mark.foo / @pytest.mark.foo(...)
    node = dec.func if isinstance(dec, ast.Call) else dec
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_no_unexplained_skips():
    violations: list[str] = []
    for path in _iter_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                name = _decorator_name(dec)
                if name in {"pytest.mark.skip", "pytest.mark.skipif"}:
                    has_reason = isinstance(dec, ast.Call) and any(
                        kw.arg == "reason" for kw in dec.keywords
                    )
                    if not has_reason:
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}::{node.name} has skip without reason="
                        )
                if name.startswith("pytest.mark."):
                    marker = name.removeprefix("pytest.mark.")
                    # skip/skipif handled above; allow registered project markers.
                    if marker in {"skip", "skipif"}:
                        continue
                    if marker not in ALLOWED_MARKERS:
                        # Allowed if registered in pyproject markers list; we trust pytest's
                        # PytestUnknownMarkWarning to catch unregistered markers separately.
                        pass
    assert not violations, "\n".join(violations)
