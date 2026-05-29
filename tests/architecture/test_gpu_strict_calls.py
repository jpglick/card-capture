"""Static AST scan: forbidden CPU/IO calls inside files tagged GPU-resident.

Phase 1: advisory (no files in scope, always passes).
Phase 2: populates `pyproject.toml [tool.gpu_strict_lint] files` and tightens.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh).get("tool", {}).get("gpu_strict_lint", {})


def _resolve_attr_chain(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _scan_file(path: Path, forbidden: set[str]) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _resolve_attr_chain(node.func)
            for f in forbidden:
                # Match exact qualified name (cv2.imread) or method name suffix (.cpu / .numpy)
                if name == f or name.endswith("." + f.split(".")[-1]) and f.split(".")[-1] in {"cpu", "numpy"}:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}")
                    break
    return violations


def test_no_forbidden_calls_in_gpu_files():
    cfg = _load_config()
    forbidden = set(cfg.get("forbidden_calls", []))
    files = []
    for glob in cfg.get("files", []):
        files.extend(REPO_ROOT.glob(glob))
    violations: list[str] = []
    for p in files:
        violations.extend(_scan_file(p, forbidden))
    assert not violations, "\n".join(violations)
