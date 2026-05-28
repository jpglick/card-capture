"""Static AST scan: raw SQL string literals outside card_capture.data and migrations.

Phase 1: advisory.
Phase 4: blocking (after data layer migration).
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ROOTS = (
    "src/card_capture/data/",
    "migrations/",
    "tests/",            # tests may contain raw SQL fixtures
    "harness/schema.py",
)

# Heuristic: a string literal that begins with SELECT/INSERT/UPDATE/DELETE/CREATE/PRAGMA/ALTER/DROP
SQL_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|PRAGMA|ALTER|DROP|WITH)\b", re.IGNORECASE
)


def _iter_python_files():
    for root in ("src", "app", "pipeline", "harness"):
        for p in (REPO_ROOT / root).rglob("*.py"):
            rel = str(p.relative_to(REPO_ROOT))
            if any(rel.startswith(a) for a in ALLOWED_ROOTS):
                continue
            yield p


def _scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if SQL_RE.match(node.value):
                out.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: raw SQL literal")
    return out


@pytest.mark.skipif(
    os.environ.get("V55_RAW_SQL_BLOCKING") != "1",
    reason="Phase 1 advisory: set V55_RAW_SQL_BLOCKING=1 to fail on violations",
)
def test_no_raw_sql_outside_data_blocking():
    violations: list[str] = []
    for p in _iter_python_files():
        violations.extend(_scan(p))
    assert not violations, "\n".join(violations)


def test_raw_sql_advisory():
    print("=== Raw-SQL scan (advisory) ===")
    for p in _iter_python_files():
        for v in _scan(p):
            print(v)
