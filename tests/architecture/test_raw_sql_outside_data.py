"""Static AST scan: raw SQL string literals outside card_capture.data and migrations.

Blocking by default at Phase E. Allowed roots are listed below; adding a new
root requires a paired plan amendment.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ROOTS = (
    "src/card_capture/data/",
    "migrations/",
    "tests/",            # tests may contain raw SQL fixtures
    "harness/schema.py",
)

SQL_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|PRAGMA|ALTER|DROP|WITH)\b",
    re.IGNORECASE,
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


def test_no_raw_sql_outside_data() -> None:
    violations: list[str] = []
    for p in _iter_python_files():
        violations.extend(_scan(p))
    assert not violations, "\n".join(violations)
