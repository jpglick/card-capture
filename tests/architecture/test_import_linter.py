"""Import Linter contracts. Advisory in Phase 1, blocking in Phase 2."""
from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.skipif(
    os.environ.get("V55_IMPORT_LINT_BLOCKING") != "1",
    reason="Phase 1 advisory mode: set V55_IMPORT_LINT_BLOCKING=1 to fail on violations",
)
def test_import_contracts_blocking():
    result = subprocess.run(
        ["lint-imports"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.fail(f"Import Linter violations:\n{result.stdout}\n{result.stderr}")


def test_import_contracts_advisory():
    """Run Import Linter and print results without failing (Phase 1)."""
    try:
        result = subprocess.run(
            ["lint-imports"], capture_output=True, text=True, check=False
        )
        print("=== Import Linter (advisory) ===")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except FileNotFoundError:
        print("lint-imports not found; skipping advisory check")
    # No assertion: advisory only in Phase 1.
