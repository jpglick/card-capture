"""CLI validator for truth.json files.

Usage::

    python -m harness.validator path/to/truth.json [path/to/another.json ...]

Exits with code 0 if all files are valid, 1 otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from harness.schema import TruthFile


def validate_file(path: Path) -> int:
    """Validate a single truth.json file.

    Returns 0 on success, 1 on failure (error printed to stderr).
    """
    try:
        TruthFile.model_validate_json(path.read_text())
    except ValidationError as exc:
        print(f"{path}: invalid\n{exc}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, OSError) as exc:
        print(f"{path}: could not read — {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m harness.validator <truth.json> [...]", file=sys.stderr)
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    exit_code = 0
    for p in paths:
        exit_code |= validate_file(p)
    sys.exit(exit_code)
