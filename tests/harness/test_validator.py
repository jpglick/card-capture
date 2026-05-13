"""Tests for harness.validator — truth.json CLI validator."""
import subprocess
import sys
from pathlib import Path

FIX = Path("tests/harness/fixtures/truth_examples")


def _run_validator(*paths: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "harness.validator", *(str(p) for p in paths)],
        capture_output=True,
        text=True,
    )


def test_validator_passes_on_valid():
    result = _run_validator(FIX / "minimal_valid.json")
    assert result.returncode == 0, result.stderr


def test_validator_passes_on_optional_fields():
    result = _run_validator(FIX / "with_optional_fields.json")
    assert result.returncode == 0, result.stderr


def test_validator_fails_on_missing_required_field():
    result = _run_validator(FIX / "missing_required.json")
    assert result.returncode != 0
    assert "physical_card_key" in result.stderr


def test_validator_multiple_files_all_valid():
    result = _run_validator(
        FIX / "minimal_valid.json",
        FIX / "with_optional_fields.json",
    )
    assert result.returncode == 0, result.stderr


def test_validator_multiple_files_one_invalid_returns_nonzero():
    result = _run_validator(
        FIX / "minimal_valid.json",
        FIX / "missing_required.json",
    )
    assert result.returncode != 0


def test_validator_nonexistent_file_returns_nonzero():
    result = _run_validator(Path("tests/harness/fixtures/truth_examples/no_such_file.json"))
    assert result.returncode != 0
