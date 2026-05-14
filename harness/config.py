"""Harness configuration helpers.

Provides load_pipeline_config() which locates and loads the current
pipeline configuration snapshot for archival in regression baselines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CANDIDATE_NAMES = [
    "card_capture_config.json",
    "harness_config.json",
]


def load_pipeline_config(search_dir: Path | None = None) -> dict[str, Any]:
    """Load the current pipeline configuration from disk.

    Searches ``search_dir`` (default: cwd) for known config file names.
    Returns an empty dict if no config file is found or if the file cannot
    be parsed.

    Parameters
    ----------
    search_dir:
        Directory to search. Defaults to the current working directory.

    Returns
    -------
    dict containing the pipeline configuration, or ``{}`` if not found.
    """
    base = Path(search_dir) if search_dir is not None else Path(".")
    for name in _CANDIDATE_NAMES:
        candidate = base / name
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                return {}
    return {}
