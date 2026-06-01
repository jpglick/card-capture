"""Apply-all + subset-bisection driver.

`find_culprits` is the pure core: given an ordered change list and a predicate
`is_good(subset)` (True when that subset keeps the gate green), it returns the
minimal set of changes that must be removed for the remainder to be good. It
handles MULTIPLE independent culprits via recursive split (delta-debugging).

The apply/revert glue (`drive`) maps each change to a git commit and wires the
predicate to gate.run_gate(); it is exercised manually in Phase 6, not unit
tested, because it mutates the working tree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]


def find_culprits(changes: list[str], is_good: Callable[[list[str]], bool]) -> list[str]:
    """Return every change that, applied alone, breaks the gate.

    Invariant: applying NONE of the changes is good (the bisect base is green),
    and badness is monotonic — any subset containing a culprit tests bad.
    `is_good(s)` answers "is the gate green with exactly subset s applied on the
    green base?". We test each subset IN ISOLATION (no shared context) so an
    innocent change is never tainted by a culprit sitting in its context. This
    recovers ALL independent culprits.
    """

    def search(candidates: list[str]) -> list[str]:
        if not candidates:
            return []
        if is_good(candidates):          # subset clean -> no culprits inside
            return []
        if len(candidates) == 1:         # bad singleton -> it is a culprit
            return list(candidates)
        mid = len(candidates) // 2
        return search(candidates[:mid]) + search(candidates[mid:])

    found = search(list(changes))
    seen, ordered = set(), []
    for c in found:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# ---- working-tree glue (manual use in Phase 6) -------------------------------

def load_manifest() -> list[dict]:
    return json.loads((REPO / "scripts/deadcode/manifest.json").read_text())


def removable_ids(manifest: list[dict]) -> list[str]:
    return [e["id"] for e in manifest if e["decision"] == "remove"]
