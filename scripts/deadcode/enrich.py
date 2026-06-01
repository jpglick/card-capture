"""Enrich raw vulture findings into a scored, triaged manifest.

Scoring rules (design D-tiers):
  - app/ findings              -> keep (out of scope)
  - confidence >= 90, no refs  -> remove, tier 1
  - confidence >= 90, has refs -> investigate (dynamic usage suspected)
  - 60 <= confidence < 90      -> investigate, tier 2
  - confidence < 60            -> investigate, tier 3
Whole-module detection and non-MPS (tier 0) tagging are layered on top.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

from scripts.deadcode.models import Finding, ManifestEntry

REPO = Path(__file__).resolve().parents[2]


def dynamic_refs(name: str) -> int:
    """Count dynamic-style references vulture cannot see: getattr/string dispatch,
    entry points, relative re-exports. A heuristic, not proof."""
    patterns = [f'"{name}"', f"'{name}'", f"getattr(.*{name}", f"\\b{name}\\b"]
    hits = 0
    for pat in patterns[:3]:  # skip the bare-name pattern (too noisy) for scoring
        proc = subprocess.run(
            ["grep", "-rEl", "--include=*.py", pat, "src", "harness", "scripts", "app"],
            cwd=REPO, capture_output=True, text=True,
        )
        hits += len([ln for ln in proc.stdout.splitlines() if ln])
    return hits


def adjusted_confidence(base: int, dynamic_hits: int) -> int:
    return max(0, base - 25 * dynamic_hits)


def classify(f: Finding, dynamic_hits: int) -> ManifestEntry:
    entry_id = f"{f.path}:{f.lineno}:{f.name}"
    adj = adjusted_confidence(f.confidence, dynamic_hits)

    if f.path.startswith("app/"):
        return ManifestEntry(entry_id, "symbol", f.path, f.name, f.confidence, adj,
                             "keep", 99, "excluded: app/ is out of scope")
    if not f.path.startswith("src/"):
        return ManifestEntry(entry_id, "symbol", f.path, f.name, f.confidence, adj,
                             "keep", 99, "consumer root (tests/harness/scripts): used")

    if f.confidence >= 90 and dynamic_hits == 0:
        return ManifestEntry(entry_id, "symbol", f.path, f.name, f.confidence, adj,
                             "remove", 1, "vulture>=90, no dynamic refs")
    if f.confidence >= 90 and dynamic_hits > 0:
        return ManifestEntry(entry_id, "symbol", f.path, f.name, f.confidence, adj,
                             "investigate", 2, f"dynamic refs found ({dynamic_hits})")
    if f.confidence >= 60:
        return ManifestEntry(entry_id, "symbol", f.path, f.name, f.confidence, adj,
                             "investigate", 2, "medium confidence")
    return ManifestEntry(entry_id, "symbol", f.path, f.name, f.confidence, adj,
                         "investigate", 3, "low confidence")


def build_manifest(findings: list[Finding]) -> list[ManifestEntry]:
    out: list[ManifestEntry] = []
    for f in findings:
        hits = dynamic_refs(f.name) if f.path.startswith("src/") else 0
        out.append(classify(f, hits))
    return out


def main() -> None:
    findings_path = REPO / "scripts/deadcode/findings.json"
    if not findings_path.exists():
        print(f"Error: {findings_path} not found. Run find.py first.")
        return
    
    findings = [Finding(**d) for d in json.loads(findings_path.read_text())]
    entries = build_manifest(findings)
    out = REPO / "scripts/deadcode/manifest.json"
    out.write_text(json.dumps([dataclasses.asdict(e) for e in entries], indent=2))
    counts = {}
    for e in entries:
        counts[e.decision] = counts.get(e.decision, 0) + 1
    print(f"Wrote {len(entries)} entries to {out}: {counts}")


if __name__ == "__main__":
    main()
