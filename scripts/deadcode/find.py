"""Run vulture over the repo and emit normalized findings.json.

Scan roots include consumers (tests/, harness/, scripts/, app/) so code they
reference is NOT flagged as unused. We only ACT on findings under src/ (and
never under app/) — see enrich.py's filtering.
"""
from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

from scripts.deadcode.models import Finding

REPO = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ["src", "harness", "scripts", "tests", "app"]
WHITELIST = "scripts/deadcode/whitelist.py"

# vulture line format: path:line: unused <kind> 'name' (NN% confidence)
_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): unused (?P<kind>[\w ]+) '(?P<name>[^']+)' "
    r"\((?P<conf>\d+)% confidence\)"
)


def run_vulture(min_confidence: int = 0) -> str:
    cmd = [
        sys.executable, "-m", "vulture",
        *SCAN_ROOTS, WHITELIST,
        "--min-confidence", str(min_confidence),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    # vulture exits 3 when it finds dead code; that is success for us.
    if proc.returncode not in (0, 3):
        raise RuntimeError(f"vulture failed ({proc.returncode}): {proc.stderr}")
    return proc.stdout


def parse(output: str) -> list[Finding]:
    findings: list[Finding] = []
    for line in output.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        findings.append(Finding(
            path=m.group("path"),
            lineno=int(m.group("line")),
            name=m.group("name"),
            kind=m.group("kind").strip(),
            confidence=int(m.group("conf")),
        ))
    return findings


def main() -> None:
    findings = parse(run_vulture())
    out = REPO / "scripts/deadcode/findings.json"
    out.write_text(json.dumps([dataclasses.asdict(f) for f in findings], indent=2))
    print(f"Wrote {len(findings)} findings to {out}")


if __name__ == "__main__":
    main()
