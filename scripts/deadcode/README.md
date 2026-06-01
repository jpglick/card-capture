# Dead-code toolchain runbook

1. `python -m scripts.deadcode.find`     # vulture -> findings.json
2. `python -m scripts.deadcode.enrich`   # findings.json -> manifest.json
3. Triage manifest.json by hand: confirm tier-1 `remove`, resolve every
   `investigate` to `remove`/`keep`, add tier-0 non-MPS entries.
4. Apply every `remove` entry as ONE git commit each (symbol deletion or whole
   file/dir removal), recording the commit SHA in the entry.
5. `python -m scripts.deadcode.gate`     # full gate on the all-applied tree
6. If FAIL: build `is_good(subset)` = "checkout base + cherry-pick subset's
   commits, run gate", then call `find_culprits(remove_ids, is_good)`. Demote
   each culprit to `keep`, restore its code, re-run the gate.
7. Write `report.md`: every entry with applied/reverted + reason.
