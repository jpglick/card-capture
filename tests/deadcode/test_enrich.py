from scripts.deadcode.enrich import adjusted_confidence, classify, dynamic_refs
from scripts.deadcode.models import Finding


def test_app_findings_are_excluded(tmp_path):
    f = Finding(path="app/main.py", lineno=1, name="foo", kind="function", confidence=90)
    assert classify(f, dynamic_hits=0).decision == "keep"
    assert classify(f, dynamic_hits=0).reason.startswith("excluded: app/")


def test_high_confidence_no_dynamic_refs_is_remove_tier1():
    f = Finding(path="src/card_capture/x.py", lineno=1, name="dead_fn", kind="function", confidence=95)
    e = classify(f, dynamic_hits=0)
    assert e.decision == "remove"
    assert e.tier == 1


def test_dynamic_reference_demotes_to_investigate():
    f = Finding(path="src/card_capture/x.py", lineno=1, name="maybe", kind="function", confidence=95)
    e = classify(f, dynamic_hits=2)
    assert e.decision == "investigate"
    assert e.adjusted_confidence < 95


def test_medium_confidence_is_investigate_tier2():
    f = Finding(path="src/card_capture/x.py", lineno=1, name="m", kind="function", confidence=70)
    e = classify(f, dynamic_hits=0)
    assert e.decision == "investigate"
    assert e.tier == 2
