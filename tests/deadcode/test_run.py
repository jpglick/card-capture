from scripts.deadcode.run import find_culprits


def test_no_culprits_when_full_set_is_good():
    changes = ["a", "b", "c", "d"]
    # every subset is good
    culprits = find_culprits(changes, is_good=lambda s: True)
    assert culprits == []


def test_single_culprit_is_isolated():
    changes = ["a", "b", "c", "d"]
    # bad iff "c" is present
    culprits = find_culprits(changes, is_good=lambda s: "c" not in s)
    assert culprits == ["c"]


def test_multiple_independent_culprits_all_found():
    changes = ["a", "b", "c", "d", "e"]
    # bad iff "b" OR "d" present
    bad = {"b", "d"}
    culprits = find_culprits(changes, is_good=lambda s: not (bad & set(s)))
    assert set(culprits) == {"b", "d"}


def test_predicate_call_count_is_logarithmic_for_single_culprit():
    changes = [str(i) for i in range(64)]
    calls = {"n": 0}

    def is_good(s):
        calls["n"] += 1
        return "37" not in s

    culprits = find_culprits(changes, is_good=is_good)
    assert culprits == ["37"]
    assert calls["n"] <= 20  # ~log2(64) recursion, not 64 linear probes
