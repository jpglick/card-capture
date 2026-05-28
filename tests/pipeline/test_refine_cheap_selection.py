# tests/pipeline/test_refine_cheap_selection.py
"""When triage_metrics carry cheap scores, refine should rank candidates by
combined flat+clear and cap the shortlist at 5."""
from pipeline.steps.refine import _select_cheap_candidates


def _cand(did, flatness, clarity):
    return {"detection_id": did, "frame_index": did, "corners": [(0, 0)] * 4,
            "score_total": 0.5, "confidence": 0.9}


def test_selects_top5_by_combined_flat_and_clear():
    cands = [_cand(i, flatness=i / 10.0, clarity=float(i)) for i in range(10)]
    lookup = {i: {"triage_metrics": {"flatness": i / 10.0, "clarity": float(i)}}
              for i in range(10)}
    picked = _select_cheap_candidates(cands, lookup, top_k=5)
    ids = [c["detection_id"] for c in picked]
    assert ids == [9, 8, 7, 6, 5]  # highest flat+clear first


def test_returns_all_when_fewer_than_k():
    cands = [_cand(i, i / 10.0, float(i)) for i in range(3)]
    lookup = {i: {"triage_metrics": {"flatness": i / 10.0, "clarity": float(i)}}
              for i in range(3)}
    assert len(_select_cheap_candidates(cands, lookup, top_k=5)) == 3
