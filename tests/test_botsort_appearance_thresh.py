from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from card_capture.pipeline.stages.dedup import SAME_CARD_EMB_THRESHOLD


class _FakeBoTSORT:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.active_tracks = []
        _FakeBoTSORT.calls.append(kwargs)

    def update(self, det_input, img=None, embs=None):
        return []


def test_botsort_uses_calibrated_appearance_thresh():
    _FakeBoTSORT.calls = []
    with patch.dict("sys.modules", {"supervision": SimpleNamespace(Detections=object)}):
        with patch("card_capture.tracking.botsort_adapter._import_botsort", return_value=_FakeBoTSORT):
            from card_capture.tracking.botsort_adapter import BoTSORTAdapter

            adapter = BoTSORTAdapter(min_track_length=1)
            adapter.reset()

    assert len(_FakeBoTSORT.calls) >= 2
    for kwargs in _FakeBoTSORT.calls[:2]:
        assert kwargs["appearance_thresh"] == SAME_CARD_EMB_THRESHOLD
        assert kwargs["with_reid"] is True

