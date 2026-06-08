from __future__ import annotations

from types import SimpleNamespace

from card_capture.shared.stage_metrics import emit_stage_metrics


class _EventsRepo:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_stage_metrics(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_emit_stage_metrics_records_event() -> None:
    events = _EventsRepo()
    state = {
        "request": SimpleNamespace(run_id="r1"),
        "video_id": 7,
        "repos": {"events": events},
    }
    emit_stage_metrics(state, stage="detect", metrics={"detections": 11})
    assert events.calls == [
        {
            "run_id": "r1",
            "video_id": 7,
            "stage": "detect",
            "metrics": {"detections": 11},
        }
    ]


def test_emit_stage_metrics_is_noop_without_events_repo() -> None:
    state = {"request": SimpleNamespace(run_id="r1"), "repos": {}}
    emit_stage_metrics(state, stage="detect", metrics={"detections": 1})


def test_emit_stage_metrics_buffers_into_state_without_repos() -> None:
    # The telemetry bridge buffer must populate even with no events repo wired.
    state: dict = {}
    emit_stage_metrics(state, stage="detect", metrics={"detections": 11})
    assert state["stage_metrics"]["detect"] == {"detections": 11}


def test_emit_stage_metrics_buffer_merges_across_calls() -> None:
    state: dict = {}
    emit_stage_metrics(state, stage="dedup", metrics={"dedup_groups": 3})
    emit_stage_metrics(state, stage="dedup", metrics={"final_cards": 2})
    assert state["stage_metrics"]["dedup"] == {"dedup_groups": 3, "final_cards": 2}

