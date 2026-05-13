"""PipelineRunner tests — verifies SSE event emission during a pipeline run."""
from __future__ import annotations

import asyncio

import pytest

from app.services.event_bus import Event, EventBus
from app.services.pipeline_runner import PipelineRunner


class _FakeFlow:
    """Minimal stub that drives the EventBus through a fixed sequence."""

    def __init__(
        self,
        *,
        run_id: str,
        bus: EventBus,
        video: str,
        output_dir: str,
        db: str,
        detector: str = "docaligner",
        config_preset: str = "balanced",
    ) -> None:
        # Emit two stage events so the collector can observe them.
        for stage in ("detect", "score"):
            bus.emit(run_id, Event(name="stage_started", stage=stage))
            bus.emit(run_id, Event(name="stage_completed", stage=stage))


class _BrokenFlow:
    """Stub that always raises."""

    def __init__(self, **kwargs):
        raise RuntimeError("flow crashed")


def test_runner_emits_stage_events_for_each_step():
    """PipelineRunner must emit run_started, stage events, and run_completed."""
    bus = EventBus()
    events_seen: list[Event] = []

    async def main() -> None:
        # Collect 6 events: run_started + 2×(stage_started+stage_completed) + run_completed
        async def collector():
            q = bus.subscribe("run_1")
            for _ in range(6):
                events_seen.append(await q.get())

        runner = PipelineRunner(bus=bus, flow_cls=_FakeFlow)

        await asyncio.wait_for(
            asyncio.gather(
                collector(),
                runner.run_async(
                    "run_1",
                    video="x.mov",
                    output_dir="/tmp/o",
                    db="/tmp/db.sqlite",
                ),
            ),
            timeout=10,
        )

    asyncio.run(main())

    names = [e.name for e in events_seen]
    assert "run_started" in names
    assert "stage_started" in names
    assert "stage_completed" in names
    assert "run_completed" in names


def test_runner_emits_run_failed_on_exception():
    """If the flow raises, the runner emits run_failed before re-raising."""
    bus = EventBus()
    events_seen: list[Event] = []

    async def main() -> None:
        async def collector():
            q = bus.subscribe("run_fail")
            # Collect up to 2 events (run_started + run_failed)
            for _ in range(2):
                events_seen.append(await q.get())

        runner = PipelineRunner(bus=bus, flow_cls=_BrokenFlow)

        with pytest.raises(RuntimeError, match="flow crashed"):
            await asyncio.wait_for(
                asyncio.gather(
                    collector(),
                    runner.run_async(
                        "run_fail",
                        video="x.mov",
                        output_dir="/tmp/o",
                        db="/tmp/db.sqlite",
                    ),
                    return_exceptions=False,
                ),
                timeout=10,
            )

    asyncio.run(main())

    names = [e.name for e in events_seen]
    assert "run_failed" in names
