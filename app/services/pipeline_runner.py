from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional, Type

from card_capture.data.connection import open_connection
from card_capture.data.sql_queries import (
    PIPELINE_RUN_COUNT_CARDS,
    PIPELINE_RUN_FINISH,
    PIPELINE_RUN_INSERT_START,
    PIPELINE_RUN_LOG_INSERT,
    PIPELINE_RUN_UPDATE_HOST_INFO,
)

_REPO_ROOT = str(Path(__file__).parent.parent.parent)
logger = logging.getLogger(__name__)

from app.services.event_bus import Event, EventBus
from app.services import _event_bus_registry
from app.services.pipeline_telemetry import EventBusTelemetry


class PipelineRunner:
    def __init__(self, bus: EventBus, flow_cls: Optional[Type] = None, db_path: Optional[Path] = None) -> None:
        self.bus = bus
        self.flow_cls = flow_cls
        self.db_path = db_path  # for updating video status on failure

    async def run_async(
        self,
        run_id: str,
        *,
        video_id: int,
        video: str,
        output_dir: str,
        db: str,
        detector: str = "docaligner",
        config_preset: str = "balanced",
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                self._run_blocking,
                run_id,
                video_id,
                video,
                output_dir,
                db,
                detector,
                config_preset,
            )
        except Exception:
            # Already handled in _run_blocking (status updated, event emitted).
            # Swallow here so the exception doesn't propagate through Starlette's
            # ASGI background task layer and generate a spurious 500 traceback.
            pass

    def _run_blocking(
        self,
        run_id: str,
        video_id: int,
        video: str,
        output_dir: str,
        db: str,
        detector: str,
        config_preset: str,
    ) -> None:
        os.environ["EVENT_BUS_RUN_ID"] = run_id
        os.environ["EVENT_BUS_INPROC"] = "1"
        _event_bus_registry.set(run_id, self.bus)

        try:
            self.bus.emit(run_id, Event(name="run_started"))
            self._record_run_start(run_id, video_id)
            print(f"[{run_id}] pipeline starting — video={video}", flush=True)

            if self.flow_cls is not None:
                # Tests inject a fake flow_cls; keep that hook intact.
                self.flow_cls(
                    run_id=run_id, bus=self.bus, video=video,
                    output_dir=output_dir, db=db,
                    detector=detector, config_preset=config_preset,
                )
            else:
                self._run_unified_inprocess(
                    run_id=run_id,
                    video_id=video_id,
                    video=video,
                    output_dir=output_dir,
                    db=db,
                    detector=detector,
                    config_preset=config_preset,
                )

            print(f"[{run_id}] pipeline completed", flush=True)
            self.bus.emit(run_id, Event(name="run_completed"))
            self._set_video_status(video_id, "completed")
            self._record_run_finish(run_id, "completed")
            self._sample_presence_frames(run_id, video_id, video)

        except Exception as exc:
            print(f"[{run_id}] pipeline failed: {exc}", flush=True)
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            self._set_video_status(video_id, "failed")
            self._record_run_finish(run_id, "failed")
            raise
        finally:
            _event_bus_registry.clear(run_id)
            os.environ.pop("EVENT_BUS_RUN_ID", None)
            os.environ.pop("EVENT_BUS_INPROC", None)

    def _run_unified_inprocess(
        self,
        *,
        run_id: str,
        video_id: int,
        video: str,
        output_dir: str,
        db: str,
        detector: str,
        config_preset: str,
    ) -> None:
        """Drive the v5.5 in-process unified runtime.

        Replaces the deleted ``pipeline.card_capture_flow`` subprocess. Stage
        events are reported via ``EventBusTelemetry`` so the UI's SSE stream
        keeps receiving log + stage_progress events. The resource sampler
        still runs alongside for per-stage CPU/GPU samples.
        """
        from card_capture.pipeline.request import PipelineRunRequest
        from card_capture.pipeline.runtime_local import LocalPipelineRuntime

        abs_db = str(Path(db).resolve())
        abs_video = str(Path(video).resolve())
        abs_output = str((Path(_REPO_ROOT) / output_dir).resolve())
        Path(abs_output).mkdir(parents=True, exist_ok=True)

        telemetry = EventBusTelemetry(
            self.bus,
            run_id,
            log_sink=lambda line: self._persist_log(run_id, line),
        )

        # Resource sampler is keyed by stage. Wrap the telemetry so
        # stage_started flips the sampler's current_stage at the
        # same instant we tell the UI.
        sampler = None
        if self.db_path:
            from app.services.resource_sampler import ResourceSampler
            sampler = ResourceSampler(run_id, self.db_path, time.monotonic())
            sampler.start()
            _original_stage_started = telemetry.stage_started

            def _stage_started_with_sampler(stage, metadata):
                sampler.current_stage = stage
                _original_stage_started(stage, metadata)

            telemetry.stage_started = _stage_started_with_sampler  # type: ignore[method-assign]

        # Merge full PipelineConfig defaults so back-half stages read knobs
        # like novelty_floor / foil_threshold from request.config. Caller
        # overrides win (detector arg, etc.).
        from card_capture.config import load_config
        config = load_config(Path(_REPO_ROOT) / "card_capture_config.json")
        request_config = config.to_request_config()
        request_config["detector"] = detector

        runtime = LocalPipelineRuntime(telemetry=telemetry)
        request = PipelineRunRequest(
            run_id=run_id,
            input_video=f"artifact://local/{abs_video}",
            output_root=f"artifact://local/{abs_output}/",
            runtime_mode="cpu_debug",
            config=request_config,
            db_path=abs_db,
            video_id=video_id,
            config_preset=config_preset,
        )

        banner = f"running unified pipeline: video={abs_video} db={abs_db}"
        print(f"[{run_id}] {banner}", flush=True)
        self.bus.emit(run_id, Event(name="log", payload={"line": banner}))
        self._persist_log(run_id, banner)

        try:
            result = runtime.run(request)
        finally:
            if sampler is not None:
                sampler.stop()

        violations = result.manifest.contract_violations
        if violations:
            raise RuntimeError(f"Pipeline contract violations: {violations}")

    def _record_run_start(self, run_id: str, video_id: int) -> None:
        if not self.db_path:
            return
        try:
            with open_connection(self.db_path) as conn:
                conn.execute(PIPELINE_RUN_INSERT_START, (run_id, video_id))
        except Exception as exc:
            print(f"[{run_id}] could not record run start: {exc}", flush=True)
        try:
            import json as _json
            from app.services.resource_sampler import get_host_info
            host_info = get_host_info()
            with open_connection(self.db_path) as conn:
                conn.execute(PIPELINE_RUN_UPDATE_HOST_INFO, (_json.dumps(host_info), run_id))
        except Exception as exc:
            print(f"[{run_id}] could not record host info: {exc}", flush=True)

    def _record_run_finish(self, run_id: str, status: str) -> None:
        if not self.db_path:
            return
        try:
            with open_connection(self.db_path) as conn:
                cards = conn.execute(
                    PIPELINE_RUN_COUNT_CARDS, (run_id,)
                ).fetchone()[0]
                conn.execute(PIPELINE_RUN_FINISH, (status, cards, run_id))
        except Exception as exc:
            print(f"[{run_id}] could not record run finish: {exc}", flush=True)

    def _sample_presence_frames(self, run_id: str, video_id: int, video_path: str) -> None:
        if not self.db_path:
            return
        try:
            from app.services.presence_sampler import sample_presence_frames
            from pathlib import Path as _Path
            base_output = _Path(self.db_path).parent
            n = sample_presence_frames(
                video_path=_Path(video_path),
                run_id=run_id,
                video_id=video_id,
                output_dir=base_output,
                db_path=_Path(self.db_path),
            )
            print(f"[{run_id}] queued {n} presence frames for labeling", flush=True)
        except Exception as exc:
            print(f"[{run_id}] presence sampling skipped: {exc}", flush=True)

    def _persist_log(self, run_id: str, line: str) -> None:
        if not self.db_path:
            return
        try:
            with open_connection(self.db_path) as conn:
                conn.execute(
                    PIPELINE_RUN_LOG_INSERT,
                    (run_id, line),
                )
        except Exception:
            pass  # never let a DB error kill the pipeline

    def _set_video_status(self, video_id: int, status: str) -> None:
        if not self.db_path:
            return
        try:
            from card_capture.storage import Storage
            Storage(self.db_path).update_video_status(video_id, status)
        except Exception as exc:
            logger.warning("Could not update video %s status to %s: %s", video_id, status, exc)
