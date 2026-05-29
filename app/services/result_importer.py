"""Unpack cloud result tarballs and merge worker output into local SQLite."""
from __future__ import annotations

import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Optional
from card_capture.data.connection import open_connection
from card_capture.data.sql_queries import (
    RESULT_CARD_INSTANCE_ID_BY_RUN_TRACK,
    RESULT_CARD_INSTANCES_FOR_RUN,
    RESULT_COUNT_CARDS,
    RESULT_EVENT_EXISTS,
    RESULT_EVENT_INSERT,
    RESULT_EVENTS_DELETE,
    RESULT_EVENTS_FOR_RUN,
    RESULT_LOGS_COUNT,
    RESULT_LOGS_DELETE,
    RESULT_LOGS_FOR_RUN,
    RESULT_LOGS_INSERT,
    RESULT_RESOURCE_SAMPLES_DELETE,
    RESULT_RESOURCE_SAMPLES_FOR_RUN,
    RESULT_RUN_UPDATE_HOST_INFO,
    RESULT_RUN_UPDATE_TELEMETRY,
    RESULT_RUN_VIDEO_ID,
    RESULT_TABLE_EXISTS,
    result_card_instances_insert,
    result_card_instances_update,
    result_card_views_delete_for_card_instance_ids,
    result_card_views_for_card_instance_ids,
    result_card_views_insert,
    result_pipeline_events_insert,
    result_pragma_table_info,
    result_resource_samples_insert,
    result_saved_cards_delete_for_card_instance_ids,
)


class ResultImporter:
    def __init__(self, db_path: Path, output_base: Path) -> None:
        self.db_path = db_path
        self.output_base = output_base

    def import_tarball(self, tarball_path: Path, run_id: str) -> int:
        """Unpack tarball, copy crops, import card rows. Returns cards for run."""
        run_dir = self.output_base / run_id
        crops_dir = run_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            worker_db: Optional[Path] = None
            with tarfile.open(tarball_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.startswith("crops/") and not member.isdir():
                        src = tar.extractfile(member)
                        if src:
                            (crops_dir / Path(member.name).name).write_bytes(src.read())

                export_f = tar.extractfile("export.json")
                if not export_f:
                    raise ValueError("Tarball missing export.json")
                cards: list[dict] = json.loads(export_f.read())

                try:
                    db_member = tar.getmember("cards.sqlite")
                except KeyError:
                    db_member = None
                if db_member is not None and db_member.isfile():
                    src = tar.extractfile(db_member)
                    if src:
                        worker_db = Path(tmp) / "cards.sqlite"
                        worker_db.write_bytes(src.read())

            if worker_db is not None:
                self._import_worker_db(worker_db, run_id, crops_dir)
            else:
                self._import_export_cards(cards, run_id, crops_dir)

        with open_connection(self.db_path) as conn:
            return self._count_cards(conn, run_id)

    def import_handler_output(self, handler_output: dict[str, Any], run_id: str) -> None:
        """Persist RunPod handler diagnostics that are not part of export.json."""
        with open_connection(self.db_path) as conn:
            video_id = self._local_video_id(conn, run_id)
            telemetry = {
                "status": handler_output.get("status"),
                "gpu": handler_output.get("gpu"),
                "resource_stats": handler_output.get("resource_stats"),
                "timings": handler_output.get("timings"),
                "diagnostics": handler_output.get("diagnostics"),
            }
            if self._has_column(conn, "pipeline_runs", "detect_telemetry_json"):
                conn.execute(
                    RESULT_RUN_UPDATE_TELEMETRY,
                    (json.dumps(telemetry), run_id),
                )
            if self._has_column(conn, "pipeline_runs", "host_info_json"):
                host_info = {
                    "provider": "runpod",
                    "gpu": handler_output.get("gpu"),
                    "resource_stats": handler_output.get("resource_stats"),
                }
                conn.execute(
                    RESULT_RUN_UPDATE_HOST_INFO,
                    (json.dumps(host_info), run_id),
                )

            diagnostics = handler_output.get("diagnostics") or {}
            for event_type, payload in (diagnostics.get("stage_payloads") or {}).items():
                stage = event_type[len("stage_"):] if event_type.startswith("stage_") else event_type
                self._insert_event_if_missing(conn, video_id, run_id, stage, event_type, payload)

            if handler_output.get("resource_stats"):
                self._insert_event_if_missing(
                    conn,
                    video_id,
                    run_id,
                    "runpod",
                    "runpod_resource_stats",
                    handler_output["resource_stats"],
                )

            samples = handler_output.get("resource_samples") or []
            if samples and self._table_exists(conn, "run_resource_samples"):
                conn.execute(RESULT_RESOURCE_SAMPLES_DELETE, (run_id,))
                for sample in samples:
                    self._insert_resource_sample(conn, run_id, sample)

            stdout = handler_output.get("metaflow_stdout") or handler_output.get("metaflow_stdout_tail")
            if stdout and self._table_exists(conn, "pipeline_run_logs"):
                existing = conn.execute(
                    RESULT_LOGS_COUNT, (run_id,)
                ).fetchone()[0]
                if existing == 0:
                    for line in str(stdout).splitlines():
                        conn.execute(
                            RESULT_LOGS_INSERT,
                            (run_id, line),
                        )

    def _import_export_cards(self, cards: list[dict], run_id: str, crops_dir: Path) -> None:
        with open_connection(self.db_path) as conn:
            video_id = self._local_video_id(conn, run_id)
            for card in cards:
                self._upsert_card_instance(
                    conn,
                    run_id=run_id,
                    video_id=video_id,
                    track_id=str(card.get("track_id", "")),
                    session_id=card.get("session_id"),
                    visual_hash=card.get("visual_hash"),
                    reid_embedding=None,
                    angle=card.get("angle") or card.get("side") or "Front",
                    fused_image_path=self._local_crop_path(crops_dir, card.get("fused_image_path")),
                )

    def _import_worker_db(self, worker_db: Path, run_id: str, crops_dir: Path) -> None:
        with open_connection(self.db_path) as local, open_connection(worker_db) as worker:
            local.row_factory = sqlite3.Row
            worker.row_factory = sqlite3.Row
            video_id = self._local_video_id(local, run_id)

            id_map: dict[int, int] = {}
            if self._table_exists(worker, "card_instances"):
                rows = worker.execute(
                    RESULT_CARD_INSTANCES_FOR_RUN, (run_id,)
                ).fetchall()
                for row in rows:
                    local_id = self._upsert_card_instance(
                        local,
                        run_id=run_id,
                        video_id=video_id,
                        track_id=row["track_id"],
                        session_id=row["session_id"] if "session_id" in row.keys() else None,
                        visual_hash=row["visual_hash"] if "visual_hash" in row.keys() else None,
                        reid_embedding=row["reid_embedding"] if "reid_embedding" in row.keys() else None,
                        angle=row["angle"] if "angle" in row.keys() else None,
                        fused_image_path=self._local_crop_path(
                            crops_dir, row["fused_image_path"] if "fused_image_path" in row.keys() else None
                        ),
                    )
                    id_map[int(row["id"])] = local_id

            if id_map and self._table_exists(worker, "card_views") and self._table_exists(local, "card_views"):
                self._replace_card_views(local, worker, id_map, crops_dir)

            self._replace_events(local, worker, run_id, video_id)
            self._replace_resource_samples(local, worker, run_id)
            self._replace_logs(local, worker, run_id)

    def _upsert_card_instance(
        self,
        conn: Any,
        *,
        run_id: str,
        video_id: int,
        track_id: str,
        session_id: Any,
        visual_hash: Optional[str],
        reid_embedding: Optional[bytes],
        angle: Optional[str],
        fused_image_path: Optional[str],
    ) -> int:
        cols = self._columns(conn, "card_instances")
        values = {
            "video_id": video_id,
            "run_id": run_id,
            "track_id": track_id,
            "session_id": session_id,
            "visual_hash": visual_hash,
            "reid_embedding": reid_embedding,
            "angle": angle,
            "fused_image_path": fused_image_path,
        }
        insert_cols = [c for c in values if c in cols]
        placeholders = ", ".join("?" for _ in insert_cols)
        conn.execute(
            result_card_instances_insert(", ".join(insert_cols), placeholders),
            [values[c] for c in insert_cols],
        )
        update_cols = [c for c in insert_cols if c not in {"video_id", "run_id", "track_id"}]
        if update_cols:
            assignments = ", ".join(f"{c}=?" for c in update_cols)
            conn.execute(
                result_card_instances_update(assignments),
                [values[c] for c in update_cols] + [run_id, track_id],
            )
        row = conn.execute(
            RESULT_CARD_INSTANCE_ID_BY_RUN_TRACK, (run_id, track_id)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Could not import card instance {track_id!r} for run {run_id!r}")
        return int(row[0])

    def _replace_card_views(
        self,
        local: Any,
        worker: Any,
        id_map: dict[int, int],
        crops_dir: Path,
    ) -> None:
        local_ids = list(id_map.values())
        placeholders = ", ".join("?" for _ in local_ids)
        if self._table_exists(local, "saved_cards"):
            local.execute(
                result_saved_cards_delete_for_card_instance_ids(placeholders),
                local_ids,
            )
        local.execute(result_card_views_delete_for_card_instance_ids(placeholders), local_ids)

        local_cols = [c for c in self._columns(local, "card_views") if c != "id"]
        worker_ids = list(id_map.keys())
        worker_placeholders = ", ".join("?" for _ in worker_ids)
        rows = worker.execute(
            result_card_views_for_card_instance_ids(worker_placeholders),
            worker_ids,
        ).fetchall()
        for row in rows:
            values = dict(row)
            values["card_instance_id"] = id_map[int(row["card_instance_id"])]
            if "rectified_path" in values:
                values["rectified_path"] = self._local_crop_path(crops_dir, values["rectified_path"])
            insert_cols = [c for c in local_cols if c in values]
            local.execute(
                result_card_views_insert(", ".join(insert_cols), ", ".join("?" for _ in insert_cols)),
                [values[c] for c in insert_cols],
            )

    def _replace_events(
        self, local: Any, worker: Any, run_id: str, video_id: int
    ) -> None:
        if not (self._table_exists(local, "pipeline_events") and self._table_exists(worker, "pipeline_events")):
            return
        local.execute(RESULT_EVENTS_DELETE, (run_id,))
        local_cols = [c for c in self._columns(local, "pipeline_events") if c != "id"]
        for row in worker.execute(RESULT_EVENTS_FOR_RUN, (run_id,)):
            values = dict(row)
            values["video_id"] = video_id
            values["run_id"] = run_id
            insert_cols = [c for c in local_cols if c in values]
            local.execute(
                result_pipeline_events_insert(", ".join(insert_cols), ", ".join("?" for _ in insert_cols)),
                [values[c] for c in insert_cols],
            )

    def _replace_resource_samples(
        self, local: Any, worker: Any, run_id: str
    ) -> None:
        if not (self._table_exists(local, "run_resource_samples") and self._table_exists(worker, "run_resource_samples")):
            return
        local.execute(RESULT_RESOURCE_SAMPLES_DELETE, (run_id,))
        local_cols = [c for c in self._columns(local, "run_resource_samples") if c != "id"]
        for row in worker.execute(RESULT_RESOURCE_SAMPLES_FOR_RUN, (run_id,)):
            values = dict(row)
            insert_cols = [c for c in local_cols if c in values]
            local.execute(
                result_resource_samples_insert(", ".join(insert_cols), ", ".join("?" for _ in insert_cols)),
                [values[c] for c in insert_cols],
            )

    def _replace_logs(self, local: Any, worker: Any, run_id: str) -> None:
        if not (self._table_exists(local, "pipeline_run_logs") and self._table_exists(worker, "pipeline_run_logs")):
            return
        local.execute(RESULT_LOGS_DELETE, (run_id,))
        for row in worker.execute(RESULT_LOGS_FOR_RUN, (run_id,)):
            local.execute(RESULT_LOGS_INSERT, (run_id, row["line"]))

    def _insert_event_if_missing(
        self,
        conn: Any,
        video_id: int,
        run_id: str,
        stage_id: str,
        event_type: str,
        payload: Any,
    ) -> None:
        if not self._table_exists(conn, "pipeline_events"):
            return
        exists = conn.execute(
            RESULT_EVENT_EXISTS,
            (run_id, event_type),
        ).fetchone()
        if exists:
            return
        conn.execute(RESULT_EVENT_INSERT, (video_id, run_id, stage_id, event_type, json.dumps(payload)))

    def _insert_resource_sample(self, conn: Any, run_id: str, sample: dict[str, Any]) -> None:
        cols = self._columns(conn, "run_resource_samples")
        values = {
            "run_id": run_id,
            "elapsed_s": sample.get("elapsed_s", 0),
            "cpu_pct": sample.get("cpu_pct"),
            "mem_used_mb": sample.get("mem_used_mb", sample.get("ram_used_mb")),
            "mem_pct": sample.get("mem_pct", sample.get("ram_pct")),
            "gpu_pct": sample.get("gpu_pct"),
            "vram_used_mb": sample.get("vram_used_mb"),
            "decoder_pct": sample.get("decoder_pct"),
            "encoder_pct": sample.get("encoder_pct"),
            "mem_io_pct": sample.get("mem_io_pct"),
            "stage": sample.get("stage", "unknown"),
        }
        insert_cols = [c for c in values if c in cols]
        conn.execute(
            result_resource_samples_insert(", ".join(insert_cols), ", ".join("?" for _ in insert_cols)),
            [values[c] for c in insert_cols],
        )

    def _local_video_id(self, conn: sqlite3.Connection, run_id: str) -> int:
        if self._table_exists(conn, "pipeline_runs") and self._has_column(conn, "pipeline_runs", "video_id"):
            row = conn.execute(RESULT_RUN_VIDEO_ID, (run_id,)).fetchone()
            if row and row[0] is not None:
                return int(row[0])
        return 0

    def _count_cards(self, conn: sqlite3.Connection, run_id: str) -> int:
        if not self._table_exists(conn, "card_instances"):
            return 0
        return int(conn.execute(RESULT_COUNT_CARDS, (run_id,)).fetchone()[0])

    def _local_crop_path(self, crops_dir: Path, source_path: Any) -> Optional[str]:
        if not source_path:
            return None
        return str(crops_dir / Path(str(source_path)).name)

    def _table_exists(self, conn: sqlite3.Connection, table: str) -> bool:
        return conn.execute(
            RESULT_TABLE_EXISTS, (table,)
        ).fetchone() is not None

    def _columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(result_pragma_table_info(table)).fetchall()}

    def _has_column(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        return column in self._columns(conn, table) if self._table_exists(conn, table) else False
