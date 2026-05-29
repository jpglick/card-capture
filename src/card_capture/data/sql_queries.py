"""Central SQL text constants owned by the data layer."""
from __future__ import annotations

# Video service
VIDEO_LIST = "SELECT * FROM videos ORDER BY created_at DESC LIMIT ?"
VIDEO_GET = "SELECT * FROM videos WHERE id = ?"
VIDEO_INSERT = (
    "INSERT INTO videos(source_path, file_hash, duration_ms, width, height) "
    "VALUES (?, ?, ?, ?, ?)"
)
VIDEO_UPDATE_STATUS = "UPDATE videos SET status = ? WHERE id = ?"
VIDEO_DELETE = "DELETE FROM videos WHERE id = ?"
VIDEO_DELETE_EVENTS = "DELETE FROM pipeline_events WHERE video_id = ?"
VIDEO_DELETE_CARDS = "DELETE FROM card_instances WHERE video_id = ?"

# Runs service
RUNS_LIST_BASE = """
SELECT pr.run_id, pr.video_id, pr.status, pr.cards_extracted,
       pr.started_at as created_at, pr.finished_at,
       v.source_path
FROM pipeline_runs pr
LEFT JOIN videos v ON v.id = pr.video_id
{where}
ORDER BY pr.started_at DESC
"""
RUN_DETAILS = """
SELECT pr.run_id, pr.video_id, pr.status, pr.cards_extracted,
       pr.started_at, pr.finished_at,
       v.source_path, v.duration_ms as video_duration_ms
FROM pipeline_runs pr
LEFT JOIN videos v ON v.id = pr.video_id
WHERE pr.run_id = ?
"""
RUN_EVENTS = (
    "SELECT event_type, data_json, created_at FROM pipeline_events "
    "WHERE run_id = ? ORDER BY created_at ASC"
)
RUN_LOGS = (
    "SELECT line FROM pipeline_run_logs WHERE run_id = ? ORDER BY id ASC"
)
RUN_RESOURCE_RANGE = "SELECT started_at, finished_at FROM pipeline_runs WHERE run_id = ?"
RUN_RESOURCE_SAMPLES = (
    "SELECT elapsed_s, cpu_pct, mem_used_mb, mem_pct, gpu_pct, vram_used_mb "
    "FROM run_resource_samples WHERE run_id = ? ORDER BY elapsed_s"
)
RUN_STAGE_EVENTS = (
    "SELECT stage_id, event_type, created_at FROM pipeline_events "
    "WHERE run_id = ? AND event_type LIKE 'stage_%' ORDER BY created_at"
)

# Cards service
CARDS_LIST_BASE = "SELECT id, track_id, video_id, run_id, angle, fused_image_path, created_at FROM card_instances"
CARDS_COUNT_BASE = "SELECT COUNT(*) FROM card_instances"
CARDS_CANONICAL_VIEW = (
    "SELECT confidence, rectified_path FROM card_views WHERE card_instance_id = ? AND is_canonical = 1"
)
CARDS_VIDEO_SOURCE = "SELECT source_path FROM videos WHERE id = ?"
CARD_GET_ONE = (
    "SELECT id, track_id, video_id, run_id, angle, fused_image_path, created_at "
    "FROM card_instances WHERE id = ? AND hidden = 0"
)
CARD_SOURCE_FRAMES = "SELECT frame_index FROM card_views WHERE card_instance_id = ?"

# Labeling service
TRUTH_GET = "SELECT payload_json FROM truth_files WHERE video_id = ?"
LABELS_COUNT = "SELECT COUNT(*) FROM fb_labels"
CLUSTERS_LIST_BASE = (
    "SELECT cluster_id, predicted_member_ids_json, confirmed_member_ids_json, status, updated_at FROM dedup_clusters"
)

# Training service
TRAINING_FB_DIST = "SELECT side, COUNT(*) as count FROM fb_labels GROUP BY side"
TRAINING_FB_LAST = "SELECT MAX(created_at) FROM fb_labels"
TRAINING_PENDING_PRESENCE = "SELECT COUNT(*) FROM presence_samples WHERE label IS NULL"
TRAINING_PENDING_FB = (
    "SELECT COUNT(*) FROM card_instances ci "
    "WHERE NOT EXISTS (SELECT 1 FROM fb_labels fl WHERE fl.instance_id = ci.track_id)"
)
TRAINING_PENDING_CORNERS = "SELECT COUNT(*) FROM corner_samples WHERE label IS NULL"
TRAINING_LATEST_METRICS = (
    "SELECT eval_metrics_json FROM model_versions "
    "WHERE model_name=? ORDER BY created_at DESC LIMIT 1"
)
TRAINING_HISTORY = (
    "SELECT model_name, eval_metrics_json, created_at FROM model_versions "
    "ORDER BY created_at ASC"
)
TRAINING_RECENT_RUNS = (
    "SELECT run_id, video_id, cards_extracted FROM pipeline_runs "
    "WHERE status='completed' ORDER BY started_at DESC LIMIT ?"
)
TRAINING_VIDEOS_ALL = "SELECT id, source_path FROM videos"

# Pipeline runner
PIPELINE_RUN_INSERT_START = (
    "INSERT OR IGNORE INTO pipeline_runs (run_id, video_id, status) VALUES (?, ?, 'running')"
)
PIPELINE_RUN_UPDATE_HOST_INFO = "UPDATE pipeline_runs SET host_info_json = ? WHERE run_id = ?"
PIPELINE_RUN_COUNT_CARDS = "SELECT COUNT(*) FROM card_instances WHERE run_id = ?"
PIPELINE_RUN_FINISH = (
    "UPDATE pipeline_runs SET status=?, cards_extracted=?, finished_at=datetime('now') WHERE run_id=?"
)
PIPELINE_RUN_LOG_INSERT = "INSERT INTO pipeline_run_logs (run_id, line) VALUES (?, ?)"

# Result importer
RESULT_RUN_UPDATE_TELEMETRY = "UPDATE pipeline_runs SET detect_telemetry_json=? WHERE run_id=?"
RESULT_RUN_UPDATE_HOST_INFO = "UPDATE pipeline_runs SET host_info_json=? WHERE run_id=?"
RESULT_RESOURCE_SAMPLES_DELETE = "DELETE FROM run_resource_samples WHERE run_id=?"
RESULT_LOGS_COUNT = "SELECT COUNT(*) FROM pipeline_run_logs WHERE run_id=?"
RESULT_LOGS_INSERT = "INSERT INTO pipeline_run_logs (run_id, line) VALUES (?, ?)"
RESULT_LOGS_DELETE = "DELETE FROM pipeline_run_logs WHERE run_id=?"
RESULT_CARD_INSTANCES_FOR_RUN = "SELECT * FROM card_instances WHERE run_id=? ORDER BY id"
RESULT_CARD_INSTANCE_ID_BY_RUN_TRACK = "SELECT id FROM card_instances WHERE run_id=? AND track_id=?"
RESULT_EVENTS_DELETE = "DELETE FROM pipeline_events WHERE run_id=?"
RESULT_EVENTS_FOR_RUN = "SELECT * FROM pipeline_events WHERE run_id=? ORDER BY id"
RESULT_RESOURCE_SAMPLES_FOR_RUN = "SELECT * FROM run_resource_samples WHERE run_id=? ORDER BY elapsed_s"
RESULT_LOGS_FOR_RUN = "SELECT line FROM pipeline_run_logs WHERE run_id=? ORDER BY id"
RESULT_EVENT_EXISTS = "SELECT 1 FROM pipeline_events WHERE run_id=? AND event_type=? LIMIT 1"
RESULT_EVENT_INSERT = (
    "INSERT INTO pipeline_events "
    "(video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json) "
    "VALUES (?, ?, ?, 0, 0, ?, ?)"
)
RESULT_RUN_VIDEO_ID = "SELECT video_id FROM pipeline_runs WHERE run_id=?"
RESULT_COUNT_CARDS = "SELECT COUNT(*) FROM card_instances WHERE run_id=?"
RESULT_TABLE_EXISTS = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"


def result_card_instances_insert(columns_sql: str, placeholders_sql: str) -> str:
    return f"INSERT OR IGNORE INTO card_instances ({columns_sql}) VALUES ({placeholders_sql})"


def result_card_instances_update(assignments_sql: str) -> str:
    return f"UPDATE card_instances SET {assignments_sql} WHERE run_id=? AND track_id=?"


def result_saved_cards_delete_for_card_instance_ids(placeholders_sql: str) -> str:
    return (
        "DELETE FROM saved_cards WHERE detection_id IN "
        f"(SELECT id FROM card_views WHERE card_instance_id IN ({placeholders_sql}))"
    )


def result_card_views_delete_for_card_instance_ids(placeholders_sql: str) -> str:
    return f"DELETE FROM card_views WHERE card_instance_id IN ({placeholders_sql})"


def result_card_views_for_card_instance_ids(placeholders_sql: str) -> str:
    return f"SELECT * FROM card_views WHERE card_instance_id IN ({placeholders_sql}) ORDER BY id"


def result_card_views_insert(columns_sql: str, placeholders_sql: str) -> str:
    return f"INSERT INTO card_views ({columns_sql}) VALUES ({placeholders_sql})"


def result_pipeline_events_insert(columns_sql: str, placeholders_sql: str) -> str:
    return f"INSERT INTO pipeline_events ({columns_sql}) VALUES ({placeholders_sql})"


def result_resource_samples_insert(columns_sql: str, placeholders_sql: str) -> str:
    return f"INSERT INTO run_resource_samples ({columns_sql}) VALUES ({placeholders_sql})"


def result_pragma_table_info(table: str) -> str:
    return f"PRAGMA table_info({table})"
