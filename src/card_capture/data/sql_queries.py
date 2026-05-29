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
LABELING_PRAGMA_CARD_INSTANCES = "PRAGMA table_info(card_instances)"
LABELING_FB_CANDIDATE_BY_INSTANCE = """
SELECT
    ci.instance_id AS instance_id,
    cv.frame_index,
    cv.rectified_path AS canonical_url,
    v.source_path AS video_id,
    ci.run_id
FROM card_views cv
JOIN card_instances ci ON ci.id = cv.card_instance_id
JOIN videos v ON v.id = ci.video_id
LEFT JOIN fb_labels fl ON fl.instance_id = ci.instance_id
WHERE cv.is_canonical = 1
  AND fl.label_id IS NULL
ORDER BY cv.confidence DESC
LIMIT 1
"""
LABELING_FB_CANDIDATE_BY_TRACK = """
SELECT
    ci.track_id AS instance_id,
    cv.frame_index,
    cv.rectified_path AS canonical_url,
    v.source_path AS video_id,
    ci.run_id
FROM card_views cv
JOIN card_instances ci ON ci.id = cv.card_instance_id
JOIN videos v ON v.id = ci.video_id
LEFT JOIN fb_labels fl ON fl.instance_id = ci.track_id
WHERE cv.is_canonical = 1
  AND fl.label_id IS NULL
ORDER BY cv.confidence DESC
LIMIT 1
"""
LABELING_FB_PENDING_BY_INSTANCE = """
SELECT COUNT(DISTINCT ci.instance_id)
FROM card_views cv
JOIN card_instances ci ON ci.id = cv.card_instance_id
LEFT JOIN fb_labels fl ON fl.instance_id = ci.instance_id
WHERE cv.is_canonical = 1 AND fl.label_id IS NULL
"""
LABELING_FB_PENDING_BY_TRACK = """
SELECT COUNT(DISTINCT ci.track_id)
FROM card_views cv
JOIN card_instances ci ON ci.id = cv.card_instance_id
LEFT JOIN fb_labels fl ON fl.instance_id = ci.track_id
WHERE cv.is_canonical = 1 AND fl.label_id IS NULL
"""
LABELING_CLUSTERS_ORDER = " ORDER BY updated_at DESC"

REVIEW_CARD_JOIN_BY_DETECTION = """
SELECT ci.id as instance_id, ci.fused_image_path, ci.angle, ci.session_id
FROM card_views cv
JOIN card_instances ci ON cv.card_instance_id = ci.id
WHERE cv.id = ?
"""
REVIEW_CANONICAL_VIEWS = """
SELECT id, rectified_path
FROM card_views
WHERE card_instance_id = ? AND is_canonical = 1 AND rectified_path IS NOT NULL
ORDER BY id ASC
"""
REVIEW_TIMELINE_EVENTS_BASE = "SELECT frame_index, timestamp_ms, event_type, data_json FROM pipeline_events"
REVIEW_TIMELINE_EVENTS_ORDER = " ORDER BY timestamp_ms ASC"
REVIEW_TIMELINE_INSTANCES_BASE = """
SELECT ci.id as instance_id, ci.session_id, ci.angle, ci.is_duplicate_of, ci.video_id, ci.fused_image_path,
       MIN(cv.timestamp_ms) as start_time, MAX(cv.timestamp_ms) as end_time,
       COUNT(cv.id) as detection_count,
       MIN(cv.id) as first_view_id
FROM card_instances ci
LEFT JOIN card_views cv ON cv.card_instance_id = ci.id
"""
REVIEW_TIMELINE_INSTANCES_ORDER = " GROUP BY ci.id ORDER BY start_time ASC"
REVIEW_VIDEO_BY_ID = "SELECT id, source_path FROM videos WHERE id = ?"
REVIEW_LABEL_INSTANCES_BY_VIDEO = """
SELECT ci.id AS instance_id, ci.angle, ci.session_id, ci.is_duplicate_of,
       MIN(cv.timestamp_ms) AS start_time, MAX(cv.timestamp_ms) AS end_time,
       ci.fused_image_path
FROM card_instances ci
LEFT JOIN card_views cv ON cv.card_instance_id = ci.id
WHERE ci.video_id = ?
GROUP BY ci.id
ORDER BY start_time ASC
"""
REVIEW_VIDEO_SOURCE_BY_ID = "SELECT source_path FROM videos WHERE id = ?"
REVIEW_VIDEO_COUNT = "SELECT COUNT(*) FROM videos"
REVIEW_VIDEOS_LIST = "SELECT id, source_path FROM videos ORDER BY id"
REVIEW_FUSED_IMAGE_BY_INSTANCE = "SELECT fused_image_path FROM card_instances WHERE id = ?"
REVIEW_CARD_VIEW_BY_ID = """
SELECT cv.rectified_path, cv.card_instance_id, ci.fused_image_path
FROM card_views cv
JOIN card_instances ci ON ci.id = cv.card_instance_id
WHERE cv.id = ?
"""

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


# Storage
STORAGE_INIT_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS card_instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    run_id TEXT,
    track_id TEXT NOT NULL,
    session_id TEXT,
    visual_hash TEXT,
    reid_embedding BLOB,
    is_duplicate_of INTEGER REFERENCES card_instances(id),
    angle TEXT,
    fused_image_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, track_id)
);

CREATE TABLE IF NOT EXISTS card_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_instance_id INTEGER NOT NULL REFERENCES card_instances(id),
    frame_index INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    corners_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    rectified_path TEXT,
    quality_score_json TEXT,
    is_canonical INTEGER NOT NULL DEFAULT 0,
    glare_x REAL,
    glare_y REAL,
    sharpness REAL,
    glare_mask_b64 TEXT,
    laplacian_heatmap_b64 TEXT,
    initial_confidence REAL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evidence_frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_view_id INTEGER NOT NULL REFERENCES card_views(id),
    source_frame_path TEXT NOT NULL,
    frame_width INTEGER NOT NULL,
    frame_height INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS saved_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    image_path TEXT NOT NULL,
    final_score REAL NOT NULL,
    review_state TEXT NOT NULL DEFAULT 'pending',
    source_path TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    score_components_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_card_id INTEGER NOT NULL REFERENCES saved_cards(id),
    decision TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    frame_index INTEGER NOT NULL,
    t_ingest REAL NOT NULL,
    t_detect REAL NOT NULL,
    t_refine REAL NOT NULL,
    t_io REAL NOT NULL,
    queue_wait REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS track_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    track_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    polygon_area REAL NOT NULL,
    aspect_ratio REAL NOT NULL,
    centroid_x REAL NOT NULL,
    centroid_y REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    run_id TEXT,
    stage_id TEXT,
    frame_index INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    data_json TEXT,
    artifact_ref TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

STORAGE_VIDEO_INSERT = (
    "INSERT INTO videos (source_path, file_hash, duration_ms, width, height, status) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)
STORAGE_VIDEO_ID_BY_SOURCE = "SELECT id FROM videos WHERE source_path = ? OR source_path = ?"
STORAGE_VIDEO_INSERT_PROCESSING = (
    "INSERT INTO videos (source_path, file_hash, duration_ms, width, height, status) "
    "VALUES (?, ?, ?, ?, ?, 'processing')"
)
STORAGE_VIDEO_UPDATE_STATUS = "UPDATE videos SET status = ? WHERE id = ?"
STORAGE_PERFORMANCE_LOG_INSERT = (
    "INSERT INTO performance_logs (video_id, frame_index, t_ingest, t_detect, t_refine, t_io, queue_wait) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)
STORAGE_TRACK_TELEMETRY_INSERT = (
    "INSERT INTO track_telemetry (video_id, track_id, frame_index, polygon_area, aspect_ratio, centroid_x, centroid_y) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)
STORAGE_PIPELINE_EVENT_INSERT = (
    "INSERT INTO pipeline_events (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json, artifact_ref) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)
STORAGE_CARD_INSTANCE_INSERT = (
    "INSERT INTO card_instances (video_id, run_id, track_id, angle, session_id, reid_embedding) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)
STORAGE_CARD_INSTANCE_DEDUP_WITH_EMBED = (
    "UPDATE card_instances SET visual_hash = ?, is_duplicate_of = ?, reid_embedding = ?, updated_at = CURRENT_TIMESTAMP "
    "WHERE id = ?"
)
STORAGE_CARD_INSTANCE_DEDUP_NO_EMBED = (
    "UPDATE card_instances SET visual_hash = ?, is_duplicate_of = ?, updated_at = CURRENT_TIMESTAMP "
    "WHERE id = ?"
)
STORAGE_CARD_INSTANCE_FUSION_UPDATE = (
    "UPDATE card_instances SET fused_image_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
)
STORAGE_CARD_INSTANCE_CANONICALS = (
    "SELECT id, visual_hash FROM card_instances "
    "WHERE visual_hash IS NOT NULL AND is_duplicate_of IS NULL"
)
STORAGE_CARD_VIEW_INSERT = """
INSERT INTO card_views (
    card_instance_id,
    frame_index,
    timestamp_ms,
    corners_json,
    confidence,
    rectified_path,
    quality_score_json,
    is_canonical,
    glare_x,
    glare_y,
    sharpness,
    glare_mask_b64,
    laplacian_heatmap_b64,
    initial_confidence,
    metadata_json
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
STORAGE_EVIDENCE_FRAME_INSERT = (
    "INSERT INTO evidence_frames (card_view_id, source_frame_path, frame_width, frame_height, metrics_json) "
    "VALUES (?, ?, ?, ?, ?)"
)
STORAGE_CARD_INSTANCES_BY_VIDEO = """
SELECT
    id,
    video_id,
    track_id,
    visual_hash,
    is_duplicate_of,
    angle,
    fused_image_path,
    created_at,
    updated_at
FROM card_instances
WHERE video_id = ?
ORDER BY id ASC
"""
STORAGE_SAVED_CARD_SOURCE = """
SELECT
    card_views.card_instance_id,
    card_views.timestamp_ms,
    card_views.quality_score_json,
    videos.id AS video_id,
    videos.source_path
FROM card_views
JOIN card_instances ON card_instances.id = card_views.card_instance_id
JOIN videos ON videos.id = card_instances.video_id
WHERE card_views.id = ?
"""
STORAGE_SAVED_CARD_INSERT = """
INSERT INTO saved_cards (
    detection_id,
    video_id,
    image_path,
    final_score,
    review_state,
    source_path,
    timestamp_ms,
    score_components_json
)
VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
"""
STORAGE_REVIEW_DECISION_INSERT = (
    "INSERT INTO review_decisions (saved_card_id, decision, notes) VALUES (?, ?, ?)"
)
STORAGE_SAVED_CARD_REVIEW_UPDATE = "UPDATE saved_cards SET review_state = ? WHERE id = ?"
STORAGE_SAVED_CARDS_BASE = """
SELECT
    sc.id,
    sc.detection_id,
    sc.image_path,
    sc.final_score,
    sc.review_state,
    sc.source_path,
    sc.timestamp_ms,
    sc.score_components_json
FROM saved_cards sc
JOIN card_views cv ON cv.id = sc.detection_id
JOIN card_instances ci ON ci.id = cv.card_instance_id
"""


def storage_pragma_table_info(table: str) -> str:
    return f"PRAGMA table_info({table})"


def storage_alter_table_add_column(table: str, column: str, ddl: str) -> str:
    return f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"
