CREATE TABLE IF NOT EXISTS presence_samples (
    id           INTEGER PRIMARY KEY,
    run_id       TEXT    NOT NULL,
    video_id     INTEGER NOT NULL,
    frame_index  INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    image_path   TEXT    NOT NULL,
    label        TEXT    CHECK (label IN ('present', 'absent')),
    labeled_at   TEXT,
    created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_presence_unlabeled
    ON presence_samples (label) WHERE label IS NULL;

CREATE TABLE IF NOT EXISTS corner_samples (
    id                INTEGER PRIMARY KEY,
    run_id            TEXT    NOT NULL,
    video_id          INTEGER NOT NULL,
    frame_index       INTEGER NOT NULL,
    image_path        TEXT    NOT NULL,
    predicted_corners TEXT    NOT NULL,  -- JSON: [[x,y],[x,y],[x,y],[x,y]]
    confidence        REAL    NOT NULL,
    label             TEXT    CHECK (label IN ('correct', 'adjusted', 'negative')),
    corrected_corners TEXT,              -- JSON: [[x,y],...] when label='adjusted'
    labeled_at        TEXT,
    created_at        TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_corner_unlabeled
    ON corner_samples (label) WHERE label IS NULL;

CREATE TABLE IF NOT EXISTS benchmark_snapshots (
    id              INTEGER PRIMARY KEY,
    job_id          TEXT    NOT NULL,   -- TrainingJob.job_id at retrain time
    run_id          TEXT    NOT NULL,   -- pipeline_runs.run_id being snapshotted
    cards_extracted INTEGER NOT NULL,
    snapshotted_at  TEXT    DEFAULT CURRENT_TIMESTAMP
);
