-- Allow 'no_card' as a valid fb_labels side value.
-- SQLite can't ALTER CHECK constraints, so we recreate the table.

CREATE TABLE IF NOT EXISTS fb_labels_new (
    label_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_run_id INTEGER,
    instance_id   TEXT    NOT NULL,
    frame_index   INTEGER NOT NULL,
    side          TEXT    NOT NULL CHECK (side IN ('front', 'back', 'uncertain', 'no_card')),
    labeler       TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO fb_labels_new SELECT * FROM fb_labels;

DROP TABLE fb_labels;
ALTER TABLE fb_labels_new RENAME TO fb_labels;

CREATE INDEX IF NOT EXISTS idx_fb_labels_instance ON fb_labels(instance_id);
