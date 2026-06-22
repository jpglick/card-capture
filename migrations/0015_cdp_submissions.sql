-- migrations/0015_cdp_submissions.sql
-- Tracks CardDealerPro submission state for each card instance.

CREATE TABLE IF NOT EXISTS cdp_submissions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    card_instance_id    INTEGER NOT NULL REFERENCES card_instances(id) ON DELETE CASCADE,
    cdp_batch_id        TEXT NOT NULL,
    cdp_card_id         TEXT,
    status              TEXT NOT NULL DEFAULT 'submitted',
    -- status values: 'submitted' | 'processing' | 'identified' | 'failed'
    identified_name     TEXT,
    suggested_price     REAL,
    raw_response        TEXT,
    submitted_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(card_instance_id)
);

CREATE INDEX IF NOT EXISTS idx_cdp_submissions_batch ON cdp_submissions(cdp_batch_id);
CREATE INDEX IF NOT EXISTS idx_cdp_submissions_status ON cdp_submissions(status);
