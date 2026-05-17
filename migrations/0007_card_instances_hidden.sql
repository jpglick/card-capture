ALTER TABLE card_instances ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_card_instances_hidden ON card_instances(hidden);
