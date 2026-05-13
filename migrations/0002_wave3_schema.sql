-- migrations/0002_wave3_schema.sql

-- Add reid_embedding to card_instances for cross-session/video deduplication.
-- BLOB stores the raw float32 vector (512-dim).
ALTER TABLE card_instances ADD COLUMN reid_embedding BLOB;
