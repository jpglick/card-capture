-- migrations/0009_detect_telemetry.sql
-- Persist detect-stage diagnostic snapshot per run.

ALTER TABLE pipeline_runs ADD COLUMN detect_telemetry_json TEXT;
