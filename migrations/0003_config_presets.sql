-- migrations/0003_config_presets.sql
-- Add user-defined config presets table.

CREATE TABLE IF NOT EXISTS config_presets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    preset_name TEXT    NOT NULL UNIQUE,
    description TEXT    NOT NULL DEFAULT '',
    config_json TEXT    NOT NULL,           -- JSON blob of threshold overrides
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
