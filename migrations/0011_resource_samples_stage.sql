-- migrations/0011_resource_samples_stage.sql
ALTER TABLE run_resource_samples ADD COLUMN stage TEXT DEFAULT 'init';