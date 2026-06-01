# V5.5 Data Access Layer Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate all remaining raw `sqlite3` callsites to use `card_capture.data.connection.read_connection` for reads and repositories for writes, fulfilling the V5.5 DAL mandates.

**Architecture:**
- Services are updated to use injected repositories from `app.state`.
- Read operations use `read_connection` which provides a read-only, row-factory enabled connection.
- Write operations use the `Writer` via repositories to ensure single-writer integrity.

**Tech Stack:** Python, SQLite, FastAPI, Pytest.

---

### Task 1: Migrate TrainingService

**Files:**
- Modify: `app/services/training_service.py`

- [ ] **Step 1: Update imports and constructor**
Remove `import sqlite3` (if global) and ensure `read_connection` is used.

- [ ] **Step 2: Update `get_stats` to use `read_connection` and indexed access**
```python
    def get_stats(self) -> dict:
        from card_capture.data.connection import read_connection
        with read_connection(self.db_path) as conn:
            presence_pending = conn.execute(
                "SELECT COUNT(*) FROM presence_samples WHERE label IS NULL"
            ).fetchone()[0]
            fb_pending = conn.execute(
                """SELECT COUNT(*) FROM card_instances ci
                   WHERE NOT EXISTS (
                       SELECT 1 FROM fb_labels fl WHERE fl.instance_id = ci.track_id
                   )"""
            ).fetchone()[0]
            corner_pending = conn.execute(
                "SELECT COUNT(*) FROM corner_samples WHERE label IS NULL"
            ).fetchone()[0]

            accuracies = {}
            for model in ("presence", "fb_classifier"):
                row = conn.execute(
                    "SELECT eval_metrics_json FROM model_versions "
                    "WHERE model_name=? ORDER BY created_at DESC LIMIT 1",
                    (model,),
                ).fetchone()
                if row and row["eval_metrics_json"]:
                    import json
                    m = json.loads(row["eval_metrics_json"])
                    accuracies[model] = m.get("accuracy")

            history_rows = conn.execute(
                "SELECT model_name, eval_metrics_json, created_at FROM model_versions "
                "ORDER BY created_at ASC"
            ).fetchall()
            history = []
            for r in history_rows:
                if r["eval_metrics_json"]:
                    import json
                    m = json.loads(r["eval_metrics_json"])
                    history.append({
                        "model": r["model_name"],
                        "accuracy": m.get("accuracy"),
                        "created_at": r["created_at"],
                    })
        # ... return dict ...
```

- [ ] **Step 3: Update `snapshot_baseline` to use `self._training_repo`**
Ensure `TrainingRepository` has `snapshot_baseline` or implement it using `writer.queue`.

- [ ] **Step 4: Update `_record_model_version` to use `self._ml_repo`**
```python
    def _record_model_version(self, model_name: str, metrics: dict) -> None:
        if self._ml_repo:
            self._ml_repo.record_version(model_name, metrics)
```

- [ ] **Step 5: Verify with tests**
Run: `pytest tests/app/test_training_service.py` (if exists) or similar.

---

### Task 2: Migrate LabelingService

**Files:**
- Modify: `app/services/labeling_service.py`

- [ ] **Step 1: Update `get_truth` to use `read_connection`**
- [ ] **Step 2: Update `put_truth` to use `self._repo.put_truth`**
- [ ] **Step 3: Update `post_fb_label` to use `self._repo.post_fb_label`**
- [ ] **Step 4: Update `next_fb_candidate` to use `read_connection`**
- [ ] **Step 5: Update `list_clusters` to use `read_connection`**
- [ ] **Step 6: Update `patch_cluster` to use `self._repo.patch_cluster`**

---

### Task 3: Migrate RegressionService, VideoService, RunService, CardService

**Files:**
- Modify: `app/services/regression_service.py`
- Modify: `app/services/video_service.py`
- Modify: `app/services/runs_service.py`
- Modify: `app/services/cards_service.py`

- [ ] **Step 1: RegressionService:** Update `compare` to use `read_connection`.
- [ ] **Step 2: VideoService:** Update constructor to accept `videos_repo`. Update `list_videos`, `get_video` to use `read_connection`. Update `add_video`, `update_status`, `delete_video` to use `videos_repo`.
- [ ] **Step 3: RunService:** Update constructor to accept `runs_repo` and `events_repo`. Update read methods to use `read_connection`.
- [ ] **Step 4: CardService:** Update constructor to accept `cards_repo`. Update read methods to use `read_connection`.

---

### Task 4: Migrate MiningService and ResourceSampler

**Files:**
- Modify: `app/services/mining_service.py`
- Modify: `app/services/resource_sampler.py`

- [ ] **Step 1: MiningService:** Update constructor to accept `training_repo`. Use `read_connection` for `list_hard_cases`. Use `training_repo` for `promote_to_training` (DB update).
- [ ] **Step 2: ResourceSampler:** Update `_sample` to use `TelemetryRepository` for insertion instead of raw `sqlite3`.

---

### Task 5: Final Cleanup and Verification

- [ ] **Step 1: Search and remove remaining `import sqlite3`**
- [ ] **Step 2: Run all tests**
Run: `pytest tests/data/ tests/app/`
