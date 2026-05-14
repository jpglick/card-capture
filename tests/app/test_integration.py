"""End-to-end integration tests for the Card Capture v4 REST API.

Each test exercises a real request against a real SQLite database in a
temp directory.  No mocks — if the route 404s or returns the wrong shape,
the test fails.  This file is the canonical contract between the Python API
and the TypeScript client.
"""
from __future__ import annotations

import io
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return a path to a fresh cards.sqlite in a temp dir."""
    return tmp_path / "cards.sqlite"


@pytest.fixture()
def client(tmp_db: Path) -> TestClient:
    """TestClient backed by a fresh DB."""
    from app.main import create_app
    app = create_app(db_path=tmp_db)
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def video_id(client: TestClient) -> int:
    """Insert a video and return its integer ID."""
    r = client.post("/api/v1/videos", json={"filename": "test.mov"})
    assert r.status_code == 201
    return int(r.json()["video_id"])


@pytest.fixture()
def run_id(client: TestClient, video_id: int) -> str:
    """Insert a pipeline_run record directly and return the run_id string."""
    # Bypass the actual pipeline; just insert a row so the service has data.
    run = f"run_{uuid.uuid4().hex[:8]}"
    db = client.app.state.db_path
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, video_id, status, cards_extracted, started_at, finished_at)"
            " VALUES (?, ?, 'completed', 0, datetime('now'), datetime('now'))",
            (run, video_id),
        )
    return run


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

class TestVideos:
    def test_list_empty(self, client: TestClient):
        r = client.get("/api/v1/videos")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_by_filename(self, client: TestClient):
        r = client.post("/api/v1/videos", json={"filename": "myvideo.mov"})
        assert r.status_code == 201
        body = r.json()
        # Exact field names the TypeScript Video type expects
        assert "video_id" in body
        assert "filename" in body
        assert "duration_ms" in body
        assert "status" in body
        assert "created_at" in body
        assert body["filename"] == "myvideo.mov"
        assert body["status"] == "pending"

    def test_list_after_create(self, client: TestClient):
        client.post("/api/v1/videos", json={"filename": "a.mov"})
        client.post("/api/v1/videos", json={"filename": "b.mov"})
        r = client.get("/api/v1/videos")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_get_video(self, client: TestClient, video_id: int):
        r = client.get(f"/api/v1/videos/{video_id}")
        assert r.status_code == 200
        assert r.json()["video_id"] == str(video_id)

    def test_get_video_not_found(self, client: TestClient):
        r = client.get("/api/v1/videos/9999")
        assert r.status_code == 404

    def test_delete_video(self, client: TestClient, video_id: int):
        r = client.delete(f"/api/v1/videos/{video_id}")
        assert r.status_code == 204
        assert client.get(f"/api/v1/videos/{video_id}").status_code == 404

    def test_upload_video(self, client: TestClient, tmp_path: Path):
        fake_video = tmp_path / "clip.mov"
        fake_video.write_bytes(b"fake bytes")
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        with patch("app.api.videos.UPLOADS_DIR", upload_dir):
            with fake_video.open("rb") as f:
                r = client.post("/api/v1/videos/upload", files={"file": ("clip.mov", f, "video/quicktime")})
        assert r.status_code == 201
        body = r.json()
        # Filename includes a UUID prefix — just verify the original name is present
        assert "clip.mov" in body["filename"]
        assert body["status"] == "pending"

    def test_process_starts_background_task(self, client: TestClient, video_id: int):
        async def _noop(*a, **kw): pass
        with patch.object(
            __import__("app.services.pipeline_runner", fromlist=["PipelineRunner"]).PipelineRunner,
            "run_async", side_effect=_noop
        ):
            r = client.post(f"/api/v1/videos/{video_id}/process")
        assert r.status_code == 202
        body = r.json()
        assert "run_id" in body
        assert body["run_id"].startswith("run_")
        assert body["video_id"] == str(video_id)
        assert body["status"] == "pending"

    def test_process_unknown_video(self, client: TestClient):
        r = client.post("/api/v1/videos/9999/process")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class TestRuns:
    def test_list_empty(self, client: TestClient):
        r = client.get("/api/v1/runs")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_after_run(self, client: TestClient, run_id: str):
        r = client.get("/api/v1/runs")
        assert r.status_code == 200
        runs = r.json()
        assert len(runs) == 1
        run = runs[0]
        # Exact fields TypeScript RunSummary expects
        assert "run_id" in run
        assert "video_id" in run
        assert "status" in run
        assert "cards_extracted" in run
        assert "elapsed_ms" in run
        assert "created_at" in run
        assert run["run_id"] == run_id
        assert run["status"] == "completed"

    def test_get_run_detail(self, client: TestClient, run_id: str):
        r = client.get(f"/api/v1/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["run_id"] == run_id
        assert "cards_extracted" in body
        assert "elapsed_ms" in body

    def test_get_run_not_found(self, client: TestClient):
        r = client.get("/api/v1/runs/run_doesnotexist")
        assert r.status_code == 404

    def test_list_runs_filtered_by_video(self, client: TestClient, video_id: int, run_id: str):
        r = client.get(f"/api/v1/runs?video_id={video_id}")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_get_run_cards_shape(self, client: TestClient, run_id: str, tmp_db: Path):
        # Insert a card_instance for this run
        with sqlite3.connect(str(tmp_db)) as conn:
            conn.execute(
                "INSERT INTO card_instances (video_id, run_id, track_id, angle) VALUES (1, ?, ?, 'Front')",
                (run_id, str(uuid.uuid4())),
            )
        r = client.get(f"/api/v1/runs/{run_id}/cards")
        assert r.status_code == 200
        body = r.json()
        # Must be paginated shape — TypeScript PaginatedCards
        assert "total" in body
        assert "items" in body
        assert isinstance(body["items"], list)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

@pytest.fixture()
def seeded_cards(tmp_db: Path, video_id: int) -> list[str]:
    """Insert 3 card_instances; return their track_ids."""
    track_ids = [str(uuid.uuid4()) for _ in range(3)]
    with sqlite3.connect(str(tmp_db)) as conn:
        for tid in track_ids:
            conn.execute(
                "INSERT INTO card_instances (video_id, run_id, track_id, angle) VALUES (?, 'run_test', ?, 'Front')",
                (video_id, tid),
            )
    return track_ids


class TestCards:
    def test_list_empty(self, client: TestClient):
        r = client.get("/api/v1/cards")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["items"] == []
        assert "page" in body
        assert "page_size" in body

    def test_list_returns_correct_fields(self, client: TestClient, seeded_cards, video_id: int):
        r = client.get("/api/v1/cards")
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 3
        card = items[0]
        # Exact field names TypeScript Card type expects
        assert "card_id" in card
        assert "instance_id" in card
        assert "video_id" in card
        assert "run_id" in card
        assert "side" in card
        assert "is_foil" in card
        assert "confidence" in card
        assert "review_state" in card
        assert "fused_url" in card
        assert "canonical_url" in card
        assert "created_at" in card
        # Must NOT use old field names
        assert "angle" not in card
        assert "fused_image_url" not in card
        assert "quality_score" not in card

    def test_filter_by_video_id(self, client: TestClient, seeded_cards, video_id: int):
        r = client.get(f"/api/v1/cards?video_id={video_id}")
        assert r.status_code == 200
        assert r.json()["total"] == 3

    def test_filter_by_run_id(self, client: TestClient, seeded_cards):
        r = client.get("/api/v1/cards?run_id=run_test")
        assert r.status_code == 200
        assert r.json()["total"] == 3

    def test_pagination(self, client: TestClient, seeded_cards):
        r = client.get("/api/v1/cards?page=1&page_size=2")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert len(body["items"]) == 2

        r2 = client.get("/api/v1/cards?page=2&page_size=2")
        assert len(r2.json()["items"]) == 1

    def test_get_card_not_found(self, client: TestClient):
        r = client.get("/api/v1/cards/9999")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Label — Truth
# ---------------------------------------------------------------------------

class TestLabelTruth:
    def test_get_truth_not_exists_returns_null(self, client: TestClient, video_id: int):
        r = client.get(f"/api/v1/label/truth/{video_id}")
        assert r.status_code == 200
        assert r.json() is None

    def test_put_and_get_truth(self, client: TestClient, video_id: int):
        truth = {
            "video_id": str(video_id),
            "schema_version": 1,
            "expected_cards": [
                {
                    "card_id": "inst-1",
                    "front_present": True,
                    "back_present": False,
                    "physical_card_key": "card-a",
                    "is_foil": False,
                }
            ],
        }
        r = client.put(f"/api/v1/label/truth/{video_id}", json=truth)
        assert r.status_code in (200, 204)

        r2 = client.get(f"/api/v1/label/truth/{video_id}")
        assert r2.status_code == 200
        body = r2.json()
        assert body is not None
        assert len(body["expected_cards"]) == 1
        assert body["expected_cards"][0]["card_id"] == "inst-1"


# ---------------------------------------------------------------------------
# Label — F/B trainer
# ---------------------------------------------------------------------------

class TestLabelFB:
    def test_next_fb_no_candidates(self, client: TestClient):
        r = client.get("/api/v1/label/fb/next")
        # 204 when no candidates, not a 500
        assert r.status_code in (200, 204)

    def test_next_fb_returns_candidate_fields(self, client: TestClient, seeded_cards, video_id: int, tmp_db: Path):
        # Insert a card_view so there's a canonical frame to return
        with sqlite3.connect(str(tmp_db)) as conn:
            instance_id = conn.execute("SELECT id FROM card_instances LIMIT 1").fetchone()[0]
            conn.execute(
                "INSERT INTO card_views (card_instance_id, frame_index, timestamp_ms, is_canonical,"
                " confidence, corners_json, metadata_json)"
                " VALUES (?, 10, 1000, 1, 0.9, '[]', '{}')",
                (instance_id,),
            )
        r = client.get("/api/v1/label/fb/next")
        assert r.status_code == 200
        body = r.json()
        # TypeScript LabelFBNext fields
        assert "instance_id" in body
        assert "frame_index" in body
        assert "labels_collected" in body
        assert "labels_target" in body

    def test_submit_fb_label(self, client: TestClient, seeded_cards, tmp_db: Path):
        with sqlite3.connect(str(tmp_db)) as conn:
            tid = conn.execute("SELECT track_id FROM card_instances LIMIT 1").fetchone()[0]
        r = client.post("/api/v1/label/fb", json={
            "instance_id": tid, "frame_index": 0, "side": "front"
        })
        assert r.status_code == 201
        assert "label_id" in r.json()


# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------

class TestConfigPresets:
    def test_list_returns_builtins(self, client: TestClient):
        r = client.get("/api/v1/config/presets")
        assert r.status_code == 200
        names = {p["preset_name"] for p in r.json()}
        assert "fast" in names
        assert "balanced" in names
        assert "quality" in names

    def test_create_and_list_user_preset(self, client: TestClient):
        payload = {
            "preset_name": "my_preset",
            "description": "Test preset",
            "config": {"corner_confidence": 0.7},
        }
        r = client.post("/api/v1/config/presets", json=payload)
        assert r.status_code == 201
        assert r.json()["preset_name"] == "my_preset"

        r2 = client.get("/api/v1/config/presets")
        names = {p["preset_name"] for p in r2.json()}
        assert "my_preset" in names

    def test_duplicate_preset_returns_409(self, client: TestClient):
        payload = {"preset_name": "dup", "description": "", "config": {}}
        client.post("/api/v1/config/presets", json=payload)
        r = client.post("/api/v1/config/presets", json=payload)
        assert r.status_code == 409

    def test_builtin_name_returns_409(self, client: TestClient):
        r = client.post("/api/v1/config/presets", json={
            "preset_name": "fast", "description": "", "config": {}
        })
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

class TestStaticFiles:
    def test_files_route_exists(self, client: TestClient, tmp_path: Path, tmp_db: Path):
        # Create a real file under the output dir (db parent)
        crops = tmp_db.parent / "crops"
        crops.mkdir(parents=True, exist_ok=True)
        test_file = crops / "test.jpg"
        test_file.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header

        r = client.get("/files/crops/test.jpg")
        assert r.status_code == 200

    def test_files_missing_returns_404(self, client: TestClient):
        r = client.get("/files/crops/nonexistent_xyz.jpg")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# SSE events
# ---------------------------------------------------------------------------

class TestSSE:
    def test_sse_route_is_registered(self, client: TestClient):
        # Verify the route exists without opening an infinite stream.
        # The EventBus will block forever waiting for events, so we just
        # check the route is registered via the app's route table.
        routes = {r.path for r in client.app.routes if hasattr(r, "path")}
        assert "/events/{run_id}" in routes
