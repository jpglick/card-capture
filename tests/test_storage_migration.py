import pytest
from pathlib import Path
from card_capture.stages.store.storage import Storage

def test_storage_fusion_columns(tmp_path: Path):
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("test.mov", "h1", 100, 100, 100)
    instance_id = storage.add_card_instance(video_id, "card_1")

    # Verify fusion column exists
    with storage._connect() as conn:
        cols = conn.execute("PRAGMA table_info(card_instances)").fetchall()
        col_names = [c["name"] for c in cols]
        assert "fused_image_path" in col_names

    # Test update_instance_fusion (to be implemented)
    storage.update_instance_fusion(instance_id, "fused/1.jpg")
    
    with storage._connect() as conn:
        row = conn.execute("SELECT fused_image_path FROM card_instances WHERE id = ?", (instance_id,)).fetchone()
        assert row["fused_image_path"] == "fused/1.jpg"

def test_storage_view_metrics(tmp_path: Path):
    from card_capture.core.models import CornerDetection
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("test.mov", "h1", 100, 100, 100)
    instance_id = storage.add_card_instance(video_id, "card_1")

    # Verify columns exist
    with storage._connect() as conn:
        cols = conn.execute("PRAGMA table_info(card_views)").fetchall()
        col_names = [c["name"] for c in cols]
        assert "glare_x" in col_names
        assert "glare_y" in col_names
        assert "sharpness" in col_names

    # Test add_card_view with metrics
    view_id = storage.add_card_view(
        card_instance_id=instance_id,
        frame_index=1,
        timestamp_ms=100,
        detection=CornerDetection(((0,0),(1,0),(1,1),(0,1)), 0.9, {}),
        glare_x=10.0,
        glare_y=20.0,
        sharpness=0.5
    )
    
    with storage._connect() as conn:
        row = conn.execute("SELECT glare_x, glare_y, sharpness FROM card_views WHERE id = ?", (view_id,)).fetchone()
        assert row["glare_x"] == 10.0
        assert row["glare_y"] == 20.0
        assert row["sharpness"] == 0.5
