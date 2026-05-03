from pathlib import Path

import numpy as np
import cv2
from fastapi.testclient import TestClient

from card_capture.models import CardDetection, CornerDetection, QualityScore
from card_capture.review import create_app
from card_capture.storage import Storage


def test_review_page_serves_saved_card_images(tmp_path: Path):
    image_path = tmp_path / "card.jpg"
    cv2.imwrite(str(image_path), np.full((10, 10, 3), 200, dtype=np.uint8))

    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("video.mov", "hash", 1000, 100, 100)
    detection_id = storage.add_detection(
        video_id=video_id,
        detection=CardDetection(
            frame_index=0,
            timestamp_ms=0,
            polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            confidence=0.9,
        ),
        crop_path=str(image_path),
        source_frame_path=None,
        score=QualityScore(total=0.8, components={"confidence": 0.9}),
        crop_width=10,
        crop_height=10,
    )
    saved_id = storage.add_saved_card(detection_id, str(image_path), 0.8)

    client = TestClient(create_app(tmp_path / "cards.sqlite"))
    page = client.get("/")
    image = client.get(f"/images/{saved_id}")

    assert f'/images/{saved_id}' in page.text
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"


def test_card_view_route_falls_back_to_fused_when_rectified_missing(tmp_path: Path):
    fused_path = tmp_path / "fused.jpg"
    cv2.imwrite(str(fused_path), np.full((10, 10, 3), 180, dtype=np.uint8))

    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("video.mov", "hash", 1000, 100, 100)
    instance_id = storage.add_card_instance(video_id=video_id, track_id="track_1", angle="Front")
    storage.update_instance_fusion(instance_id, str(fused_path))

    view_id = storage.add_card_view(
        card_instance_id=instance_id,
        frame_index=1,
        timestamp_ms=100,
        detection=CornerDetection(
            corners=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            confidence=0.9,
        ),
        rectified_path=str(tmp_path / "missing_rectified.jpg"),
        is_canonical=True,
    )

    client = TestClient(create_app(tmp_path / "cards.sqlite"))
    response = client.get(f"/card_views/{view_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
