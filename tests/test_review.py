from pathlib import Path

import numpy as np
import cv2
from fastapi.testclient import TestClient

from card_capture.models import CardDetection, QualityScore
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
