from pathlib import Path

import numpy as np

from card_capture.models import CardDetection, QualityScore
from card_capture.storage import Storage


def test_storage_records_video_detection_saved_card_and_review(tmp_path: Path):
    db_path = tmp_path / "cards.sqlite"
    storage = Storage(db_path)
    storage.initialize()

    video_id = storage.add_video(
        source_path="/videos/input.mov",
        file_hash="abc123",
        duration_ms=1200,
        width=1920,
        height=1080,
    )

    detection = CardDetection(
        frame_index=12,
        timestamp_ms=400,
        polygon=((10.0, 20.0), (210.0, 20.0), (210.0, 320.0), (10.0, 320.0)),
        confidence=0.91,
    )
    score = QualityScore(
        total=0.82,
        components={"sharpness": 0.9, "glare": 0.8, "size": 0.7, "confidence": 0.91},
    )

    detection_id = storage.add_detection(
        video_id=video_id,
        detection=detection,
        crop_path="output/crops/card.jpg",
        source_frame_path="output/frames/frame.jpg",
        score=score,
        crop_width=200,
        crop_height=300,
    )
    saved_id = storage.add_saved_card(
        detection_id=detection_id,
        image_path="output/best/card.jpg",
        final_score=0.82,
    )
    decision_id = storage.set_review_decision(saved_id, "accepted", "front is readable")

    saved_cards = storage.list_saved_cards()

    assert video_id == 1
    assert detection_id == 1
    assert saved_id == 1
    assert decision_id == 1
    assert saved_cards == [
        {
            "id": 1,
            "detection_id": 1,
            "image_path": "output/best/card.jpg",
            "final_score": 0.82,
            "review_state": "accepted",
            "source_path": "/videos/input.mov",
            "timestamp_ms": 400,
            "score_components": {
                "sharpness": 0.9,
                "glare": 0.8,
                "size": 0.7,
                "confidence": 0.91,
            },
        }
    ]


def test_storage_filters_saved_cards_by_review_state(tmp_path: Path):
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("video.mov", "hash", 1000, 100, 100)
    detection = CardDetection(
        frame_index=1,
        timestamp_ms=100,
        polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        confidence=0.8,
    )
    score = QualityScore(total=0.5, components={"confidence": 0.8})
    first_detection = storage.add_detection(video_id, detection, "a.jpg", None, score, 10, 10)
    second_detection = storage.add_detection(video_id, detection, "b.jpg", None, score, 10, 10)
    accepted_id = storage.add_saved_card(first_detection, "a.jpg", 0.5)
    storage.add_saved_card(second_detection, "b.jpg", 0.4)
    storage.set_review_decision(accepted_id, "accepted", "")

    accepted = storage.list_saved_cards(review_state="accepted")

    assert [card["image_path"] for card in accepted] == ["a.jpg"]
