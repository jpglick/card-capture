from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.cards import CardsRepository
from card_capture.data.repositories.runs import RunsRepository
from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer
from card_capture.pipeline.request import CardRecord


def _setup_run(prod_db: Path, writer: Writer) -> tuple[int, str]:
    vid = VideosRepository(writer=writer, db_path=prod_db).register(
        source_path="/v.MOV", file_hash="h", duration_ms=1, width=100, height=100,
    )
    writer.flush()
    RunsRepository(writer=writer, db_path=prod_db).mark_started("run_x", vid)
    writer.flush()
    return vid, "run_x"


def test_store_and_list_for_run(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        video_id, run_id = _setup_run(prod_db, writer)
        repo = CardsRepository(writer=writer, db_path=prod_db)
        repo.store_final_cards(run_id=run_id, video_id=video_id, cards=[
            CardRecord(
                card_instance_id="card_0",
                front_crop="artifact://local/run_x/crops/card_0_front.png",
                back_crop="artifact://local/run_x/crops/card_0_back.png",
                quality={"sharpness": 12.3, "glare": 0.05},
            ),
            CardRecord(
                card_instance_id="card_1",
                front_crop="artifact://local/run_x/crops/card_1_front.png",
                quality={"sharpness": 14.0},
            ),
        ])
        writer.flush()
        cards = repo.list_for_run(run_id)
    finally:
        writer.stop()

    assert len(cards) == 2
    by_id = {c["card_instance_id"]: c for c in cards}
    assert by_id["card_0"]["back_crop"].endswith("card_0_back.png")
    assert by_id["card_0"]["quality"]["sharpness"] == 12.3
    assert by_id["card_1"]["back_crop"] is None
