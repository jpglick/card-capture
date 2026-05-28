from pathlib import Path

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.cards import CardsRepository
from card_capture.pipeline.request import CardRecord


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE card_instances(
            card_instance_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            front_crop TEXT NOT NULL,
            back_crop TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE card_views(
            card_instance_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (card_instance_id, metric)
        )
    """)
    conn.close()


def test_store_and_get(tmp_path):
    db = tmp_path / "c.db"; _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = CardsRepository(writer=writer, db_path=db)
        repo.store_final_cards("r1", [
            CardRecord(
                card_instance_id="card_0",
                front_crop="artifact://local/r1/crops/card_0.png",
                back_crop=None,
                quality={"sharpness": 12.3, "glare": 0.05},
            ),
        ])
        writer.flush()
        cards = repo.list_for_run("r1")
    finally:
        writer.stop()
    assert len(cards) == 1
    assert cards[0]["card_instance_id"] == "card_0"
    assert cards[0]["quality"]["sharpness"] == 12.3
