import json

from card_capture.data.connection import read_connection
from card_capture.data.sql_queries import TIMELINE_EVENTS_BY_FRAME, TIMELINE_INSTANCES_SUMMARY


def get_timeline_data(db_path="var/db/cards.sqlite"):
    with read_connection(db_path) as conn:
        events = conn.execute(TIMELINE_EVENTS_BY_FRAME).fetchall()
        instances = conn.execute(TIMELINE_INSTANCES_SUMMARY).fetchall()

    return [dict(e) for e in events], [dict(i) for i in instances]
