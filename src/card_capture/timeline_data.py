import json

from card_capture.data.connection import read_connection


def get_timeline_data(db_path="card_capture_output/cards.sqlite"):
    with read_connection(db_path) as conn:
        events = conn.execute("SELECT * FROM pipeline_events ORDER BY frame_index").fetchall()
        instances = conn.execute(
            """
            SELECT ci.id, ci.video_id, ci.session_id, ci.angle, ci.is_duplicate_of,
                   MIN(cv.timestamp_ms) as start_time, MAX(cv.timestamp_ms) as end_time,
                   COUNT(cv.id) as detection_count,
                   MAX(sc.final_score) as max_score
            FROM card_instances ci
            LEFT JOIN card_views cv ON cv.card_instance_id = ci.id
            LEFT JOIN saved_cards sc ON sc.detection_id = cv.id
            GROUP BY ci.id
            ORDER BY start_time
            """
        ).fetchall()

    return [dict(e) for e in events], [dict(i) for i in instances]
