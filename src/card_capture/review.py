import json
from pathlib import Path
from typing import Optional

from .storage import Storage

def create_app(db_path: Path):
    try:
        from fastapi import FastAPI, Form, Request
        from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
        from fastapi.templating import Jinja2Templates
    except ImportError as exc:
        raise RuntimeError("Review UI requires: pip install '.[review]'") from exc

    app = FastAPI(title="Card Capture Review")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    storage = Storage(db_path)
    storage.initialize()
    db_path = Path(db_path).resolve()

    def _truth_path_for_video(source_path: str) -> Path:
        stem = Path(source_path).stem
        return (
            Path("tests/fixtures/golden_corpus") / stem / f"{stem}.truth.json"
        ).resolve()

    def _resolve_existing_path(path_value: Optional[str]) -> Optional[Path]:
        if not path_value:
            return None
        raw = Path(path_value).expanduser()
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(
                [
                    (Path.cwd() / raw),
                    (db_path.parent / raw),
                    (db_path.parent.parent / raw),
                ]
            )
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return candidate
        return None

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, state: str = "pending"):
        cards = storage.list_saved_cards(
            review_state=None if state == "all" else state,
            include_duplicates=False,
        )
        cards.sort(key=lambda card: (card["timestamp_ms"], card["id"]))
        
        with storage._connect() as conn:
            for card in cards:
                # Join to find the instance_id linked to the card_view (detection_id)
                row = conn.execute("""
                    SELECT ci.id as instance_id, ci.fused_image_path, ci.angle, ci.session_id
                    FROM card_views cv
                    JOIN card_instances ci ON cv.card_instance_id = ci.id
                    WHERE cv.id = ?
                """, (card["detection_id"],)).fetchone()
                
                if row:
                    card["instance_id"] = row["instance_id"]
                    card["fused_image_path"] = row["fused_image_path"]
                    card["angle"] = row["angle"]
                    card["session_id"] = row["session_id"]
                    canonical_rows = conn.execute(
                        """
                        SELECT id, rectified_path
                        FROM card_views
                        WHERE card_instance_id = ? AND is_canonical = 1 AND rectified_path IS NOT NULL
                        ORDER BY id ASC
                        """,
                        (row["instance_id"],),
                    ).fetchall()
                    card["canonical_views"] = [
                        {"view_id": int(view["id"]), "rectified_path": view["rectified_path"]}
                        for view in canonical_rows
                    ]
                else:
                    card["instance_id"] = None
                    card["fused_image_path"] = None
                    card["angle"] = None
                    card["session_id"] = None
                    card["canonical_views"] = []
            
            # Fetch pipeline events (resets, flips)
            events = conn.execute("""
                SELECT frame_index, timestamp_ms, event_type, data_json
                FROM pipeline_events
                ORDER BY timestamp_ms ASC
            """).fetchall()
            
            # Map events to dicts
            pipeline_events = []
            for e in events:
                pipeline_events.append({
                    "frame_index": e["frame_index"],
                    "timestamp_ms": e["timestamp_ms"],
                    "event_type": e["event_type"],
                    "data": json.loads(e["data_json"]) if e["data_json"] else {}
                })
        
        return templates.TemplateResponse(
            request,
            "review.html",
            {"cards": cards, "state": state, "events": pipeline_events},
        )

    @app.get("/timeline", response_class=HTMLResponse)
    def timeline(request: Request, video: Optional[int] = None):
        with storage._connect() as conn:
            # Fetch pipeline events (resets, flips)
            events_query = "SELECT frame_index, timestamp_ms, event_type, data_json FROM pipeline_events"
            params = []
            if video is not None:
                events_query += " WHERE video_id = ?"
                params.append(video)
            events_query += " ORDER BY timestamp_ms ASC"
            events = conn.execute(events_query, params).fetchall()
            
            pipeline_events = []
            for e in events:
                pipeline_events.append({
                    "frame_index": e["frame_index"],
                    "timestamp_ms": e["timestamp_ms"],
                    "event_type": e["event_type"],
                    "data": json.loads(e["data_json"]) if e["data_json"] else {}
                })

            instances_query = """
                SELECT ci.id as instance_id, ci.session_id, ci.angle, ci.is_duplicate_of, ci.video_id, ci.fused_image_path,
                       MIN(cv.timestamp_ms) as start_time, MAX(cv.timestamp_ms) as end_time,
                       COUNT(cv.id) as detection_count,
                       MIN(cv.id) as first_view_id
                FROM card_instances ci
                LEFT JOIN card_views cv ON cv.card_instance_id = ci.id
            """
            if video is not None:
                instances_query += " WHERE ci.video_id = ?"
            instances_query += " GROUP BY ci.id ORDER BY start_time ASC"
            
            instances = conn.execute(instances_query, params).fetchall()
            
            instance_data = []
            for i in instances:
                instance_data.append({
                    "instance_id": i["instance_id"],
                    "session_id": i["session_id"],
                    "angle": i["angle"],
                    "is_duplicate_of": i["is_duplicate_of"],
                    "video_id": i["video_id"],
                    "start_time": i["start_time"],
                    "end_time": i["end_time"],
                    "detection_count": i["detection_count"],
                    "fused_image_path": i["fused_image_path"],
                    "first_view_id": i["first_view_id"]
                })
                
        return templates.TemplateResponse(
            request,
            "timeline.html",
            {"events": pipeline_events, "instances": instance_data, "video_id": video},
        )

    @app.get("/label/{video_id}", response_class=HTMLResponse)
    def label_get(request: Request, video_id: int):
        with storage._connect() as conn:
            video_row = conn.execute(
                "SELECT id, source_path FROM videos WHERE id = ?", (video_id,),
            ).fetchone()
            if video_row is None:
                return HTMLResponse(f"video {video_id} not found", status_code=404)

            instances = conn.execute(
                """
                SELECT ci.id AS instance_id, ci.angle, ci.session_id, ci.is_duplicate_of,
                       MIN(cv.timestamp_ms) AS start_time, MAX(cv.timestamp_ms) AS end_time,
                       ci.fused_image_path
                FROM card_instances ci
                LEFT JOIN card_views cv ON cv.card_instance_id = ci.id
                WHERE ci.video_id = ?
                GROUP BY ci.id
                ORDER BY start_time ASC
                """,
                (video_id,),
            ).fetchall()

        instance_data = [dict(r) for r in instances]
        truth_path = _truth_path_for_video(video_row["source_path"])
        existing_truth = json.loads(truth_path.read_text()) if truth_path.exists() else None

        return templates.TemplateResponse(
            request, "labeling.html",
            {
                "video_id": video_id,
                "video_path": video_row["source_path"],
                "instances": instance_data,
                "truth": existing_truth,
                "truth_path": str(truth_path),
            },
        )

    @app.post("/label/{video_id}/save")
    async def label_save(video_id: int, request: Request):
        payload = await request.json()
        with storage._connect() as conn:
            video_row = conn.execute(
                "SELECT source_path FROM videos WHERE id = ?", (video_id,),
            ).fetchone()
            if video_row is None:
                return {"ok": False, "error": "video not found"}

        truth_path = _truth_path_for_video(video_row["source_path"])
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(json.dumps(payload, indent=2))
        return {"ok": True, "path": str(truth_path)}

    @app.post("/cards/{saved_card_id}/decision")
    def decide(saved_card_id: int, decision: str = Form(...), notes: str = Form("")):
        storage.set_review_decision(saved_card_id, decision, notes)
        return RedirectResponse("/", status_code=303)

    @app.get("/images/{saved_card_id}")
    def image(saved_card_id: int):
        for card in storage.list_saved_cards(review_state=None, include_duplicates=True):
            if card["id"] == saved_card_id:
                resolved = _resolve_existing_path(card["image_path"])
                if resolved is not None:
                    return FileResponse(str(resolved), media_type="image/jpeg")
        return RedirectResponse("/", status_code=303)
    
    @app.get("/fused_images/{instance_id}")
    def get_fused_image(instance_id: int):
        with storage._connect() as conn:
            row = conn.execute("SELECT fused_image_path FROM card_instances WHERE id = ?", (instance_id,)).fetchone()
            if row and row["fused_image_path"]:
                resolved = _resolve_existing_path(row["fused_image_path"])
                if resolved is not None:
                    return FileResponse(str(resolved), media_type="image/jpeg")
        return RedirectResponse("/", status_code=303)

    @app.get("/card_views/{view_id}")
    def get_card_view(view_id: int):
        with storage._connect() as conn:
            row = conn.execute(
                """
                SELECT cv.rectified_path, cv.card_instance_id, ci.fused_image_path
                FROM card_views cv
                JOIN card_instances ci ON ci.id = cv.card_instance_id
                WHERE cv.id = ?
                """,
                (view_id,),
            ).fetchone()
            if row:
                resolved_rectified = _resolve_existing_path(row["rectified_path"])
                if resolved_rectified is not None:
                    return FileResponse(str(resolved_rectified), media_type="image/jpeg")
                # Fallback: show fused image if canonical raw view is missing.
                resolved_fused = _resolve_existing_path(row["fused_image_path"])
                if resolved_fused is not None:
                    return FileResponse(str(resolved_fused), media_type="image/jpeg")
        return RedirectResponse("/", status_code=303)

    return app
