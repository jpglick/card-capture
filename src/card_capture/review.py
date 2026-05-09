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
