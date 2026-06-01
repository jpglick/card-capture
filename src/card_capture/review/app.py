import json
from pathlib import Path
from typing import Optional

from card_capture.data.sql_queries import (
    REVIEW_CANONICAL_VIEWS,
    REVIEW_CARD_JOIN_BY_DETECTION,
    REVIEW_CARD_VIEW_BY_ID,
    REVIEW_FUSED_IMAGE_BY_INSTANCE,
    REVIEW_LABEL_INSTANCES_BY_VIDEO,
    REVIEW_TIMELINE_EVENTS_BASE,
    REVIEW_TIMELINE_EVENTS_ORDER,
    REVIEW_TIMELINE_INSTANCES_BASE,
    REVIEW_TIMELINE_INSTANCES_ORDER,
    REVIEW_VIDEO_BY_ID,
    REVIEW_VIDEO_COUNT,
    REVIEW_VIDEOS_LIST,
    REVIEW_VIDEO_SOURCE_BY_ID,
)
from card_capture.stages.store.storage import Storage

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
                row = conn.execute(REVIEW_CARD_JOIN_BY_DETECTION, (card["detection_id"],)).fetchone()
                
                if row:
                    card["instance_id"] = row["instance_id"]
                    card["fused_image_path"] = row["fused_image_path"]
                    card["angle"] = row["angle"]
                    card["session_id"] = row["session_id"]
                    canonical_rows = conn.execute(REVIEW_CANONICAL_VIEWS, (row["instance_id"],)).fetchall()
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
            events = conn.execute(REVIEW_TIMELINE_EVENTS_BASE + REVIEW_TIMELINE_EVENTS_ORDER).fetchall()
            
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
            events_query = REVIEW_TIMELINE_EVENTS_BASE
            params = []
            if video is not None:
                events_query += " WHERE video_id = ?"
                params.append(video)
            events_query += REVIEW_TIMELINE_EVENTS_ORDER
            events = conn.execute(events_query, params).fetchall()
            
            pipeline_events = []
            for e in events:
                pipeline_events.append({
                    "frame_index": e["frame_index"],
                    "timestamp_ms": e["timestamp_ms"],
                    "event_type": e["event_type"],
                    "data": json.loads(e["data_json"]) if e["data_json"] else {}
                })

            instances_query = REVIEW_TIMELINE_INSTANCES_BASE
            if video is not None:
                instances_query += " WHERE ci.video_id = ?"
            instances_query += REVIEW_TIMELINE_INSTANCES_ORDER
            
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
            video_row = conn.execute(REVIEW_VIDEO_BY_ID, (video_id,)).fetchone()
            if video_row is None:
                return HTMLResponse(f"video {video_id} not found", status_code=404)

            instances = conn.execute(REVIEW_LABEL_INSTANCES_BY_VIDEO, (video_id,)).fetchall()

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
            video_row = conn.execute(REVIEW_VIDEO_SOURCE_BY_ID, (video_id,)).fetchone()
            if video_row is None:
                return {"ok": False, "error": "video not found"}

        truth_path = _truth_path_for_video(video_row["source_path"])
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(json.dumps(payload, indent=2))
        return {"ok": True, "path": str(truth_path)}

    @app.get("/setup", response_class=HTMLResponse)
    def setup(request: Request):
        with storage._connect() as conn:
            video_count = conn.execute(REVIEW_VIDEO_COUNT).fetchone()[0]
            videos = conn.execute(REVIEW_VIDEOS_LIST).fetchall()

        corpus_root = Path("tests/fixtures/golden_corpus")
        truth_count = len(list(corpus_root.glob("*/*.truth.json"))) if corpus_root.exists() else 0
        reports_dir = Path("reports")
        any_report = any(
            p for p in reports_dir.glob("*.json") if p.name != ".gitkeep"
        ) if reports_dir.exists() else False
        has_baseline = Path("reports/baseline_v3.json").exists()

        return templates.TemplateResponse(
            request, "setup.html",
            {
                "video_count": video_count,
                "videos": [
                    {"id": r["id"], "name": Path(r["source_path"]).name}
                    for r in videos
                ],
                "truth_count": truth_count,
                "any_report": any_report,
                "has_baseline": has_baseline,
            },
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
            row = conn.execute(REVIEW_FUSED_IMAGE_BY_INSTANCE, (instance_id,)).fetchone()
            if row and row["fused_image_path"]:
                resolved = _resolve_existing_path(row["fused_image_path"])
                if resolved is not None:
                    return FileResponse(str(resolved), media_type="image/jpeg")
        return RedirectResponse("/", status_code=303)

    @app.get("/card_views/{view_id}")
    def get_card_view(view_id: int):
        with storage._connect() as conn:
            row = conn.execute(REVIEW_CARD_VIEW_BY_ID, (view_id,)).fetchone()
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
