from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .storage import Storage

def create_app(db_path: Path):
    app = FastAPI(title="Card Capture Review")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    storage = Storage(db_path)
    storage.initialize()

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, state: str = "pending"):
        cards = storage.list_saved_cards(review_state=None if state == "all" else state)
        
        # Join with card_instances to get fused_image_path and angle
        with storage._connect() as conn:
            for card in cards:
                row = conn.execute("""
                    SELECT ci.fused_image_path, ci.angle
                    FROM card_views cv
                    JOIN card_instances ci ON cv.card_instance_id = ci.id
                    WHERE cv.id = ?
                """, (card["detection_id"],)).fetchone()
                if row:
                    card["fused_image_path"] = row["fused_image_path"]
                    card["angle"] = row["angle"]
                else:
                    card["fused_image_path"] = None
                    card["angle"] = None
        
        return templates.TemplateResponse(
            request,
            "review.html",
            {"cards": cards, "state": state},
        )

    @app.post("/cards/{saved_card_id}/decision")
    def decide(saved_card_id: int, decision: str = Form(...), notes: str = Form("")):
        storage.set_review_decision(saved_card_id, decision, notes)
        return RedirectResponse("/", status_code=303)

    @app.get("/images/{saved_card_id}")
    def image(saved_card_id: int):
        for card in storage.list_saved_cards(review_state=None):
            if card["id"] == saved_card_id:
                return FileResponse(card["image_path"], media_type="image/jpeg")
        return RedirectResponse("/", status_code=303)

    @app.get("/fused_images/{instance_id}")
    def get_fused_image(instance_id: int):
        with storage._connect() as conn:
            row = conn.execute("SELECT fused_image_path FROM card_instances WHERE id = ?", (instance_id,)).fetchone()
            if row and row["fused_image_path"]:
                return FileResponse(row["fused_image_path"], media_type="image/jpeg")
        return RedirectResponse("/", status_code=303)

    return app
