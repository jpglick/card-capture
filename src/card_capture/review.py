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

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, state: str = "pending"):
        cards = storage.list_saved_cards(review_state=None if state == "all" else state)
        # Augment with fused image/angle if available (storage implementation details)
        # For simplicity here, we assume list_saved_cards already returns these or we join.
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
    def fused_image(instance_id: int):
        # Implementation to retrieve fused path from DB
        return RedirectResponse("/", status_code=303)

    return app
