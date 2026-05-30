"""Stage 8: Front/Back Resolution.

V5.5 in-process port of V4 ``pipeline/steps/resolve.py``. Groups tracks
into temporal sessions, uses the F/B classifier (via ``predict_array``)
and textiness heuristic to resolve which side of the card each track
represents, and performs identity resolution to group related tracks.

Three substitutions:
- ``cv2.imread(best_path)`` -> ``refined_track['best_canonical_image']``
- ``FBPredictor.predict(path)`` -> ``FBPredictor.predict_array(ndarray)``
- Output for fuse stage is ``state['prepared_tracks']`` (flat list of tracks).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from card_capture.deduplicator import VisualDeduplicator
from card_capture.pipeline_utils import _resolve_session_tracks


_PREDICTOR_SINGLETON: object = None


def _get_fb_predictor():
    global _PREDICTOR_SINGLETON
    if _PREDICTOR_SINGLETON is not None:
        return _PREDICTOR_SINGLETON
    try:
        from card_capture.ml.inference.fb_predict import FBPredictor
        from pathlib import Path
        ckpt = Path(__file__).parent.parent.parent.parent.parent / "models" / "presence_classifier.pt"
        if not ckpt.exists():
             # Fallback to current dir if models not found at root (dev convenience)
             ckpt = Path("models/presence_classifier.pt")
        
        if FBPredictor.is_available(ckpt):
            _PREDICTOR_SINGLETON = FBPredictor(checkpoint_path=ckpt)
        else:
            _PREDICTOR_SINGLETON = None
    except Exception:
        _PREDICTOR_SINGLETON = None
    return _PREDICTOR_SINGLETON


def run(state: dict, *, telemetry) -> None:
    config = state["request"].config
    refined_tracks = state.get("refined_tracks") or []
    use_classifier = bool(config.get("use_fb_classifier", True))

    # 1. Group by session_id
    by_session: Dict[int, List[Dict[str, Any]]] = {}
    for track in refined_tracks:
        sid = int(track.get("session_id", 0))
        by_session.setdefault(sid, []).append(track)

    resolved_sessions: List[Dict[str, Any]] = []
    prepared_tracks: List[Dict[str, Any]] = []

    predictor = _get_fb_predictor() if use_classifier else None
    deduplicator = VisualDeduplicator()

    for sid, tracks in by_session.items():
        # 2. Front/Back prediction for each track's best image
        for track in tracks:
            img = track.get("best_canonical_image")
            if img is not None and predictor is not None:
                try:
                    label, conf = predictor.predict_array(img)
                    # map to [front_prob, back_prob] for compatibility with heuristic
                    if label == "front":
                        track["fb_probs"] = [conf, 1.0 - conf]
                    else:
                        track["fb_probs"] = [1.0 - conf, conf]
                except Exception as exc:
                    telemetry.resource_sample(
                        {"event": "fb_predict_failed", "error": repr(exc)}
                    )

        # 3. Resolve identities within session
        # Heuristic/Classifier-aware resolution
        _resolve_session_tracks(tracks, deduplicator)

        resolved_sessions.append({"session_id": sid, "tracks": tracks})
        prepared_tracks.extend(tracks)

    state["resolved_sessions"] = resolved_sessions
    state["prepared_tracks"] = prepared_tracks
