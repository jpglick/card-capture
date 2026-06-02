from __future__ import annotations

import sys
import types

from card_capture.stages import refine


def test_get_shared_embedder_is_singleton(monkeypatch):
    calls = {"count": 0}

    class _StubEmbedder:
        def __init__(self, variant: str) -> None:
            calls["count"] += 1
            self.variant = variant

    stub_module = types.SimpleNamespace(DinoEmbedder=_StubEmbedder)
    monkeypatch.setattr(refine, "_EMBEDDER_SINGLETON", None)
    monkeypatch.setitem(sys.modules, "card_capture.ml.models.dino_embedder", stub_module)

    e1 = refine.get_shared_embedder()
    e2 = refine.get_shared_embedder()
    assert e1 is e2
    assert calls["count"] == 1
