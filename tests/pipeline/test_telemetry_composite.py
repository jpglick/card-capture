from card_capture.pipeline.telemetry import CompositeTelemetry, InMemoryTelemetry

def test_composite_telemetry_broadcasts_to_all():
    t1 = InMemoryTelemetry()
    t2 = InMemoryTelemetry()
    composite = CompositeTelemetry([t1, t2])
    
    composite.stage_started("detect", {"foo": "bar"})
    composite.stage_finished("detect", 100, {"frames": 10})
    composite.progress("detect", 50, "Halfway there")
    composite.resource_sample({"gpu_mem_mb": 4096})
    composite.contract_violation("no_cards", {"detail": "Found 0 cards"})
    
    assert len(t1.events) == 5
    assert t1.events[0].kind == "stage_started"
    assert t1.events[1].kind == "stage_finished"
    assert t1.events[2].kind == "progress"
    assert t1.events[3].kind == "resource_sample"
    assert t1.events[4].kind == "contract_violation"
    
    assert t1.events[0].payload == {"stage": "detect", "foo": "bar"}
    assert t1.events[1].payload == {"stage": "detect", "elapsed_ms": 100, "frames": 10}
    assert t1.events[2].payload == {"stage": "detect", "pct": 50, "detail": "Halfway there"}
    assert t1.events[3].payload == {"gpu_mem_mb": 4096}
    assert t1.events[4].payload == {"code": "no_cards", "detail": "Found 0 cards"}
    
    assert len(t2.events) == 5
