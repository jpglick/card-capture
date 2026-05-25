# tests/pipeline/test_flow_fused_branch.py
"""The flow uses fused_refine for the cuda detector and forwards downstream."""


def test_flow_imports_fused_refine():
    import pipeline.card_capture_flow as flow
    # fused_refine must be importable by the flow module.
    assert hasattr(flow, "fused_refine")
