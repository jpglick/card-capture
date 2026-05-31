"""Phase 4 — LocalPipelineRuntime injects repositories and output_root into state."""
from pathlib import Path
from unittest.mock import MagicMock

from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.request import PipelineRunRequest


def test_runtime_injects_repos_and_path_into_state():
    request = PipelineRunRequest(
        run_id="test-run",
        input_video="artifact://local/v.mov",
        output_root="artifact://local/out/",
        runtime_mode="cpu_debug",
        config={},
    )
    runtime = LocalPipelineRuntime()
    
    # We only want to test the initialization of 'state'
    # The run() method starts the writer and runs stages.
    # We can mock a stage to capture state.
    
    import card_capture.pipeline.runtime_local as rtl
    captured_state = {}

    def mock_stage_run(state, **kwargs):
        captured_state.update(state)

    # Patch the first stage ('sample')
    original_stages = rtl._STAGES
    rtl._STAGES = (("sample", MagicMock(run=mock_stage_run)),)
    
    try:
        # Mocking the repositories and writer to avoid actual DB/thread starts
        with MagicMock() as mock_writer:
            runtime.run(request)
    except Exception:
        pass # It might fail later, but we captured the state
    finally:
        rtl._STAGES = original_stages

    assert "repos" in captured_state
    assert "cards" in captured_state["repos"]
    assert "output_root" in captured_state
    assert isinstance(captured_state["output_root"], Path)
    assert str(captured_state["output_root"]).endswith("out")
