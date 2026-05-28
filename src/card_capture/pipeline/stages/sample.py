"""Stage 1: Adaptive Presence Sampler.

The key V5.5 change: we open the input video ONCE and stash the open handle
in state so detect/refine can reuse it.
"""
from __future__ import annotations

from pathlib import Path
from card_capture.sampler import StrideSampler


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    video_path = request.input_video.replace("artifact://local/", "")
    telemetry.resource_sample({"event": "decode_open", "path": video_path})
    
    # We use StrideSampler directly for now. In a full implementation, we'd
    # select the backend based on request.config.
    sampler = StrideSampler(video_path=Path(video_path))
    
    # Consume the generator to get all frames
    frames = list(sampler.sample())
    
    state["sampler"] = sampler
    state["sampled_frames"] = frames  # list of FrameSample
    state["video_path"] = video_path
