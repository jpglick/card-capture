"""In-process pipeline stages, organized as vertical slices.

Each slice package exposes a single `run(state, *, telemetry)` callable from its
__init__ and owns its algorithm modules. Model loading/decode lifecycle belong to
the runtime, not the slices.
"""
