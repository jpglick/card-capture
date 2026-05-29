"""In-process pipeline stages.

Each module exposes a single `run()` function that takes a stage-specific
input and `GpuSession | None` and returns a stage-specific output. Stages
do not own model loading or decode lifecycle — those belong to the runtime.

Stages map 1:1 onto pipeline/steps/*.py in V4; this is the deliberate
re-homing: same algorithmic work, no subprocess boundary, no datastore
pickling between stages.
"""
