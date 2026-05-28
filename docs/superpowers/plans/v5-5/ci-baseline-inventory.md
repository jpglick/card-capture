# V5.5 CI Baseline Inventory

| Test ID | Status | Category | Disposition |
|---|---|---|---|
| `tests/app/test_pipeline_runner.py::test_runner_emits_stage_events_for_each_step` | FAIL | real-bug | fix (stale API: missing `video_id`) |
| `tests/app/test_pipeline_runner.py::test_runner_emits_run_failed_on_exception` | FAIL | real-bug | fix (stale API: missing `video_id`) |
| `tests/app/test_schemas.py::test_all_v1_schemas_importable_and_have_examples` | FAIL | plan-changed-behavior | fix (add examples to Pydantic models) |
| `tests/app/test_vastai_worker.py` (all) | FAIL | real-bug | fix (asyncio loop issue in tests) |
| `tests/pipeline/test_detect_cuda.py` | FAIL | missing-hardware | quarantine (requires CUDA) |
| `tests/pipeline/test_flow_runs.py` | FAIL | real-bug | fix (PYTHONPATH/Metaflow module resolution) |
| `tests/test_detector_coreml.py` | ERROR | real-bug | fix (`TorchDeviceStatus` init mismatch) |
| `tests/test_detector_trt.py` | FAIL | real-bug | fix (YOLO mock `task` param) |
| `tests/test_detectors.py::test_detector_rescales_polygon_to_original_frame_space` | FAIL | real-bug | fix (`_device` attribute missing) |
| `tests/test_detectors.py::test_probe_torch_device_status_reports_mps_unavailable_state` | FAIL | real-bug | fix (`mps_built` attribute missing) |
| `tests/test_detectors.py::test_detector_skips_hf_download_when_cached` | FAIL | missing-credentials | quarantine (HF unauthorized) |
| `tests/test_sampler.py::test_video_sampler_uses_decord_backend_when_requested` | FAIL | real-bug | fix (`pixel_format` mock mismatch) |
| `tests/test_wave2_robustness.py::test_per_pixel_bg_tracks_variance` | FAIL | real-bug | fix (`variance_gray` missing) |
| `tests/test_wave2_robustness.py::test_mahalanobis_novelty_with_variance` | FAIL | real-bug | fix (`use_variance` param missing) |
| `tests/migrations/test_schema.py::test_migrations_are_idempotent` | FAIL | real-bug | fix (known bug) |
| `tests/test_wave1_robustness.py` | FAIL | real-bug | fix (ByteTrack adapter / numpy attr mismatch) |
| `tests/test_wave2_robustness.py` | FAIL | real-bug | fix (several) |
| `tests/pipeline/test_path_equivalence.py` | FAIL | real-bug | fix (known Metaflow bug) |
