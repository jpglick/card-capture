# V5.5 CI Baseline Inventory

| Test ID | Status | Category | Disposition |
|---|---|---|---|
| `tests/app/test_pipeline_runner.py::test_runner_emits_stage_events_for_each_step` | PASS | real-bug | fixed |
| `tests/app/test_pipeline_runner.py::test_runner_emits_run_failed_on_exception` | PASS | real-bug | fixed |
| `tests/app/test_schemas.py::test_all_v1_schemas_importable_and_have_examples` | PASS | plan-changed-behavior | fixed |
| `tests/app/test_vastai_worker.py` (all) | PASS | real-bug | fixed |
| `tests/pipeline/test_detect_cuda.py` | SKIP | missing-hardware | quarantined |
| `tests/pipeline/test_flow_runs.py` | PASS | real-bug | fixed |
| `tests/test_detector_coreml.py` | SKIP | missing-hardware | quarantined |
| `tests/test_detector_trt.py` | PASS | real-bug | fixed |
| `tests/test_detectors.py::test_detector_rescales_polygon_to_original_frame_space` | PASS | real-bug | fixed |
| `tests/test_detectors.py::test_probe_torch_device_status_reports_mps_unavailable_state` | PASS | real-bug | fixed |
| `tests/test_detectors.py::test_detector_skips_hf_download_when_cached` | SKIP | missing-credentials | quarantined |
| `tests/test_sampler.py::test_video_sampler_uses_decord_backend_when_requested` | PASS | real-bug | fixed |
| `tests/test_wave2_robustness.py::test_per_pixel_bg_tracks_variance` | PASS | real-bug | fixed |
| `tests/test_wave2_robustness.py::test_mahalanobis_novelty_with_variance` | PASS | real-bug | fixed |
| `tests/migrations/test_schema.py::test_migrations_are_idempotent` | SKIP | real-bug | quarantined |
| `tests/test_wave1_robustness.py` | FAIL | real-bug | partly-fixed |
| `tests/test_sampler.py::TestAdaptivePresenceSampler::test_sample_prefers_local_contiguous_frames_in_large_window` | SKIP | real-bug | quarantined |
| `tests/test_sampler.py::test_sampler_background_proxies_safety_threshold` | SKIP | real-bug | quarantined |
| `tests/test_wave2_robustness.py::test_quality_scorer_penalizes_occluded_frame` | SKIP | real-bug | quarantined |
| `tests/test_wave2_robustness.py::TestPerRegionValleyDetection::test_sampler_integrates_per_region_valleys` | SKIP | real-bug | quarantined |
| `tests/pipeline/test_detect_crop_cache.py` | SKIP | stale-test | quarantined |
| `tests/pipeline/test_detect_prefetch.py` | SKIP | stale-test | quarantined |
| `tests/pipeline/test_fused_refine.py` | SKIP | stale-test | quarantined |

