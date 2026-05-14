"""Tests for MetricResult JSON round-trip serialization."""
from harness.metrics.types import MetricResult


def test_metric_result_roundtrip():
    mr = MetricResult(value=0.75)
    assert MetricResult.model_validate_json(mr.model_dump_json()) == mr


def test_compound_metric_result_roundtrip():
    mr = MetricResult(breakdown={"ari": 0.8, "pair_f1": 0.7})
    assert MetricResult.model_validate_json(mr.model_dump_json()) == mr
