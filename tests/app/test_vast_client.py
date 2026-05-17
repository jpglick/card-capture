"""Tests for VastAIClient — mocks subprocess so no real API calls needed."""
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app.services.vast_client import VastAIClient, GPU_TYPE_QUERIES


def _mock_run(stdout_data):
    """Return a mock CompletedProcess with JSON stdout."""
    m = MagicMock()
    m.stdout = json.dumps(stdout_data)
    return m


def test_search_offers_returns_list():
    client = VastAIClient(api_key="test-key")
    offers = [{"id": 1, "dph_total": 0.5}, {"id": 2, "dph_total": 0.3}]
    with patch("subprocess.run", return_value=_mock_run(offers)) as mock_run:
        result = client.search_offers("RTX 4090")
    assert result == [{"id": 2, "dph_total": 0.3}, {"id": 1, "dph_total": 0.5}]
    call_args = mock_run.call_args[0][0]
    cmd_str = " ".join(call_args)
    assert "RTX_4090" in cmd_str or "RTX 4090" in cmd_str


def test_provision_returns_instance_dict():
    client = VastAIClient(api_key="test-key")
    instance = {"id": 42, "status": "created"}
    with patch("subprocess.run", return_value=_mock_run(instance)):
        result = client.provision(offer_id=1, template_id="pytorch/pytorch:latest")
    assert result["id"] == 42


def test_destroy_calls_vastai():
    client = VastAIClient(api_key="test-key")
    with patch("subprocess.run", return_value=_mock_run({"success": True})) as mock_run:
        client.destroy(instance_id=42)
    call_args = mock_run.call_args[0][0]
    assert "destroy" in call_args
    assert "42" in call_args


def test_get_instance_ip_found():
    client = VastAIClient(api_key="test-key")
    instances = [{"id": 42, "public_ipaddr": "1.2.3.4"}, {"id": 99, "public_ipaddr": "5.6.7.8"}]
    with patch("subprocess.run", return_value=_mock_run(instances)):
        ip = client.get_instance_ip(42)
    assert ip == "1.2.3.4"


def test_get_instance_ip_not_found():
    client = VastAIClient(api_key="test-key")
    with patch("subprocess.run", return_value=_mock_run([])):
        ip = client.get_instance_ip(999)
    assert ip is None


def test_gpu_type_queries_has_all_options():
    for key in ["RTX 4090", "Flagship", "RTX 5060 Ti"]:
        assert key in GPU_TYPE_QUERIES
