"""Tests for VastAIClient — mocks httpx so no real API calls are made."""
from unittest.mock import MagicMock, patch

import pytest

from app.services.vast_client import VastAIClient, GPU_TYPE_QUERIES


def _mock_response(data: object, status_code: int = 200):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = data
    m.raise_for_status = MagicMock()
    return m


def test_search_offers_returns_sorted_list():
    client = VastAIClient(api_key="test-key")
    offers = [
        {"id": 1, "dph_total": 0.8, "gpu_name": "RTX 4090"},
        {"id": 2, "dph_total": 0.4, "gpu_name": "RTX 4090"},
    ]
    # search_offers uses POST /bundles/ with a JSON body
    with patch("httpx.post", return_value=_mock_response({"offers": offers})):
        result = client.search_offers("RTX 4090")
    assert result[0]["dph_total"] == 0.4


def test_provision_returns_instance_dict():
    client = VastAIClient(api_key="test-key")
    with patch("httpx.put", return_value=_mock_response({"new_contract": 42, "success": True})):
        result = client.provision(offer_id=1, template_id="ghcr.io/jpglick/card-capture-cuda:latest")
    assert result["id"] == 42


def test_destroy_calls_api():
    client = VastAIClient(api_key="test-key")
    with patch("httpx.delete", return_value=_mock_response({})) as mock_del:
        client.destroy(instance_id=42)
    assert mock_del.called
    assert "42" in mock_del.call_args[0][0]


def test_get_instance_ip_found():
    client = VastAIClient(api_key="test-key")
    instances = [{"id": 42, "public_ipaddr": "1.2.3.4"}, {"id": 99, "public_ipaddr": "5.6.7.8"}]
    with patch("httpx.get", return_value=_mock_response({"instances": instances})):
        ip = client.get_instance_ip(42)
    assert ip == "1.2.3.4"


def test_get_instance_ip_not_found():
    client = VastAIClient(api_key="test-key")
    with patch("httpx.get", return_value=_mock_response({"instances": []})):
        ip = client.get_instance_ip(999)
    assert ip is None


def test_gpu_type_queries_has_all_options():
    for key in ["RTX 4090", "Flagship", "RTX 5060 Ti"]:
        assert key in GPU_TYPE_QUERIES
