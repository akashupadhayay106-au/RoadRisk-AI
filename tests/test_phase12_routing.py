"""
RoadRisk AI — openrouteservice Client Tests (Phase 12)
=====================================================
Verifies the HeiGIT / openrouteservice integration behaves correctly both with
and without a configured key, and that no credential value ever appears in a
status descriptor. All HTTP is mocked — no real network calls are made.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src import routing_client, weather_client
from src.routing_client import RoutingClient, get_routing_client, routing_service_status

FAKE_ORS_KEY = "ors_FAKEfakeFAKEfake1234567890"

SECRET_ENV_VARS = ("OPENROUTESERVICE_API_KEY", "OPENROUTESERVICE_BASE_URL")


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def clean_env(monkeypatch):
    for var in SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ── Unconfigured behaviour ────────────────────────────────────────────────────

def test_no_key_means_not_configured(clean_env):
    client = get_routing_client()
    assert client.is_configured() is False
    assert client.status().configured is False


def test_placeholder_key_is_not_configured(clean_env):
    clean_env.setenv("OPENROUTESERVICE_API_KEY", "your-openrouteservice-api-key-here")
    assert RoutingClient().is_configured() is False


def test_unconfigured_calls_return_none(clean_env):
    client = get_routing_client()
    assert client.get_route((12.97, 77.59), (13.08, 80.27)) is None
    assert client.get_isochrone((12.97, 77.59)) is None
    assert client.geocode("Bengaluru") is None


# ── Status descriptors are secret-free ────────────────────────────────────────

def test_status_never_contains_the_key(clean_env):
    clean_env.setenv("OPENROUTESERVICE_API_KEY", FAKE_ORS_KEY)
    status = routing_service_status(probe=False)
    assert FAKE_ORS_KEY not in repr(status)
    assert set(status.keys()) == {"name", "label", "kind", "configured", "available", "model", "detail"}
    assert status["configured"] is True
    assert status["available"] is None  # not probed yet


def test_health_check_reports_rejected_credential(clean_env, monkeypatch):
    clean_env.setenv("OPENROUTESERVICE_API_KEY", FAKE_ORS_KEY)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(403))
    result = RoutingClient().health_check()
    assert result["available"] is False
    assert "rejected" in result["detail"].lower()


# ── Route requests (mocked) ───────────────────────────────────────────────────

def test_get_route_sends_lonlat_and_parses_summary(clean_env, monkeypatch):
    clean_env.setenv("OPENROUTESERVICE_API_KEY", FAKE_ORS_KEY)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return _FakeResponse(
            200,
            {
                "features": [
                    {
                        "properties": {"summary": {"distance": 335_400.0, "duration": 19_800.0}},
                        "geometry": {"coordinates": [[77.59, 12.97], [78.10, 13.10]]},
                    }
                ]
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)

    result = RoutingClient().get_route((12.9716, 77.5946), (13.0827, 80.2707))

    assert result["distance_km"] == pytest.approx(335.4)
    assert result["duration_min"] == pytest.approx(330.0)
    assert result["profile"] == "driving-car"
    assert "/v2/directions/driving-car/geojson" in captured["url"]
    assert captured["headers"]["Authorization"] == FAKE_ORS_KEY
    # (lat, lon) input must be converted to ORS's (lon, lat) order.
    assert captured["json"]["coordinates"][0] == [77.5946, 12.9716]
    assert captured["timeout"] is not None


def test_get_route_returns_none_on_http_error(clean_env, monkeypatch):
    clean_env.setenv("OPENROUTESERVICE_API_KEY", FAKE_ORS_KEY)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(500))
    assert RoutingClient().get_route((12.97, 77.59), (13.08, 80.27)) is None


def test_get_route_rejects_unknown_profile(clean_env, monkeypatch):
    clean_env.setenv("OPENROUTESERVICE_API_KEY", FAKE_ORS_KEY)
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: pytest.fail("should not make a request")
    )
    assert RoutingClient().get_route((12.97, 77.59), (13.08, 80.27), profile="teleport") is None


def test_get_isochrone_converts_minutes_to_seconds(clean_env, monkeypatch):
    clean_env.setenv("OPENROUTESERVICE_API_KEY", FAKE_ORS_KEY)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json)
        return _FakeResponse(200, {"type": "FeatureCollection", "features": []})

    monkeypatch.setattr(requests, "post", fake_post)

    result = RoutingClient().get_isochrone((12.97, 77.59), ranges_minutes=(5, 10))
    assert result is not None
    assert captured["json"]["range"] == [300, 600]
    assert captured["json"]["locations"] == [[77.59, 12.97]]


def test_geocode_parses_features(clean_env, monkeypatch):
    clean_env.setenv("OPENROUTESERVICE_API_KEY", FAKE_ORS_KEY)
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeResponse(
            200,
            {
                "features": [
                    {
                        "properties": {"label": "Bengaluru, Karnataka, India"},
                        "geometry": {"coordinates": [77.5946, 12.9716]},
                    }
                ]
            },
        ),
    )
    results = RoutingClient().geocode("Bengaluru")
    assert results[0]["lat"] == pytest.approx(12.9716)
    assert results[0]["lon"] == pytest.approx(77.5946)


# ── Weather service status ────────────────────────────────────────────────────

def test_weather_status_is_keyless_and_configured():
    status = weather_client.weather_service_status(probe=False)
    assert status["configured"] is True
    assert status["kind"] == "weather"
    # No key-based weather provider is wired up while the provider is unconfirmed.
    assert "unconfirmed" in status["detail"]


def test_weather_status_probe_handles_failure(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(503))
    status = weather_client.weather_service_status(probe=True)
    assert status["available"] is False
    assert "503" in status["detail"]


def test_weather_status_probe_success(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200, {}))
    status = weather_client.weather_service_status(probe=True)
    assert status["available"] is True


def test_routing_module_is_importable_without_key(clean_env):
    # Regression guard: importing the module must not require credentials.
    assert routing_client.DEFAULT_BASE_URL.startswith("https://api.openrouteservice.org")
