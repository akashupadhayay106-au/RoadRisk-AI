"""
RoadRisk AI — Provider Abstraction Regression Tests (Phase 11)
===============================================================
Covers the external-API provider layer introduced for Phase 3 of the upgrade:

* behaviour with NO credentials configured (offline degradation),
* behaviour with a configured provider (HTTP mocked — no real network calls),
* graceful fallback between providers,
* and the hard guarantee that status descriptors never leak a credential value.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src import providers
from src.ai_assistant import ask_roadrisk_assistant

# A syntactically valid but fake credential. It contains none of the
# placeholder markers, so the provider layer must treat it as "configured".
FAKE_GEMINI_KEY = "AIzaSyFAKEfakeFAKEfakeFAKEfake1234"
FAKE_GROQ_KEY = "gsk_FAKEfakeFAKEfakeFAKEfake1234"
FAKE_HF_TOKEN = "hf_FAKEfakeFAKEfakeFAKEfake1234"

SECRET_ENV_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "HF_TOKEN", "LLM_PROVIDER")


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure no real credentials leak in from the developer's environment."""
    for var in SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ── No-credential (offline) behaviour ─────────────────────────────────────────

def test_no_keys_means_no_active_provider(clean_env):
    assert providers.get_active_llm_provider() is None


def test_no_keys_generate_returns_none(clean_env):
    answer, provider_name = providers.generate_with_fallback("q", "sys")
    assert answer is None
    assert provider_name is None


def test_assistant_uses_offline_mode_without_keys(clean_env):
    result = ask_roadrisk_assistant("What are the top 5 states by accidents?")
    assert result["mode"] == "offline"
    assert result["is_blocked"] is False


def test_placeholder_values_are_treated_as_unconfigured(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "your-gemini-api-key-here")
    providers_map = providers.get_providers()
    assert providers_map["gemini"].is_configured() is False


# ── Status descriptors must never leak credentials ────────────────────────────

def test_status_descriptors_are_secret_free(clean_env):
    clean_env.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    clean_env.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    clean_env.setenv("HF_TOKEN", FAKE_HF_TOKEN)

    statuses = providers.provider_statuses(probe=False)
    serialized = repr(statuses)

    for secret in (FAKE_GEMINI_KEY, FAKE_GROQ_KEY, FAKE_HF_TOKEN):
        assert secret not in serialized

    allowed_fields = {"name", "label", "kind", "configured", "available", "model", "detail"}
    for entry in statuses:
        assert set(entry.keys()) == allowed_fields


def test_status_reports_configuration_booleans(clean_env):
    clean_env.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    by_name = {row["name"]: row for row in providers.provider_statuses(probe=False)}

    assert by_name["groq"]["configured"] is True
    assert by_name["gemini"]["configured"] is False
    assert by_name["huggingface"]["configured"] is False
    # The deterministic engine is always present and always usable.
    assert by_name["offline"]["configured"] is True
    assert by_name["offline"]["available"] is True


def test_offline_engine_is_always_listed(clean_env):
    names = [row["name"] for row in providers.provider_statuses(probe=False)]
    assert names[-1] == "offline"
    assert "gemini" in names and "groq" in names and "huggingface" in names


# ── Groq: OpenAI-compatible contract (mocked) ─────────────────────────────────

def test_groq_generate_sends_bearer_auth_and_openai_payload(clean_env, monkeypatch):
    clean_env.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return _FakeResponse(200, {"choices": [{"message": {"content": "  grounded answer  "}}]})

    monkeypatch.setattr(providers.requests, "post", fake_post)

    provider = providers.get_providers()["groq"]
    answer = provider.generate("What is X?", "SYSTEM RULES")

    assert answer == "grounded answer"
    assert captured["url"] == providers.GroqProvider.CHAT_URL
    assert captured["headers"]["Authorization"] == f"Bearer {FAKE_GROQ_KEY}"
    assert captured["json"]["messages"][0] == {"role": "system", "content": "SYSTEM RULES"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "What is X?"}
    assert captured["timeout"] is not None


def test_groq_generate_returns_none_on_http_error(clean_env, monkeypatch):
    clean_env.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    monkeypatch.setattr(
        providers.requests, "post", lambda *a, **k: _FakeResponse(500, {}, text="boom")
    )
    assert providers.get_providers()["groq"].generate("q", "s") is None


def test_groq_generate_returns_none_when_unconfigured(clean_env):
    assert providers.get_providers()["groq"].generate("q", "s") is None


# ── Fallback chain ────────────────────────────────────────────────────────────

def test_falls_back_to_next_provider_when_first_fails(clean_env, monkeypatch):
    clean_env.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    clean_env.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)

    monkeypatch.setattr(
        providers.GeminiProvider, "generate", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        providers.GroqProvider, "generate", lambda self, *a, **k: "groq answer"
    )

    answer, provider_name = providers.generate_with_fallback("q", "sys")
    assert answer == "groq answer"
    assert provider_name == "groq"


def test_preferred_provider_is_honoured(clean_env, monkeypatch):
    clean_env.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    clean_env.setenv("HF_TOKEN", FAKE_HF_TOKEN)
    clean_env.setenv("LLM_PROVIDER", "huggingface")

    active = providers.get_active_llm_provider(preferred="huggingface")
    assert active is not None and active.name == "huggingface"


def test_assistant_degrades_to_offline_when_provider_errors(clean_env, monkeypatch):
    clean_env.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    monkeypatch.setattr(
        providers.GeminiProvider, "generate", lambda self, *a, **k: None
    )
    result = ask_roadrisk_assistant("What are the top 5 states by accidents?")
    assert result["mode"] == "offline"


def test_assistant_uses_live_provider_when_available(clean_env, monkeypatch):
    clean_env.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    monkeypatch.setattr(
        providers.GeminiProvider, "generate", lambda self, *a, **k: "live grounded answer"
    )
    result = ask_roadrisk_assistant("What are the top 5 states by accidents?")
    assert result["mode"] == "gemini"
    assert result["response"] == "live grounded answer"


# ── Health probes ─────────────────────────────────────────────────────────────

def test_gemini_health_check_reports_unconfigured(clean_env):
    assert providers.get_providers()["gemini"].health_check()["available"] is False


def test_huggingface_health_check_detects_missing_model(clean_env, monkeypatch):
    clean_env.setenv("HF_TOKEN", FAKE_HF_TOKEN)
    monkeypatch.setattr(
        providers.requests,
        "get",
        lambda *a, **k: _FakeResponse(404, {}, text="not found"),
    )
    result = providers.get_providers()["huggingface"].health_check()
    assert result["available"] is False
    assert "not found" in result["detail"].lower() or "404" in result["detail"]


def test_huggingface_health_check_flags_rejected_credential(clean_env, monkeypatch):
    clean_env.setenv("HF_TOKEN", FAKE_HF_TOKEN)
    monkeypatch.setattr(providers.requests, "get", lambda *a, **k: _FakeResponse(401, {}))
    result = providers.get_providers()["huggingface"].health_check()
    assert result["available"] is False
    assert "rejected" in result["detail"].lower()


def test_probe_marks_configured_provider_as_available(clean_env, monkeypatch):
    clean_env.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    monkeypatch.setattr(
        providers.GeminiProvider,
        "health_check",
        lambda self, timeout=None: {"available": True, "detail": "Model reachable."},
    )
    row = next(r for r in providers.provider_statuses(probe=True) if r["name"] == "gemini")
    assert row["configured"] is True
    assert row["available"] is True
