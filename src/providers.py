"""
RoadRisk AI — External AI Provider Abstraction
===============================================
A single, uniform interface for every optional external AI service the
assistant can talk to (Google Gemini, Groq, Hugging Face Inference).

Design rules enforced by this module:

* Every credential is resolved through :func:`src.config.get_secret`
  (Streamlit secrets first, then environment variables) and is **never**
  stored on an instance, logged, or returned by any public function.
* :func:`provider_statuses` exposes only non-sensitive descriptors —
  whether a provider is *configured* and, when probed, whether it is
  *available*. It never reveals the key or any part of it.
* Every network call uses an explicit timeout and returns ``None`` (never
  raises) so callers can degrade gracefully to the offline grounded engine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config import get_secret

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "20"))

# Provider names in the order they are tried for a live answer.
PROVIDER_PREFERENCE: Tuple[str, ...] = ("gemini", "groq", "huggingface")

# Placeholder markers used in the shipped example files. A value containing any
# of these is treated as "not configured" so demo configs never trigger calls.
_PLACEHOLDER_MARKERS = (
    "your-", "your_", "changeme", "replace-me", "replace_me",
    "example", "placeholder", "xxxx", "<", ">",
)


def _is_real_secret(value: Optional[str]) -> bool:
    """True only when *value* looks like an actual credential, not a placeholder."""
    if not value:
        return False
    candidate = str(value).strip()
    if len(candidate) < 12:
        return False
    lowered = candidate.lower()
    return not any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


@dataclass
class ProviderStatus:
    """Non-sensitive description of a provider's configuration state."""

    name: str
    label: str
    kind: str
    configured: bool
    available: Optional[bool]
    model: Optional[str]
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Base interface ────────────────────────────────────────────────────────────

class LLMProvider:
    """Minimal contract every provider must satisfy."""

    name: str = "base"
    kind: str = "llm"
    api_key_env: str = ""
    model_env: str = ""
    default_model: str = ""
    #: Human-readable label used in the UI (no credentials involved).
    label: str = "LLM"

    # -- configuration --------------------------------------------------------

    def api_key(self) -> Optional[str]:
        """Resolve the credential. Never stored, never returned to callers."""
        return get_secret(self.api_key_env)

    @property
    def model(self) -> str:
        """Configured model id, falling back to the documented default."""
        configured = get_secret(self.model_env)
        if configured and str(configured).strip():
            return str(configured).strip()
        return self.default_model

    def is_configured(self) -> bool:
        return _is_real_secret(self.api_key())

    # -- status ---------------------------------------------------------------

    def status(self, probe: bool = False) -> ProviderStatus:
        """
        Describe this provider without exposing its credential.

        When *probe* is True a lightweight liveness call is made so the
        ``available`` flag reflects reality; otherwise it stays ``None``
        (configured but unverified).
        """
        configured = self.is_configured()
        if not configured:
            return ProviderStatus(
                name=self.name,
                label=self.label,
                kind=self.kind,
                configured=False,
                available=False,
                model=self.model,
                detail="Not configured — no credential present. Offline mode is used.",
            )

        if not probe:
            return ProviderStatus(
                name=self.name,
                label=self.label,
                kind=self.kind,
                configured=True,
                available=None,
                model=self.model,
                detail="Credential present; availability not yet verified.",
            )

        probe_result = self.health_check()
        return ProviderStatus(
            name=self.name,
            label=self.label,
            kind=self.kind,
            configured=True,
            available=probe_result.get("available"),
            model=self.model,
            detail=probe_result.get("detail", ""),
        )

    # -- overridable behaviour ------------------------------------------------

    def health_check(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Lightweight liveness probe. Returns ``{'available', 'detail'}``."""
        return {"available": None, "detail": "No health check implemented."}

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """Return the model's answer, or ``None`` on any failure."""
        raise NotImplementedError


# ── Google Gemini ─────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    name = "gemini"
    api_key_env = "GEMINI_API_KEY"
    model_env = "GEMINI_MODEL"
    #: Overridable via GEMINI_MODEL — model ids change often, so we never
    #: hard-assert availability; the health probe reports what the account sees.
    default_model = "gemini-2.0-flash"
    label = "Google Gemini"

    def _client(self):
        from google import genai  # provided by the `google-genai` package
        return genai.Client(api_key=self.api_key())

    def health_check(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self.is_configured():
            return {"available": False, "detail": "Not configured."}
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return {"available": False, "detail": "`google-genai` package is not installed."}
        try:
            client = self._client()
            names = [getattr(m, "name", "") for m in client.models.list()]
            matched = any(self.model in n for n in names)
            if matched:
                return {"available": True, "detail": f"Model '{self.model}' reachable."}
            return {
                "available": True,
                "detail": (
                    f"API reachable ({len(names)} models listed) but '{self.model}' "
                    "was not in the list — set GEMINI_MODEL to a model the key can access."
                ),
            }
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            logger.warning("Gemini health probe failed: %s", type(exc).__name__)
            return {"available": False, "detail": f"Probe failed ({type(exc).__name__})."}

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        if not self.is_configured():
            return None
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.warning("`google-genai` is not installed; skipping Gemini.")
            return None
        try:
            client = self._client()
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                top_p=0.8,
            )
            response = client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
            text = getattr(response, "text", None)
            return text.strip() if text else None
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("Gemini invocation failed (%s).", type(exc).__name__)
            return None


# ── Groq (OpenAI-compatible) ──────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    name = "groq"
    api_key_env = "GROQ_API_KEY"
    model_env = "GROQ_MODEL"
    default_model = "llama-3.3-70b-versatile"
    label = "Groq Cloud"

    CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODELS_URL = "https://api.groq.com/openai/v1/models"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key()}"}

    def health_check(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self.is_configured():
            return {"available": False, "detail": "Not configured."}
        try:
            resp = requests.get(
                self.MODELS_URL,
                headers=self._headers(),
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            if resp.status_code != 200:
                return {"available": False, "detail": f"HTTP {resp.status_code} from Groq."}
            ids = [m.get("id", "") for m in resp.json().get("data", [])]
            if self.model in ids:
                return {"available": True, "detail": f"Model '{self.model}' available."}
            return {
                "available": True,
                "detail": (
                    f"API reachable ({len(ids)} models) but '{self.model}' not listed — "
                    "set GROQ_MODEL to an available model."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groq health probe failed: %s", type(exc).__name__)
            return {"available": False, "detail": f"Probe failed ({type(exc).__name__})."}

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        if not self.is_configured():
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.8,
        }
        try:
            resp = requests.post(
                self.CHAT_URL,
                json=payload,
                headers=self._headers(),
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            choices = resp.json().get("choices") or []
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content")
            return content.strip() if content else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groq invocation failed (%s).", type(exc).__name__)
            return None


# ── Hugging Face Inference ────────────────────────────────────────────────────

class HuggingFaceProvider(LLMProvider):
    name = "huggingface"
    api_key_env = "HF_TOKEN"
    model_env = "HF_MODEL"
    default_model = "meta-llama/Llama-3.1-8B-Instruct"
    label = "Hugging Face"

    #: {model} is substituted with the configured model id.
    INFERENCE_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model}"
    MODEL_INFO_URL_TEMPLATE = "https://huggingface.co/api/models/{model}"

    def inference_url(self) -> str:
        template = get_secret("HF_INFERENCE_URL") or self.INFERENCE_URL_TEMPLATE
        return str(template).format(model=self.model)

    def health_check(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self.is_configured():
            return {"available": False, "detail": "Not configured."}
        url = self.MODEL_INFO_URL_TEMPLATE.format(model=self.model)
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key()}"},
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                return {"available": True, "detail": f"Model '{self.model}' is reachable."}
            if resp.status_code == 404:
                return {
                    "available": False,
                    "detail": f"Model '{self.model}' not found for this account — set HF_MODEL.",
                }
            if resp.status_code in (401, 403):
                return {"available": False, "detail": "Credential rejected by Hugging Face."}
            return {"available": False, "detail": f"HTTP {resp.status_code} from Hugging Face."}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hugging Face health probe failed: %s", type(exc).__name__)
            return {"available": False, "detail": f"Probe failed ({type(exc).__name__})."}

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        if not self.is_configured():
            return None
        payload = {
            "inputs": f"{system_instruction}\n\n{prompt}",
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }
        try:
            resp = requests.post(
                self.inference_url(),
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key()}"},
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                text = data[0].get("generated_text")
                return text.strip() if text else None
            if isinstance(data, dict):
                text = data.get("generated_text") or data.get("error")
                return str(text).strip() if text else None
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hugging Face invocation failed (%s).", type(exc).__name__)
            return None


# ── Registry & orchestration ──────────────────────────────────────────────────

def get_providers() -> Dict[str, LLMProvider]:
    """Return a fresh name → provider mapping (cheap; reads env at call time)."""
    providers: List[LLMProvider] = [GeminiProvider(), GroqProvider(), HuggingFaceProvider()]
    return {p.name: p for p in providers}


def provider_statuses(probe: bool = False) -> List[Dict[str, Any]]:
    """
    Public, secret-free summary of every provider. Safe to render in the UI.

    Includes a synthetic ``offline`` entry describing the always-available
    deterministic engine so the interface can show a complete picture.
    """
    providers = get_providers()
    statuses = [providers[name].status(probe=probe).as_dict() for name in PROVIDER_PREFERENCE]
    statuses.append(
        ProviderStatus(
            name="offline",
            label="Grounded Offline Engine",
            kind="engine",
            configured=True,
            available=True,
            model=None,
            detail="Deterministic BM25 retrieval + parameterized SQLite analytics. Always available.",
        ).as_dict()
    )
    return statuses


def get_active_llm_provider(preferred: Optional[str] = None) -> Optional[LLMProvider]:
    """First configured provider, honouring *preferred* if it is usable."""
    providers = get_providers()
    order: List[str] = []
    if preferred:
        order.append(preferred)
    order.extend(name for name in PROVIDER_PREFERENCE if name != preferred)

    for name in order:
        provider = providers.get(name)
        if provider and provider.is_configured():
            return provider
    return None


def generate_with_fallback(
    prompt: str,
    system_instruction: str,
    preferred: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try the configured providers in preference order.

    Returns ``(answer_text, provider_name)``; ``(None, None)`` when no provider
    is configured or every configured provider failed. Never raises.
    """
    active = get_active_llm_provider(preferred)
    if active is None:
        return None, None

    answer = active.generate(prompt, system_instruction, timeout=timeout)
    if answer:
        return answer, active.name

    providers = get_providers()
    for name in PROVIDER_PREFERENCE:
        if name == active.name:
            continue
        alternative = providers[name]
        if alternative.is_configured():
            answer = alternative.generate(prompt, system_instruction, timeout=timeout)
            if answer:
                return answer, alternative.name

    return None, None
