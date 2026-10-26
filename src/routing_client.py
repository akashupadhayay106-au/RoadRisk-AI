"""
RoadRisk AI — HeiGIT / openrouteservice Routing & Geospatial Client
===================================================================
Optional integration with **openrouteservice (ORS)** — the public routing
service operated by HeiGIT (Heidelberg Institute for Geoinformation Technology).

Capabilities exposed here:

* :meth:`RoutingClient.get_route` — real road-network distance/duration between
  two coordinates (used to turn map clicks into travel-time context).
* :meth:`RoutingClient.get_isochrone` — reachable-area polygons around a point,
  useful for "how far can you get from this hotspot in N minutes".
* :meth:`RoutingClient.geocode` — place-name → coordinates (Pelias).

Guarantees:

* The API key is read through :func:`src.config.get_secret` and is **never**
  stored on the instance, logged, or returned by any public method.
* :meth:`RoutingClient.status` exposes only whether the service is configured
  and reachable — never the key.
* Every network call has an explicit timeout and returns ``None`` on failure so
  callers can fall back to the local hotspot analytics already computed offline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from src.config import get_secret
from src.providers import ProviderStatus, is_real_secret

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.environ.get("ROUTING_TIMEOUT", "15"))
DEFAULT_BASE_URL = "https://api.openrouteservice.org"

#: Routing profiles documented by openrouteservice.
SUPPORTED_PROFILES = (
    "driving-car",
    "driving-hgv",
    "cycling-regular",
    "cycling-road",
    "foot-walking",
    "foot-hiking",
)

#: A short, valid Berlin route used only as a liveness probe.
_HEALTH_PROBE_COORDS = ((13.388860, 52.517037), (13.397634, 52.529407))


def _split_latlon(point: Sequence[float], already_lonlat: bool) -> Tuple[float, float]:
    """Normalise a 2-tuple to ``(lon, lat)`` as required by ORS."""
    first, second = float(point[0]), float(point[1])
    return (first, second) if already_lonlat else (second, first)


class RoutingClient:
    """Thin, secret-safe wrapper around the openrouteservice REST API."""

    name = "openrouteservice"
    label = "HeiGIT openrouteservice"
    kind = "routing"

    # ── configuration ────────────────────────────────────────────────────────

    def api_key(self) -> Optional[str]:
        """Resolve the credential. Never logged, never returned to callers."""
        return get_secret("OPENROUTESERVICE_API_KEY")

    def base_url(self) -> str:
        configured = get_secret("OPENROUTESERVICE_BASE_URL")
        base = str(configured).strip() if configured else DEFAULT_BASE_URL
        return base.rstrip("/")

    def is_configured(self) -> bool:
        return is_real_secret(self.api_key())

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": self.api_key() or "",
            "Content-Type": "application/json",
            "Accept": "application/json, application/geo+json",
        }

    # ── status ───────────────────────────────────────────────────────────────

    def status(self, probe: bool = False) -> ProviderStatus:
        """Describe the service without exposing the credential."""
        if not self.is_configured():
            return ProviderStatus(
                name=self.name,
                label=self.label,
                kind=self.kind,
                configured=False,
                available=False,
                model=None,
                detail="Not configured — routing falls back to offline hotspot analytics.",
            )
        if not probe:
            return ProviderStatus(
                name=self.name,
                label=self.label,
                kind=self.kind,
                configured=True,
                available=None,
                model=None,
                detail="Credential present; availability not yet verified.",
            )
        result = self.health_check()
        return ProviderStatus(
            name=self.name,
            label=self.label,
            kind=self.kind,
            configured=True,
            available=result.get("available"),
            model=None,
            detail=result.get("detail", ""),
        )

    def health_check(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send a minimal real route request to confirm the key works."""
        if not self.is_configured():
            return {"available": False, "detail": "Not configured."}
        try:
            resp = requests.post(
                f"{self.base_url()}/v2/directions/driving-car/geojson",
                json={"coordinates": [list(c) for c in _HEALTH_PROBE_COORDS]},
                headers=self._auth_headers(),
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                return {"available": True, "detail": "Route service reachable."}
            if resp.status_code in (401, 403):
                return {"available": False, "detail": "Credential rejected by openrouteservice."}
            return {"available": False, "detail": f"HTTP {resp.status_code} from openrouteservice."}
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            logger.warning("openrouteservice health probe failed: %s", type(exc).__name__)
            return {"available": False, "detail": f"Probe failed ({type(exc).__name__})."}

    # ── API surface ──────────────────────────────────────────────────────────

    def get_route(
        self,
        origin: Sequence[float],
        destination: Sequence[float],
        profile: str = "driving-car",
        *,
        origin_is_lonlat: bool = False,
        destination_is_lonlat: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return road-network distance/duration between two points.

        Coordinates are given as ``(lat, lon)`` by default (matching the
        project's accident records); pass ``*_is_lonlat=True`` when they are
        already ``(lon, lat)``.

        Returns ``{'distance_km', 'duration_min', 'profile', 'geometry'}`` or
        ``None`` when unconfigured or the request fails.
        """
        if not self.is_configured():
            return None
        if profile not in SUPPORTED_PROFILES:
            logger.warning("Unsupported routing profile requested: %s", profile)
            return None

        coordinates = [
            list(_split_latlon(origin, origin_is_lonlat)),
            list(_split_latlon(destination, destination_is_lonlat)),
        ]
        try:
            resp = requests.post(
                f"{self.base_url()}/v2/directions/{profile}/geojson",
                json={"coordinates": coordinates},
                headers=self._auth_headers(),
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            features = resp.json().get("features") or []
            if not features:
                return None
            feature = features[0]
            summary = feature.get("properties", {}).get("summary", {})
            geometry = feature.get("geometry", {}).get("coordinates", [])
            return {
                "distance_km": round(summary.get("distance", 0.0) / 1000.0, 3),
                "duration_min": round(summary.get("duration", 0.0) / 60.0, 2),
                "profile": profile,
                "geometry": geometry,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("openrouteservice route request failed (%s).", type(exc).__name__)
            return None

    def get_isochrone(
        self,
        center: Sequence[float],
        ranges_minutes: Sequence[int] = (5, 10),
        profile: str = "driving-car",
        *,
        center_is_lonlat: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return reachable-area GeoJSON around *center* for the given minute ranges.

        ``None`` when unconfigured or the request fails.
        """
        if not self.is_configured():
            return None
        ranges_seconds = sorted({int(m) * 60 for m in ranges_minutes if int(m) > 0})
        if not ranges_seconds:
            return None
        try:
            resp = requests.post(
                f"{self.base_url()}/v2/isochrones/{profile}",
                json={
                    "locations": [list(_split_latlon(center, center_is_lonlat))],
                    "range": ranges_seconds,
                },
                headers=self._auth_headers(),
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("openrouteservice isochrone request failed (%s).", type(exc).__name__)
            return None

    def geocode(
        self,
        query: str,
        size: int = 1,
        timeout: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Resolve a place name to coordinates (Pelias geocoder).

        Returns a list of ``{'label', 'lat', 'lon'}`` dicts, or ``None``.
        """
        if not self.is_configured() or not query.strip():
            return None
        try:
            resp = requests.get(
                f"{self.base_url()}/geocode/search",
                params={"text": query.strip(), "size": max(1, min(int(size), 10))},
                headers=self._auth_headers(),
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            features = resp.json().get("features") or []
            results: List[Dict[str, Any]] = []
            for feature in features:
                lon, lat = (feature.get("geometry", {}).get("coordinates") or [None, None])[:2]
                if lat is None or lon is None:
                    continue
                results.append(
                    {
                        "label": feature.get("properties", {}).get("label", query),
                        "lat": float(lat),
                        "lon": float(lon),
                    }
                )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("openrouteservice geocode request failed (%s).", type(exc).__name__)
            return None


_client: Optional[RoutingClient] = None


def get_routing_client() -> RoutingClient:
    """Return a shared routing client (stateless; credentials read per call)."""
    global _client
    if _client is None:
        _client = RoutingClient()
    return _client


def routing_service_status(probe: bool = False) -> Dict[str, Any]:
    """Secret-free routing status descriptor for the UI."""
    return get_routing_client().status(probe=probe).as_dict()
