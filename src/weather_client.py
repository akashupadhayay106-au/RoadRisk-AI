"""
RoadRisk AI — Historical Weather Client & Atmospheric Analytics
================================================================
Fetches, caches, and correlates historical weather data with road accident
locations using the Open-Meteo Historical Weather API.

Features:
- Open-Meteo Historical Archive API integration (free, no API key required)
- SQLite disk caching with spatial & temporal keying (prevents redundant API hits)
- WMO weather code mapping to categories, emoji, and adverse hazard flags
- Defensible spatiotemporal matching against crash records
- Offline fallback simulator ensuring robust zero-downtime execution
- Multi-dimensional correlation metrics (adverse vs clear, temperature vs severity)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import requests

from src.config import DATABASE_PATH, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Open-Meteo Historical Weather API Base Endpoint
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ── WMO Weather Interpretation Code (WW) Mapping ──────────────────────────────
WMO_CODE_MAP: Dict[int, Dict[str, Any]] = {
    0:  {"description": "Clear sky", "category": "Clear", "emoji": "☀️", "is_adverse": False},
    1:  {"description": "Mainly clear", "category": "Clear", "emoji": "🌤️", "is_adverse": False},
    2:  {"description": "Partly cloudy", "category": "Cloudy", "emoji": "⛅", "is_adverse": False},
    3:  {"description": "Overcast", "category": "Cloudy", "emoji": "☁️", "is_adverse": False},
    45: {"description": "Fog", "category": "Fog", "emoji": "🌫️", "is_adverse": True},
    48: {"description": "Depositing rime fog", "category": "Fog", "emoji": "🌫️", "is_adverse": True},
    51: {"description": "Light drizzle", "category": "Drizzle", "emoji": "🌦️", "is_adverse": True},
    53: {"description": "Moderate drizzle", "category": "Drizzle", "emoji": "🌧️", "is_adverse": True},
    55: {"description": "Dense drizzle", "category": "Drizzle", "emoji": "🌧️", "is_adverse": True},
    56: {"description": "Light freezing drizzle", "category": "Freezing Rain", "emoji": "🌨️", "is_adverse": True},
    57: {"description": "Dense freezing drizzle", "category": "Freezing Rain", "emoji": "🌨️", "is_adverse": True},
    61: {"description": "Slight rain", "category": "Rain", "emoji": "🌦️", "is_adverse": True},
    63: {"description": "Moderate rain", "category": "Rain", "emoji": "🌧️", "is_adverse": True},
    65: {"description": "Heavy rain", "category": "Rain", "emoji": "⛈️", "is_adverse": True},
    66: {"description": "Light freezing rain", "category": "Freezing Rain", "emoji": "🌨️", "is_adverse": True},
    67: {"description": "Heavy freezing rain", "category": "Freezing Rain", "emoji": "🌨️", "is_adverse": True},
    71: {"description": "Slight snow fall", "category": "Snow", "emoji": "🌨️", "is_adverse": True},
    73: {"description": "Moderate snow fall", "category": "Snow", "emoji": "❄️", "is_adverse": True},
    75: {"description": "Heavy snow fall", "category": "Snow", "emoji": "❄️", "is_adverse": True},
    77: {"description": "Snow grains", "category": "Snow", "emoji": "❄️", "is_adverse": True},
    80: {"description": "Slight rain showers", "category": "Rain Showers", "emoji": "🌦️", "is_adverse": True},
    81: {"description": "Moderate rain showers", "category": "Rain Showers", "emoji": "🌧️", "is_adverse": True},
    82: {"description": "Violent rain showers", "category": "Rain Showers", "emoji": "⛈️", "is_adverse": True},
    85: {"description": "Slight snow showers", "category": "Snow Showers", "emoji": "🌨️", "is_adverse": True},
    86: {"description": "Heavy snow showers", "category": "Snow Showers", "emoji": "❄️", "is_adverse": True},
    95: {"description": "Thunderstorm", "category": "Thunderstorm", "emoji": "⚡", "is_adverse": True},
    96: {"description": "Thunderstorm with slight hail", "category": "Thunderstorm", "emoji": "🌩️", "is_adverse": True},
    99: {"description": "Thunderstorm with heavy hail", "category": "Thunderstorm", "emoji": "🌩️", "is_adverse": True},
}

DEFAULT_WMO_INFO = {
    "description": "Variable atmospheric condition",
    "category": "Other",
    "emoji": "🌤️",
    "is_adverse": False
}


def get_wmo_metadata(code: Optional[Union[int, float]]) -> Dict[str, Any]:
    """Retrieve metadata (description, category, emoji, is_adverse) for a WMO code."""
    if code is None or pd.isna(code):
        return DEFAULT_WMO_INFO
    try:
        int_code = int(code)
        return WMO_CODE_MAP.get(int_code, DEFAULT_WMO_INFO)
    except (ValueError, TypeError):
        return DEFAULT_WMO_INFO


# ── SQLite Weather Cache Management ──────────────────────────────────────────

DDL_WEATHER_CACHE = """
CREATE TABLE IF NOT EXISTS weather_cache (
    cache_key TEXT PRIMARY KEY,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    datetime TEXT NOT NULL,
    temperature_2m REAL,
    relative_humidity_2m REAL,
    precipitation REAL,
    rain REAL,
    weather_code INTEGER,
    wind_speed_10m REAL,
    wmo_description TEXT,
    wmo_category TEXT,
    wmo_emoji TEXT,
    is_adverse INTEGER,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

DDL_WEATHER_CACHE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_weather_cache_coords ON weather_cache(latitude, longitude);",
    "CREATE INDEX IF NOT EXISTS idx_weather_cache_date ON weather_cache(date);",
    "CREATE INDEX IF NOT EXISTS idx_weather_cache_category ON weather_cache(wmo_category);",
]


def init_weather_cache(db_path: Path = DATABASE_PATH) -> None:
    """Initialize the weather_cache table and indexes in SQLite."""
    ensure_directories()
    with sqlite3.connect(db_path) as conn:
        conn.execute(DDL_WEATHER_CACHE)
        for idx in DDL_WEATHER_CACHE_INDEXES:
            conn.execute(idx)
        conn.commit()


def make_cache_key(lat: float, lon: float, date_str: str, hour: int) -> str:
    """
    Generate a normalized spatial-temporal cache key.
    Rounds coordinates to 2 decimal places (~1.1 km precision) to maximize
    cache hit rates for clustered incidents within the same urban area.
    """
    return f"{round(float(lat), 2)}_{round(float(lon), 2)}_{date_str}_{int(hour):02d}"


def get_cached_weather(
    lat: float,
    lon: float,
    date_str: str,
    hour: Optional[int] = None,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Query the SQLite weather cache for given coordinate and date.
    If hour is specified, returns that exact hour. Otherwise returns all cached hours for that date.
    """
    init_weather_cache(db_path)
    lat_r = round(float(lat), 2)
    lon_r = round(float(lon), 2)

    query = """
    SELECT * FROM weather_cache
    WHERE round(latitude, 2) = ? AND round(longitude, 2) = ? AND date = ?
    """
    params: List[Any] = [lat_r, lon_r, date_str]

    if hour is not None:
        query += " AND hour = ?"
        params.append(int(hour))

    query += " ORDER BY hour ASC"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def save_weather_to_cache(df_weather: pd.DataFrame, db_path: Path = DATABASE_PATH) -> int:
    """
    Upsert weather records into the SQLite cache.
    Returns the number of records saved.
    """
    if df_weather.empty:
        return 0

    init_weather_cache(db_path)
    records = []
    for _, row in df_weather.iterrows():
        key = row.get("cache_key") or make_cache_key(
            row["latitude"], row["longitude"], row["date"], row["hour"]
        )
        records.append((
            key,
            float(row["latitude"]),
            float(row["longitude"]),
            str(row["date"]),
            int(row["hour"]),
            str(row.get("datetime", f"{row['date']} {row['hour']:02d}:00:00")),
            float(row["temperature_2m"]) if pd.notna(row.get("temperature_2m")) else None,
            float(row["relative_humidity_2m"]) if pd.notna(row.get("relative_humidity_2m")) else None,
            float(row["precipitation"]) if pd.notna(row.get("precipitation")) else None,
            float(row["rain"]) if pd.notna(row.get("rain")) else None,
            int(row["weather_code"]) if pd.notna(row.get("weather_code")) else None,
            float(row["wind_speed_10m"]) if pd.notna(row.get("wind_speed_10m")) else None,
            str(row.get("wmo_description", "")),
            str(row.get("wmo_category", "")),
            str(row.get("wmo_emoji", "")),
            1 if row.get("is_adverse") else 0,
        ))

    upsert_sql = """
    INSERT INTO weather_cache (
        cache_key, latitude, longitude, date, hour, datetime,
        temperature_2m, relative_humidity_2m, precipitation, rain,
        weather_code, wind_speed_10m, wmo_description, wmo_category,
        wmo_emoji, is_adverse
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(cache_key) DO UPDATE SET
        temperature_2m = excluded.temperature_2m,
        relative_humidity_2m = excluded.relative_humidity_2m,
        precipitation = excluded.precipitation,
        rain = excluded.rain,
        weather_code = excluded.weather_code,
        wind_speed_10m = excluded.wind_speed_10m,
        wmo_description = excluded.wmo_description,
        wmo_category = excluded.wmo_category,
        wmo_emoji = excluded.wmo_emoji,
        is_adverse = excluded.is_adverse,
        fetched_at = CURRENT_TIMESTAMP;
    """

    with sqlite3.connect(db_path) as conn:
        conn.executemany(upsert_sql, records)
        conn.commit()

    return len(records)


# ── Open-Meteo API Fetcher with Fallback ─────────────────────────────────────

def fetch_historical_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    db_path: Path = DATABASE_PATH,
    timeout_sec: int = 10,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo for a given bounding date range and coordinate.
    Checks the local SQLite cache first. If any date/hour is missing, queries Open-Meteo API.
    If the network or API is unavailable, provides an intelligent seasonal fallback.
    """
    lat = float(lat)
    lon = float(lon)

    # 1. Check SQLite cache if requested
    if use_cache:
        cached_df = get_cached_weather(lat, lon, start_date, db_path=db_path)
        # If single-day request and we already have full 24 hours, return immediately
        if start_date == end_date and len(cached_df) >= 24:
            return cached_df

    # 2. Query Open-Meteo Historical Archive API
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    headers = {"User-Agent": "RoadRiskAI-DataSciencePortfolio/1.0"}

    try:
        response = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params=params,
            headers=headers,
            timeout=timeout_sec,
        )
        if response.status_code == 200:
            data = response.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            if not times:
                logger.warning("Open-Meteo returned 200 but hourly time array was empty.")
                return _generate_fallback_weather(lat, lon, start_date, end_date)

            rows = []
            for i, dt_str in enumerate(times):
                # dt_str format: "2024-01-01T00:00"
                parts = dt_str.split("T")
                d_str = parts[0]
                h_int = int(parts[1].split(":")[0])
                w_code = hourly.get("weather_code", [0])[i] if i < len(hourly.get("weather_code", [])) else 0
                meta = get_wmo_metadata(w_code)

                rows.append({
                    "cache_key": make_cache_key(lat, lon, d_str, h_int),
                    "latitude": lat,
                    "longitude": lon,
                    "date": d_str,
                    "hour": h_int,
                    "datetime": f"{d_str} {h_int:02d}:00:00",
                    "temperature_2m": hourly.get("temperature_2m", [None])[i],
                    "relative_humidity_2m": hourly.get("relative_humidity_2m", [None])[i],
                    "precipitation": hourly.get("precipitation", [0.0])[i],
                    "rain": hourly.get("rain", [0.0])[i],
                    "weather_code": w_code,
                    "wind_speed_10m": hourly.get("wind_speed_10m", [None])[i],
                    "wmo_description": meta["description"],
                    "wmo_category": meta["category"],
                    "wmo_emoji": meta["emoji"],
                    "is_adverse": 1 if meta["is_adverse"] else 0,
                    "source": "Open-Meteo Historical Archive",
                })

            df_result = pd.DataFrame(rows)
            # Save to SQLite cache for future instant offline lookups
            try:
                save_weather_to_cache(df_result, db_path=db_path)
            except Exception as e:
                logger.warning(f"Could not save weather to cache: {e}")

            return df_result
        else:
            logger.warning(
                f"Open-Meteo API returned HTTP {response.status_code}: {response.text[:200]}"
            )
            return _generate_fallback_weather(lat, lon, start_date, end_date)

    except (requests.RequestException, Exception) as exc:
        logger.warning(f"Network error or timeout contacting Open-Meteo API ({exc}). Using fallback.")
        return _generate_fallback_weather(lat, lon, start_date, end_date)


def _generate_fallback_weather(
    lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Intelligent seasonal climatological fallback generator when offline.
    Uses latitude-based temperature curves, tropical monsoon logic (June-Sept rain in India),
    and standard diurnal temperature oscillations so the UI remains fully functional.
    """
    start_dt = pd.to_datetime(f"{start_date} 00:00:00")
    end_dt = pd.to_datetime(f"{end_date} 23:00:00")
    date_range = pd.date_range(start_dt, end_dt, freq="h")

    rows = []
    # Seed by lat/lon so fallback is deterministic for the same location
    seed = int((abs(lat) * 100 + abs(lon) * 10) % 100000)
    rng = np.random.default_rng(seed)

    for dt in date_range:
        d_str = dt.strftime("%Y-%m-%d")
        h_int = dt.hour
        month = dt.month

        # Base temperature: warm tropics (India lat ~8-35), cooler in winter (Dec-Jan)
        is_winter = month in [12, 1, 2]
        is_monsoon = month in [6, 7, 8, 9]
        is_summer = month in [3, 4, 5]

        base_temp = 22.0 if is_winter else (32.0 if is_summer else 28.0)
        # Diurnal fluctuation: coolest at 5 AM, warmest at 2 PM
        diurnal = 5.0 * np.sin((h_int - 8) * np.pi / 12)
        temp = round(float(base_temp + diurnal + rng.normal(0, 1.2)), 1)

        # Monsoon precipitation logic
        if is_monsoon:
            precip_prob = 0.40
            humidity = round(float(rng.uniform(70, 95)), 1)
        else:
            precip_prob = 0.05
            humidity = round(float(rng.uniform(35, 65)), 1)

        has_rain = rng.random() < precip_prob
        if has_rain:
            precip = round(float(rng.exponential(2.5)), 1)
            w_code = int(rng.choice([61, 63, 65, 80, 95]))
        elif is_winter and (h_int < 7 or h_int > 22) and rng.random() < 0.25:
            precip = 0.0
            w_code = 45  # Fog
        else:
            precip = 0.0
            w_code = int(rng.choice([0, 1, 2, 3]))

        meta = get_wmo_metadata(w_code)
        wind = round(float(rng.uniform(4.0, 18.0)), 1)

        rows.append({
            "cache_key": make_cache_key(lat, lon, d_str, h_int),
            "latitude": lat,
            "longitude": lon,
            "date": d_str,
            "hour": h_int,
            "datetime": f"{d_str} {h_int:02d}:00:00",
            "temperature_2m": temp,
            "relative_humidity_2m": humidity,
            "precipitation": precip,
            "rain": precip,
            "weather_code": w_code,
            "wind_speed_10m": wind,
            "wmo_description": meta["description"],
            "wmo_category": meta["category"],
            "wmo_emoji": meta["emoji"],
            "is_adverse": 1 if meta["is_adverse"] else 0,
            "source": "Climatological Fallback Model (Offline)",
        })

    return pd.DataFrame(rows)


def get_incident_weather_match(
    lat: float,
    lon: float,
    date_str: str,
    hour: int,
    use_cache: bool = True,
    db_path: Path = DATABASE_PATH,
) -> Dict[str, Any]:
    """
    Look up the exact historical weather for an individual crash incident by coordinate, date, and hour.
    """
    df_day = fetch_historical_weather(
        lat=lat,
        lon=lon,
        start_date=date_str,
        end_date=date_str,
        use_cache=use_cache,
        db_path=db_path,
    )
    if df_day.empty:
        return {}

    match = df_day[df_day["hour"] == int(hour)]
    if not match.empty:
        return match.iloc[0].to_dict()

    # Fallback to nearest available hour
    df_day["hour_diff"] = (df_day["hour"] - int(hour)).abs()
    nearest = df_day.sort_values("hour_diff").iloc[0]
    return nearest.to_dict()


# ── Macro Statistical Weather Correlation Engine ─────────────────────────────

def compute_weather_risk_metrics(df_accidents: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistical correlation metrics between weather and accident severity.

    Input DataFrame requires at minimum:
    ['weather', 'accident_severity'] or equivalent columns.
    Optional: ['temperature', 'visibility', 'time_of_day']
    """
    if df_accidents.empty:
        return {}

    total_accidents = len(df_accidents)

    # Normalize weather strings
    weather_col = "weather" if "weather" in df_accidents.columns else "Weather_Conditions"
    severity_col = "accident_severity" if "accident_severity" in df_accidents.columns else "Severity"

    if weather_col not in df_accidents.columns or severity_col not in df_accidents.columns:
        return {"total_accidents": total_accidents}

    df = df_accidents.copy()
    df["weather_clean"] = df[weather_col].astype(str).str.strip().str.title()
    df["severity_clean"] = df[severity_col].astype(str).str.strip().str.title()

    # Define Adverse weather flag (Rain, Fog, Storm, Mist, Heavy Rain)
    adverse_keywords = ["Rain", "Fog", "Storm", "Mist", "Snow", "Wind", "Hail", "Freezing"]
    df["is_adverse_weather"] = df["weather_clean"].apply(
        lambda w: any(k.lower() in w.lower() for k in adverse_keywords)
    )

    # Severity weights: Fatal=3, Serious=2, Minor/Slight=1
    severity_weights = {"Fatal": 3, "Serious": 2, "Minor": 1, "Slight": 1}
    df["severity_score"] = df["severity_clean"].map(lambda s: severity_weights.get(s, 1))

    # 1. Weather Category Breakdown
    weather_counts = df["weather_clean"].value_counts().to_dict()
    weather_pct = (df["weather_clean"].value_counts(normalize=True) * 100).round(2).to_dict()

    # 2. Severity Cross-Tabulation
    cross_tab = pd.crosstab(df["weather_clean"], df["severity_clean"], normalize="index") * 100
    cross_tab = cross_tab.round(2).to_dict(orient="index")

    # 3. Adverse vs Non-Adverse Comparison
    adverse_group = df.groupby("is_adverse_weather")
    adverse_stats = {}
    for is_adv, grp in adverse_group:
        label = "Adverse Weather (Rain/Fog/Storm)" if is_adv else "Normal Weather (Clear/Cloudy)"
        fatal_count = (grp["severity_clean"] == "Fatal").sum()
        serious_count = (grp["severity_clean"] == "Serious").sum()
        total_grp = len(grp)

        adverse_stats[label] = {
            "total_accidents": int(total_grp),
            "pct_of_all": round(float(total_grp / total_accidents * 100), 2),
            "fatal_rate": round(float(fatal_count / total_grp * 100), 2) if total_grp else 0.0,
            "serious_rate": round(float(serious_count / total_grp * 100), 2) if total_grp else 0.0,
            "severe_combined_rate": round(float((fatal_count + serious_count) / total_grp * 100), 2) if total_grp else 0.0,
            "mean_severity_score": round(float(grp["severity_score"].mean()), 2) if total_grp else 0.0,
        }

    # Relative Risk Ratio (Severe rate in Adverse / Severe rate in Normal)
    adv_label = "Adverse Weather (Rain/Fog/Storm)"
    norm_label = "Normal Weather (Clear/Cloudy)"
    rel_risk = 1.0
    if adv_label in adverse_stats and norm_label in adverse_stats:
        norm_rate = adverse_stats[norm_label]["severe_combined_rate"]
        adv_rate = adverse_stats[adv_label]["severe_combined_rate"]
        if norm_rate > 0:
            rel_risk = round(adv_rate / norm_rate, 2)

    # 4. Temperature Correlation (if present)
    temp_analysis = {}
    if "temperature" in df.columns and pd.api.types.is_numeric_dtype(df["temperature"]):
        valid_temp = df.dropna(subset=["temperature"])
        if not valid_temp.empty:
            temp_analysis = {
                "overall_mean": round(float(valid_temp["temperature"].mean()), 1),
                "overall_min": round(float(valid_temp["temperature"].min()), 1),
                "overall_max": round(float(valid_temp["temperature"].max()), 1),
                "by_severity": valid_temp.groupby("severity_clean")["temperature"].agg(["mean", "median", "std"]).round(2).to_dict(orient="index"),
            }

    # 5. Visibility Correlation (if present)
    vis_analysis = {}
    if "visibility" in df.columns:
        vis_cross = pd.crosstab(df["visibility"], df["severity_clean"], normalize="index") * 100
        vis_analysis = vis_cross.round(2).to_dict(orient="index")

    return {
        "total_accidents": total_accidents,
        "weather_counts": weather_counts,
        "weather_percentages": weather_pct,
        "severity_by_weather": cross_tab,
        "adverse_comparison": adverse_stats,
        "relative_severe_risk_ratio": rel_risk,
        "temperature_analysis": temp_analysis,
        "visibility_analysis": vis_analysis,
    }


def warm_weather_cache(db_path: Path = DATABASE_PATH) -> int:
    """
    Pre-populates the SQLite cache with real or realistic historical weather
    observations for major Indian accident locations. Ensures tests and offline
    demos execute instantly without external network latency.
    """
    sample_cities = [
        {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777, "date": "2024-01-15"},
        {"city": "Delhi", "lat": 28.6139, "lon": 77.2090, "date": "2024-01-15"},
        {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "date": "2024-01-15"},
        {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639, "date": "2024-01-15"},
        {"city": "Chennai", "lat": 13.0827, "lon": 80.2707, "date": "2024-01-15"},
    ]

    total_loaded = 0
    for target in sample_cities:
        df = fetch_historical_weather(
            lat=target["lat"],
            lon=target["lon"],
            start_date=target["date"],
            end_date=target["date"],
            use_cache=True,
            db_path=db_path,
        )
        total_loaded += len(df)

    logger.info(f"Weather cache warmed with {total_loaded} hourly records across {len(sample_cities)} major cities.")
    return total_loaded


def weather_service_status(probe: bool = False) -> Dict[str, Any]:
    """
    Secret-free status descriptor for the weather integration.

    Open-Meteo is public and keyless, so it is always the default source and is
    reported as configured. A probe performs a tiny reachability request.

    NOTE: an unidentified third-party weather key was supplied separately, but
    its provider (WeatherAPI.com vs OpenWeatherMap) is still unconfirmed, so no
    key-based weather provider is wired up. The status deliberately says so
    instead of guessing.
    """
    detail = (
        "Keyless Open-Meteo historical archive is the default weather source. "
        "No key-based weather provider is configured (the supplied weather key's "
        "provider is still unconfirmed)."
    )
    available: Optional[bool] = None
    if probe:
        try:
            resp = requests.get(
                OPEN_METEO_ARCHIVE_URL,
                params={
                    "latitude": 19.0760,
                    "longitude": 72.8777,
                    "start_date": "2024-01-15",
                    "end_date": "2024-01-15",
                    "hourly": "temperature_2m",
                },
                headers={"User-Agent": "RoadRiskAI-DataSciencePortfolio/1.0"},
                timeout=10,
            )
            available = resp.status_code == 200
            detail = (
                "Open-Meteo archive reachable."
                if available
                else f"Open-Meteo returned HTTP {resp.status_code}; cached/seasonal fallback is used."
            )
        except Exception as exc:  # noqa: BLE001
            available = False
            detail = f"Open-Meteo probe failed ({type(exc).__name__}); cached/seasonal fallback is used."

    return {
        "name": "open-meteo",
        "label": "Open-Meteo (keyless)",
        "kind": "weather",
        "configured": True,
        "available": available,
        "model": None,
        "detail": detail,
    }


if __name__ == "__main__":
    print("Testing Open-Meteo Historical Weather Client...")
    init_weather_cache()
    df_sample = fetch_historical_weather(
        lat=19.0760, lon=72.8777, start_date="2024-01-15", end_date="2024-01-15"
    )
    print(f"Fetched {len(df_sample)} hours for Mumbai (2024-01-15).")
    print(df_sample[["hour", "temperature_2m", "wmo_description", "is_adverse"]].head())

    # Test single-incident lookup
    incident = get_incident_weather_match(19.0760, 72.8777, "2024-01-15", 14)
    print(f"14:00 Weather Match: {incident.get('temperature_2m')}°C, {incident.get('wmo_description')}")
