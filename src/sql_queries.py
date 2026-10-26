"""
RoadRisk AI — SQL Query Bank & Analytical Functions
===================================================
Parameterized SQL queries and high-level analytical query wrappers
operating directly against the SQLite database.
"""

from typing import Dict, Any, Optional
import pandas as pd
from src.database import execute_query


# ── KPI Metrics Query ────────────────────────────────────────────────────────

def get_overall_kpis(
    city: Optional[str] = None,
    state: Optional[str] = None,
    year: Optional[int] = None,
    severity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve aggregated KPIs with optional parameterized filters.
    """
    where_clauses = ["1=1"]
    params = []

    if city and city != "All":
        where_clauses.append("city = ?")
        params.append(city)
    if state and state != "All":
        where_clauses.append("state = ?")
        params.append(state)
    if year and year != "All":
        where_clauses.append("year = ?")
        params.append(int(year))
    if severity and severity != "All":
        where_clauses.append("accident_severity = ?")
        params.append(severity.lower())

    where_sql = " AND ".join(where_clauses)

    sql = f"""
    SELECT 
        COUNT(*) AS total_accidents,
        COALESCE(SUM(casualties), 0) AS total_casualties,
        COALESCE(SUM(vehicles_involved), 0) AS total_vehicles,
        COALESCE(SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END), 0) AS fatal_accidents,
        COALESCE(SUM(CASE WHEN accident_severity = 'major' THEN 1 ELSE 0 END), 0) AS major_accidents,
        COALESCE(SUM(CASE WHEN accident_severity = 'minor' THEN 1 ELSE 0 END), 0) AS minor_accidents,
        ROUND(AVG(risk_score), 3) AS avg_risk_score
    FROM accidents_india
    WHERE {where_sql};
    """
    df = execute_query(sql, params)
    if df.empty:
        return {}

    row = df.iloc[0].to_dict()
    total = row["total_accidents"]
    fatal = row["fatal_accidents"]
    row["fatality_rate_pct"] = round((fatal / total) * 100, 2) if total > 0 else 0.0
    return row


# ── Monthly Trend Analysis ───────────────────────────────────────────────────

def get_monthly_trends(
    year: Optional[int] = None,
    city: Optional[str] = None
) -> pd.DataFrame:
    """
    Retrieve monthly accident trajectory and casualty counts.
    """
    where_clauses = ["1=1"]
    params = []

    if year and year != "All":
        where_clauses.append("year = ?")
        params.append(int(year))
    if city and city != "All":
        where_clauses.append("city = ?")
        params.append(city)

    where_sql = " AND ".join(where_clauses)

    sql = f"""
    SELECT 
        year,
        month,
        month_name,
        printf('%04d-%02d', year, month) AS year_month,
        COUNT(*) AS accident_count,
        SUM(casualties) AS total_casualties,
        SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END) AS fatal_accidents
    FROM accidents_india
    WHERE {where_sql}
    GROUP BY year, month, month_name
    ORDER BY year ASC, month ASC;
    """
    return execute_query(sql, params)


# ── Severity Distribution ────────────────────────────────────────────────────

def get_severity_distribution(
    city: Optional[str] = None,
    year: Optional[int] = None
) -> pd.DataFrame:
    """
    Get frequency and percentage share of accident severity levels.
    """
    where_clauses = ["1=1"]
    params = []

    if city and city != "All":
        where_clauses.append("city = ?")
        params.append(city)
    if year and year != "All":
        where_clauses.append("year = ?")
        params.append(int(year))

    where_sql = " AND ".join(where_clauses)

    sql = f"""
    SELECT 
        accident_severity,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
    FROM accidents_india
    WHERE {where_sql}
    GROUP BY accident_severity
    ORDER BY count DESC;
    """
    return execute_query(sql, params)


# ── City Safety Benchmarking ─────────────────────────────────────────────────

def get_city_comparisons() -> pd.DataFrame:
    """
    Compare total incidents, casualties, and fatal severity rate across all cities.
    """
    sql = """
    SELECT 
        city,
        state,
        COUNT(*) AS total_accidents,
        SUM(casualties) AS total_casualties,
        SUM(vehicles_involved) AS total_vehicles,
        SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END) AS fatal_accidents,
        ROUND(AVG(risk_score), 3) AS avg_risk_score,
        ROUND(SUM(CASE WHEN accident_severity = 'fatal' THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS fatal_rate_pct
    FROM accidents_india
    GROUP BY city, state
    ORDER BY total_accidents DESC;
    """
    return execute_query(sql)


# ── Weather & Environmental Impact ───────────────────────────────────────────

def get_weather_impact_analysis() -> pd.DataFrame:
    """
    Analyze accident frequencies and casualty rates grouped by weather conditions.
    """
    sql = """
    SELECT 
        weather,
        visibility,
        COUNT(*) AS accident_count,
        SUM(casualties) AS total_casualties,
        ROUND(AVG(casualties), 2) AS avg_casualties_per_crash,
        SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END) AS fatal_count,
        ROUND(AVG(risk_score), 3) AS avg_risk_score
    FROM accidents_india
    GROUP BY weather, visibility
    ORDER BY accident_count DESC;
    """
    return execute_query(sql)


# ── Primary Cause Analysis ───────────────────────────────────────────────────

def get_cause_analysis(city: Optional[str] = None) -> pd.DataFrame:
    """
    Break down accident counts and casualties by reported primary cause.
    """
    where_sql = "1=1"
    params = []
    if city and city != "All":
        where_sql = "city = ?"
        params.append(city)

    sql = f"""
    SELECT 
        cause,
        COUNT(*) AS accident_count,
        SUM(casualties) AS total_casualties,
        SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END) AS fatal_accidents,
        ROUND(AVG(risk_score), 3) AS avg_risk_score
    FROM accidents_india
    WHERE {where_sql}
    GROUP BY cause
    ORDER BY accident_count DESC;
    """
    return execute_query(sql, params)


# ── Hourly Temporal Pattern ──────────────────────────────────────────────────

def get_hourly_distribution() -> pd.DataFrame:
    """
    Analyze incident frequency across 24 hours and peak vs off-peak times.
    """
    sql = """
    SELECT 
        hour,
        time_of_day,
        is_peak_hour,
        COUNT(*) AS accident_count,
        SUM(casualties) AS total_casualties,
        ROUND(AVG(risk_score), 3) AS avg_risk_score
    FROM accidents_india
    GROUP BY hour, time_of_day, is_peak_hour
    ORDER BY hour ASC;
    """
    return execute_query(sql)


# ── Day-of-Week vs Hour Heatmap Matrix ───────────────────────────────────────

def get_day_hour_heatmap_matrix(city: Optional[str] = None) -> pd.DataFrame:
    """
    Generate accident volume by Day of Week (Monday-Sunday) and Hour (0-23).
    """
    where_sql = "1=1"
    params = []
    if city and city != "All":
        where_sql = "city = ?"
        params.append(city)

    sql = f"""
    SELECT 
        day_of_week,
        hour,
        COUNT(*) AS incident_count
    FROM accidents_india
    WHERE {where_sql}
    GROUP BY day_of_week, hour
    ORDER BY hour ASC;
    """
    return execute_query(sql, params)


# ── State-Level Comparison ───────────────────────────────────────────────────

def get_state_comparisons() -> pd.DataFrame:
    """
    Aggregates accident count, casualties, and fatal crash rate across states.
    """
    sql = """
    SELECT 
        state,
        COUNT(*) AS total_accidents,
        SUM(casualties) AS total_casualties,
        SUM(vehicles_involved) AS total_vehicles,
        SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END) AS fatal_accidents,
        ROUND(SUM(CASE WHEN accident_severity = 'fatal' THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS fatal_rate_pct,
        ROUND(AVG(risk_score), 3) AS avg_risk_score
    FROM accidents_india
    GROUP BY state
    ORDER BY total_accidents DESC;
    """
    return execute_query(sql)


# ── Vehicle & Collision Dynamics ─────────────────────────────────────────────

def get_vehicle_collision_breakdown() -> pd.DataFrame:
    """
    Breakdown of vehicles involved and casualty outcomes.
    """
    sql = """
    SELECT 
        vehicles_involved,
        COUNT(*) AS total_accidents,
        SUM(casualties) AS total_casualties,
        ROUND(AVG(casualties), 2) AS avg_casualties_per_accident,
        SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END) AS fatal_accidents
    FROM accidents_india
    GROUP BY vehicles_involved
    ORDER BY vehicles_involved ASC;
    """
    return execute_query(sql)


# ── Single Record Inspector ──────────────────────────────────────────────────

def get_detailed_accident_record(accident_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch all 30 fields of a single incident record by accident_id.
    """
    sql = "SELECT * FROM accidents_india WHERE accident_id = ? LIMIT 1;"
    df = execute_query(sql, [int(accident_id)])
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ── Catalog of Predefined Queries for UI ─────────────────────────────────────

PREDEFINED_QUERIES = {
    "1. High-Casualty Hotspot Summary": {
        "title": "Top Incident Concentrated Cities & Casualty Ratios",
        "description": "Compares accident counts, total casualties, and fatal crash percentages across all audited cities.",
        "sql": """SELECT 
    city,
    state,
    COUNT(*) AS total_accidents,
    SUM(casualties) AS total_casualties,
    SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END) AS fatal_accidents,
    ROUND(SUM(CASE WHEN accident_severity = 'fatal' THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS fatal_percentage,
    ROUND(AVG(risk_score), 3) AS mean_risk_score
FROM accidents_india
GROUP BY city, state
ORDER BY total_accidents DESC;""",
    },
    "2. Weather and Visibility Interaction": {
        "title": "Accident Frequency by Atmospheric Condition",
        "description": "Inspects how weather conditions (clear, rain, fog) cross-tabulate with visibility levels.",
        "sql": """SELECT 
    weather,
    visibility,
    COUNT(*) AS total_accidents,
    SUM(casualties) AS total_casualties,
    ROUND(AVG(casualties), 2) AS mean_casualties,
    ROUND(AVG(temperature), 1) AS mean_temp_celsius
FROM accidents_india
GROUP BY weather, visibility
ORDER BY total_accidents DESC;""",
    },
    "3. Peak Hour vs Road Type Distribution": {
        "title": "Peak vs Off-Peak Incidents by Road Type",
        "description": "Cross-analyzes incident volumes on highways, urban roads, and rural routes during peak traffic hours.",
        "sql": """SELECT 
    road_type,
    CASE WHEN is_peak_hour = 1 THEN 'Peak Hour' ELSE 'Off-Peak' END AS peak_status,
    COUNT(*) AS incident_count,
    ROUND(AVG(vehicles_involved), 2) AS avg_vehicles_involved,
    ROUND(AVG(risk_score), 3) AS avg_risk_score
FROM accidents_india
GROUP BY road_type, is_peak_hour
ORDER BY road_type, is_peak_hour DESC;""",
    },
    "4. Weekend vs Weekday Severity Patterns": {
        "title": "Weekend Casualty & Severity Comparison",
        "description": "Measures whether weekends exhibit disproportionately higher proportions of fatal or major accidents.",
        "sql": """SELECT 
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    accident_severity,
    COUNT(*) AS accident_count,
    SUM(casualties) AS casualties,
    ROUND(AVG(risk_score), 3) AS mean_risk_score
FROM accidents_india
GROUP BY is_weekend, accident_severity
ORDER BY is_weekend, accident_count DESC;""",
    },
    "5. Highway Corridor Chainage Analysis (OpenCity)": {
        "title": "Highway Kilometer Chainage Accident Density",
        "description": "Examines accident frequencies across highway chainage kilometer markers from the OpenCity dataset.",
        "sql": """SELECT 
    CAST(chainage_km / 10 AS INT) * 10 AS km_bracket,
    COUNT(*) AS incident_count,
    COALESCE(SUM(casualties_total), 0) AS total_casualties,
    COALESCE(SUM(casualties_fatal), 0) AS fatal_casualties
FROM highway_accidents_opencity
WHERE chainage_km IS NOT NULL
GROUP BY km_bracket
ORDER BY incident_count DESC
LIMIT 15;""",
    },
}
