"""
RoadRisk AI — SQLite Database Module
====================================
Manages the relational SQLite database (data/processed/roadrisk.db).
Includes DDL definitions, primary keys, indexes, data loading pipelines,
and safe read-only analytical query execution.
"""

import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import logging
from typing import Dict, Any, Optional
import pandas as pd

from src.config import DATABASE_PATH, PROCESSED_DIR, RAW_DATA_DIR, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── DDL Statements ───────────────────────────────────────────────────────────

DDL_ACCIDENTS_INDIA = """
CREATE TABLE IF NOT EXISTS accidents_india (
    accident_id INTEGER PRIMARY KEY,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    accident_date TEXT NOT NULL,
    accident_time TEXT,
    hour INTEGER NOT NULL,
    time_of_day TEXT NOT NULL,
    day_of_week TEXT NOT NULL,
    is_weekend INTEGER NOT NULL,
    is_peak_hour INTEGER NOT NULL,
    road_type TEXT NOT NULL,
    lanes INTEGER NOT NULL,
    traffic_signal INTEGER NOT NULL,
    weather TEXT NOT NULL,
    visibility TEXT NOT NULL,
    temperature REAL NOT NULL,
    traffic_density TEXT NOT NULL,
    cause TEXT NOT NULL,
    accident_severity TEXT NOT NULL,
    vehicles_involved INTEGER NOT NULL,
    casualties INTEGER NOT NULL,
    festival TEXT NOT NULL DEFAULT 'None',
    risk_score REAL NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name TEXT NOT NULL,
    day INTEGER NOT NULL,
    quarter INTEGER NOT NULL
);
"""

DDL_INDEXES_ACCIDENTS_INDIA = [
    "CREATE INDEX IF NOT EXISTS idx_acc_date ON accidents_india(accident_date);",
    "CREATE INDEX IF NOT EXISTS idx_acc_city ON accidents_india(city);",
    "CREATE INDEX IF NOT EXISTS idx_acc_state ON accidents_india(state);",
    "CREATE INDEX IF NOT EXISTS idx_acc_severity ON accidents_india(accident_severity);",
    "CREATE INDEX IF NOT EXISTS idx_acc_weather ON accidents_india(weather);",
    "CREATE INDEX IF NOT EXISTS idx_acc_cause ON accidents_india(cause);",
    "CREATE INDEX IF NOT EXISTS idx_acc_year_month ON accidents_india(year, month);",
    "CREATE INDEX IF NOT EXISTS idx_acc_coords ON accidents_india(latitude, longitude);",
]

DDL_OPENCITY_HIGHWAY = """
CREATE TABLE IF NOT EXISTS highway_accidents_opencity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accident_index INTEGER,
    accident_date TEXT,
    day_of_week INTEGER,
    time_of_accident TEXT,
    location_code INTEGER,
    chainage_km REAL,
    road_side TEXT,
    nature_of_accident TEXT,
    severity_code INTEGER,
    cause_code TEXT,
    road_condition REAL,
    intersection_type REAL,
    weather_condition REAL,
    vehicles_involved INTEGER,
    casualties_fatal REAL,
    casualties_grievous REAL,
    casualties_minor REAL,
    casualties_total INTEGER
);
"""

DDL_DATASET_CATALOG = """
CREATE TABLE IF NOT EXISTS dataset_catalog (
    table_name TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    description TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ── Connection Helpers ───────────────────────────────────────────────────────

def get_db_connection(db_path: Path | str | None = None, read_only: bool = False) -> sqlite3.Connection:
    """
    Create a connection to the SQLite database.
    If read_only=True, opens URI connection in ro mode to protect against write operations.
    """
    path = Path(db_path or DATABASE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if read_only and path.exists():
        uri_path = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri_path, uri=True)
    else:
        conn = sqlite3.connect(str(path))

    conn.row_factory = sqlite3.Row
    return conn


# ── Schema Initialization ────────────────────────────────────────────────────

def init_database(db_path: Path | str | None = None, force_recreate: bool = False) -> None:
    """Initialize database tables, catalogs, and indexes."""
    path = Path(db_path or DATABASE_PATH)
    if force_recreate and path.exists():
        logger.warning(f"Recreating database at {path}")
        path.unlink()

    ensure_directories()
    conn = get_db_connection(path)
    cursor = conn.cursor()

    try:
        cursor.execute(DDL_ACCIDENTS_INDIA)
        for idx in DDL_INDEXES_ACCIDENTS_INDIA:
            cursor.execute(idx)

        cursor.execute(DDL_OPENCITY_HIGHWAY)
        cursor.execute(DDL_DATASET_CATALOG)
        conn.commit()
        logger.info(f"Database schema initialized successfully at {path}")
    finally:
        conn.close()


# ── Data Loading Pipeline ────────────────────────────────────────────────────

def load_data_to_database(db_path: Path | str | None = None) -> Dict[str, int]:
    """
    Load cleaned data into the SQLite database.
    Idempotent: replaces existing table records safely.
    """
    path = Path(db_path or DATABASE_PATH)
    init_database(path)

    conn = get_db_connection(path)
    counts = {}

    # 1. Load Cleaned Indian Roads Dataset
    cleaned_csv = PROCESSED_DIR / "indian_roads_cleaned.csv"
    if not cleaned_csv.exists():
        # On a fresh clone / public deployment the source datasets are intentionally
        # not shipped (their licences do not permit redistribution). Degrade to an
        # empty-but-valid schema so every page renders a clear empty state instead
        # of crashing. Obtain the dataset from its documented source for full analytics.
        from src.config import INDIAN_ACCIDENTS_FILE

        if not INDIAN_ACCIDENTS_FILE.exists():
            logger.warning(
                "Source dataset %s is unavailable; database will remain empty. "
                "The application runs in empty/demo mode until the dataset is provided.",
                INDIAN_ACCIDENTS_FILE,
            )
            conn.commit()
            conn.close()
            return counts

        from src.data_cleaning import clean_indian_roads_dataset
        df_ind, _ = clean_indian_roads_dataset()
    else:
        df_ind = pd.read_csv(cleaned_csv)

    # Rename columns to match SQLite schema
    column_mapping = {
        "date": "accident_date",
        "time": "accident_time",
    }
    df_ind_db = df_ind.rename(columns=column_mapping)
    
    # Ensure correct columns exist
    df_ind_db.to_sql("accidents_india", conn, if_exists="replace", index=False)
    counts["accidents_india"] = len(df_ind_db)
    logger.info(f"Loaded {len(df_ind_db):,} records into 'accidents_india'.")

    # Re-apply indexes after replace
    cursor = conn.cursor()
    for idx in DDL_INDEXES_ACCIDENTS_INDIA:
        cursor.execute(idx)

    # 2. Load OpenCity highway records if available
    oc_path = RAW_DATA_DIR / "Accdataset_hk_PS_BAEL_Combined.csv"
    if oc_path.exists():
        try:
            df_oc = pd.read_csv(oc_path)
            # Map essential columns
            oc_mapped = pd.DataFrame({
                "accident_index": df_oc.get("Accident_Index"),
                "accident_date": df_oc.get("Date").astype(str),
                "day_of_week": df_oc.get("Day_of_Week"),
                "time_of_accident": df_oc.get("Time_of_Accident").astype(str),
                "location_code": df_oc.get("Accident_Location_A"),
                "chainage_km": pd.to_numeric(df_oc.get("Accident_Location_A_Chainage_km"), errors="coerce"),
                "road_side": df_oc.get("Accident_Location_A_Chainage_km_RoadSide"),
                "nature_of_accident": df_oc.get("Nature_of_Accident_B1").astype(str),
                "severity_code": df_oc.get("Accident_Severity_C"),
                "cause_code": df_oc.get("Causes_D1").astype(str),
                "road_condition": df_oc.get("Road_Condition_F"),
                "intersection_type": df_oc.get("Intersection_Type_G"),
                "weather_condition": df_oc.get("Weather_Conditions_H"),
                "vehicles_involved": df_oc.get("Number_of_Vehicles"),
                "casualties_fatal": df_oc.get("Number_of_Casualties_Fatel"),
                "casualties_grievous": df_oc.get("Number_of_Casualties_GrievousInjury"),
                "casualties_minor": df_oc.get("Number_of_Casualties_MinorInjury"),
                "casualties_total": df_oc.get("Number_of_Casualties"),
            })
            oc_mapped.to_sql("highway_accidents_opencity", conn, if_exists="replace", index=False)
            counts["highway_accidents_opencity"] = len(oc_mapped)
            logger.info(f"Loaded {len(oc_mapped):,} records into 'highway_accidents_opencity'.")
        except Exception as e:
            logger.warning(f"Could not load OpenCity records: {e}")

    # 3. Update Dataset Catalog
    cursor.execute("DELETE FROM dataset_catalog;")
    cursor.execute("""
        INSERT INTO dataset_catalog (table_name, source_name, description, row_count)
        VALUES 
        ('accidents_india', 'Indian Road Accident Dataset (2022-2025)', 'Standardized urban and highway road accidents across 8 Indian cities', ?),
        ('highway_accidents_opencity', 'OpenCity India 2023', 'Highway corridor chainage accident records', ?);
    """, (counts.get("accidents_india", 0), counts.get("highway_accidents_opencity", 0)))

    conn.commit()
    conn.close()
    return counts


# ── Query Executor ───────────────────────────────────────────────────────────

def execute_query(
    sql: str,
    params: Optional[tuple | list] = None,
    db_path: Path | str | None = None
) -> pd.DataFrame:
    """
    Execute a parameterized SQL query safely against the SQLite database.
    Rejects any destructive DDL/DML statements to guarantee read-only safety.
    """
    # Sanitize: Reject destructive statements
    sql_stripped = sql.strip().upper()
    prohibited_keywords = ["DROP ", "DELETE ", "TRUNCATE ", "ALTER ", "UPDATE ", "INSERT ", "REPLACE "]
    for kw in prohibited_keywords:
        if sql_stripped.startswith(kw) or f";{kw}" in sql_stripped:
            raise ValueError(f"Dangerous operation detected ({kw.strip()}). Only SELECT queries are permitted.")

    path = Path(db_path or DATABASE_PATH)
    if not path.exists():
        # Auto-create and populate database if missing
        logger.info(f"Database at {path} not found. Automatically initializing...")
        load_data_to_database(path)

    conn = get_db_connection(path, read_only=True)
    try:
        if params:
            df = pd.read_sql_query(sql, conn, params=params)
        else:
            df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


def get_database_summary(db_path: Path | str | None = None) -> Dict[str, Any]:
    """Retrieve metadata, table names, row counts, and schema definitions."""
    path = Path(db_path or DATABASE_PATH)
    if not path.exists():
        load_data_to_database(path)

    conn = get_db_connection(path, read_only=True)
    cursor = conn.cursor()

    try:
        # Tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cursor.fetchall()]

        table_info = {}
        for t in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {t};")
            count = cursor.fetchone()[0]
            cursor.execute(f"PRAGMA table_info({t});")
            columns = [{"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in cursor.fetchall()]
            table_info[t] = {
                "row_count": count,
                "columns": columns,
            }

        # Indexes
        cursor.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';")
        indexes = [{"name": r[0], "table": r[1]} for r in cursor.fetchall()]

        return {
            "database_path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "tables": table_info,
            "indexes": indexes,
        }
    finally:
        conn.close()


def check_database_integrity(db_path: Path | str | None = None) -> Dict[str, Any]:
    """Execute PRAGMA integrity_check to verify SQLite database file health."""
    path = Path(db_path or DATABASE_PATH)
    if not path.exists():
        return {"is_ok": False, "error": "Database file does not exist", "tables": []}

    conn = get_db_connection(path, read_only=True)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()
        is_ok = bool(res and res[0] == "ok")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cursor.fetchall()]
        return {"is_ok": is_ok, "integrity_result": res[0] if res else "unknown", "tables": tables}
    finally:
        conn.close()


if __name__ == "__main__":
    init_database(force_recreate=True)
    results = load_data_to_database()
    print("Database loading summary:")
    print(results)
    print("\nDatabase Schema Summary:")
    summary = get_database_summary()
    for tbl, info in summary["tables"].items():
        print(f"Table '{tbl}': {info['row_count']:,} rows, {len(info['columns'])} columns")
