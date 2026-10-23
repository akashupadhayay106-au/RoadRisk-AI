"""
RoadRisk AI — Data Validation
===============================
Validation functions for data quality checks.
Full implementation in Phase 2.
"""


def validate_coordinates(lat: float, lon: float) -> bool:
    """Check if latitude/longitude are within valid ranges."""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def validate_date_string(date_str: str, fmt: str = "%Y-%m-%d") -> bool:
    """Check if a string is a valid date in the given format."""
    from datetime import datetime
    try:
        datetime.strptime(date_str, fmt)
        return True
    except (ValueError, TypeError):
        return False
