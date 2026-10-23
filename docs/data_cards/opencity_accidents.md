# Dataset Card: OpenCity Urban Accident Aggregate Tables

## 1. Overview & Data Structure
- **Dataset Name**: OpenCity Indian Metropolitan Traffic Statistics
- **Raw File**: `data/raw/road_accidents_india.csv` (and associated municipal summary tables)
- **Nature of Data**: Municipal and district-level **aggregate tables** (summary counts by year, city, vehicle category, and lighting condition).
- **Critical Aggregation Boundary**:
  > [!WARNING]
  > OpenCity aggregate tables must **NEVER** be treated as individual geocoded point accident records.
  > They represent tabular regional aggregations, not incident coordinates.

## 2. Intended Use
- City-level historical trend analysis (e.g., comparing annual fatalities in Mumbai vs Delhi vs Bengaluru).
- Macro-level vehicle category shares (two-wheelers vs cars vs commercial trucks).
- High-level validation of macro patterns against fine-grained geocoded samples.
