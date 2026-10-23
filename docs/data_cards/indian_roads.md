# Dataset Card: Indian Roads Accident Dataset (MoRTH / Open Data)

## 1. Overview & Provenance
- **Dataset Name**: Indian Roads Accident Dataset
- **Primary Raw File**: `data/raw/Accdataset_hk_PS_BAEL_Combined.csv`
- **Cleaned Target**: SQLite Table `indian_accidents` (`data/processed/roadrisk.db`)
- **Geographic Scope**: Republic of India (National Highway & State Highway network)
- **Coordinate Envelope**: Latitude [8.0°N to 37.0°N], Longitude [68.0°E to 97.5°E]
- **Target Use**: Spatial accident frequency visualization, state-level KPI benchmarking, time-of-day hazard analysis.

## 2. Schema & Key Fields
- `accident_id` (Primary Key): Unique numeric identifier per incident report.
- `date` & `time`: Incident timestamp. Parsed and categorized into `time_of_day` (Morning Rush, Afternoon, Evening Rush, Night).
- `state` & `city`: Administrative jurisdiction in India.
- `latitude` & `longitude`: Geocoded coordinates of the crash location.
- `severity`: Categorical classification (`Fatal`, `Grievous Injury`, `Minor Injury`, `Non-Injury`).
- `vehicles_involved`: Count of vehicles involved in collision.
- `collision_type`: Collision mechanism (Head-on, Rear-end, Side-swipe, Rollover, Pedestrian impact).
- `weather_condition`: Atmospheric conditions reported by investigating officer (Clear Sky, Rain, Fog/Mist).

## 3. Data Cleaning & Validation Rules
1. **Coordinate Verification**: Any record falling outside India's geographic bounding box [8.0°N–37.0°N, 68.0°E–97.5°E] is flagged as invalid and excluded from spatial clustering.
2. **Missing Value Imputation**: Missing numeric vehicle counts default to 1; missing timestamps are mapped to "Unknown".
3. **Deduplication**: Records matching on (date, latitude, longitude, collision_type) are deduplicated.

## 4. Ethical Considerations & Limitations
- **Reporting Bias**: Incidents occurring on major highways are logged with higher fidelity than rural district roads.
- **Exposure Distortion**: Raw accident counts correlate strongly with vehicle population and traffic volume. High accident counts in Maharashtra or Tamil Nadu reflect heavy vehicle exposure, not necessarily higher per-vehicle risk.
- **Educational Scope**: Not for direct emergency response or automated police dispatch.
