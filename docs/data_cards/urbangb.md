# Dataset Card: UrbanGB Road Traffic Incident Benchmark (Great Britain)

## 1. Critical Geographic Distinction
> [!IMPORTANT]
> **UrbanGB is Great Britain data and must NEVER be labelled or aggregated as Indian accident data.**
> Coordinates fall exclusively within the United Kingdom:
> - Latitude: ~49.8°N to 60.9°N
> - Longitude: ~-8.6°W to 1.8°E

## 2. Dataset Overview & Benchmark Purpose
- **Source**: UK Department for Transport (DfT) STATS19 Road Safety Data / UCI Machine Learning Repository UrbanGB Benchmark.
- **Raw Location**: `data/raw/urbanGB.tra` / `data/raw/urbanGB.tes`
- **Cleaned Target**: SQLite Table `urbangb_accidents` (`data/processed/roadrisk.db`)
- **Key Attributes**: Easting, Northing, WGS84 Latitude, WGS84 Longitude, Urban Density Index, Incident Cluster Label.
- **Role in RoadRisk AI**: Serves as a validated, high-density spatial benchmark for verifying DBSCAN and HDBSCAN geospatial density clustering algorithms against established European urban traffic standards.

## 3. Data Integration Constraints
- **Zero Join Policy**: UrbanGB records cannot be joined or merged with Indian accident tables.
- **Separate Pipeline**: Maintained in an independent database table and analyzed strictly in benchmark comparison modules.
