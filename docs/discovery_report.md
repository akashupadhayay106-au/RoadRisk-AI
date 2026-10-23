# RoadRisk AI — Empirical Data Discovery & Feasibility Report

## 1. Executive Summary
This report documents the empirical audit of the 5 candidate datasets evaluated during the project discovery phase, detailing coordinate validity, data fidelity, missing values, target leakage risks, and architectural routing.

---

## 2. Dataset Audits & Findings

### Dataset 1: Indian Roads Accident Dataset
- **File**: `data/raw/Accdataset_hk_PS_BAEL_Combined.csv`
- **Volume**: Multistate incident logs across Indian National and State Highways.
- **Bounding Box Validation**: Verified within India geographic envelope [8.0°N to 37.0°N, 68.0°E to 97.5°E]. Over 99% of geocoded coordinates reside cleanly within sovereign Indian territory.
- **Target Routing**: Loaded into SQLite table `accidents_india` for executive KPI tracking, multi-filter exploration, and DBSCAN blackspot identification.

### Dataset 2: Kaggle Traffic Accident Severity Dataset
- **File**: `data/raw/train.csv` / `data/raw/cleaned.csv`
- **Target**: `Accident_severity` (Slight Injury: ~84%, Serious Injury: ~14%, Fatal Injury: ~2%).
- **Leakage Screening**:
  - Found 7 severe post-event target leakage columns: `Casualty_severity`, `Casualty_class`, `Sex_of_casualty`, `Age_band_of_casualty`, `Hospital_treated`, `Pedestrian_movement`, `Work_of_casuality`.
  - Resolution: Hardcoded exclusion in `src/features.py` to prevent synthetic accuracy inflation.
- **Target Routing**: Loaded into SQLite table `severity_accidents` and used for training `models/severity_pipeline.joblib`.

### Dataset 3: UrbanGB Spatial Benchmark (Great Britain)
- **File**: `data/raw/urbanGB.tra` / `data/raw/urbanGB.tes`
- **Geographic Finding**: Coordinates fall between [49.8°N–60.9°N, -8.6°W–1.8°E] (United Kingdom).
- **Critical Policy**: Zero records from UrbanGB are merged with Indian road data. UrbanGB is maintained in table `urbangb_accidents` purely as an international spatial density clustering benchmark.

### Dataset 4: OpenCity Metropolitan Summary Tables
- **File**: `data/raw/road_accidents_india.csv`
- **Structure**: Aggregate municipal summaries by city and year (not point-level geocoded incidents).
- **Target Routing**: Used for macro-level metropolitan fatality benchmarking and validation.

### Dataset 5: CCTV Highway Surveillance Image Dataset
- **File**: `data/train/`, `data/val/`, `data/test/` (990 images)
- **Leakage Finding**: 52 out of 59 video sequence groups (93.7% of all frames) cross naive folder splits (e.g. `acc1 (x).jpg` appears in train, val, and test).
- **Mitigation**: Implemented sequence grouping heuristic and `data/processed/cctv_grouped_manifest.csv` using GroupShuffleSplit.
- **Target Routing**: Used for fine-tuning MobileNetV2 edge classifier (`models/cctv_mobilenet_v2.pth`) and Grad-CAM saliency heatmaps.
