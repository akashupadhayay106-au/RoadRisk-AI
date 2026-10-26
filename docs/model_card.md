# RoadRisk AI — Comprehensive Model Card

## 1. Severity Classification Model
- **Task**: Multi-class tabular classification predicting crash severity (`Slight Injury`, `Serious Injury`, `Fatal Injury`).
- **Primary Algorithm**: Tuned Balanced Random Forest / LightGBM Ensemble with scikit-learn Pipeline.
- **Artifact**: `models/severity_pipeline.joblib`
- **Features Used (Pre-Event Only)**:
  - Driver demographics (`Age_band_of_driver`, `Sex_of_driver`, `Driving_experience`)
  - Vehicle & infrastructure (`Type_of_vehicle`, `Area_accident_occured`, `Lanes_or_Medians`, `Road_allignment`, `Types_of_Junction`)
  - Environmental conditions (`Light_conditions`, `Weather_conditions`, `Road_surface_conditions`)
- **Strict Leakage Exclusions**: Post-event outcomes (`Number_of_casualties`, `Number_of_vehicles_involved`, casualty injury severity) are explicitly omitted.
- **Performance**:
  - Balanced Accuracy: ~72–75%
  - Macro-F1: ~0.68–0.72
- **Intended Use**: Proactive risk estimation for highway safety audits and infrastructure prioritization.

## 2. CCTV Computer Vision Model
- **Task**: Binary image classification on CCTV camera frames (`Accident` vs `Non Accident`).
- **Architecture**: MobileNetV2 with ImageNet-1K pretrained feature extractor and customized dropout head.
- **Artifact**: `models/cctv_mobilenet_v2.pth`
- **Performance**:
  - Test Accuracy: 81.0% (Naive Split) vs Baseline: 65.0%
  - Macro-F1: 0.8091
  - Latency: ~35–50 ms on CPU (~20–28 FPS)
- **Limitations & Disclaimers**:
  - Frame-level classification is offline experimental analysis, not certified real-time accident detection.
  - Video sequence audit revealed that naive splitting causes background memorization (**52/58 sequence groups; 928/989 frames = 93.83%** from leaking groups). Leakage-free grouped evaluation provides true generalization baseline.

## 3. Geospatial Hotspot Clustering Model
- **Task**: Unsupervised spatial density clustering identifying accident blackspots.
- **Algorithms**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) using Haversine distance metric, with comparative HDBSCAN and KDE.
- **Parameters**: `eps=1.5 km` (0.0135 radians), `min_samples=10`.
- **Intended Use**: Geographic hazard ranking for municipal planning and high-risk corridor identification.
