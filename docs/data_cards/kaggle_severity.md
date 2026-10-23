# Dataset Card: Kaggle Road Accident Severity Classification Dataset

## 1. Overview & Provenance
- **Dataset Name**: Road Traffic Accident Severity Dataset
- **Primary Raw File**: `data/raw/train.csv` / `data/raw/cleaned.csv`
- **Cleaned Target**: SQLite Table `severity_accidents` (`data/processed/roadrisk.db`)
- **Task**: Multi-class supervised classification predicting `Accident_severity` (Slight Injury, Serious Injury, Fatal Injury).
- **Target Distribution**: Highly imbalanced (~84% Slight Injury, ~14% Serious Injury, ~2% Fatal Injury).

## 2. Leakage Prevention Protocol (Mandatory ML Rules)
- **Target Leakage Prohibition**: All post-event outcome attributes must be rigorously excluded from model feature sets:
  - Excluded: `Number_of_casualties`, `Number_of_vehicles_involved`, `Casualty_severity`, `Casualty_class`, `Hospital_treated`.
  - Excluded: Police report administrative IDs and post-crash investigation flags.
- **Pre-Event Feature Allowlist**:
  - `Day_of_week`, `Age_band_of_driver`, `Sex_of_driver`, `Educational_level`, `Vehicle_driver_relation`, `Driving_experience`
  - `Type_of_vehicle`, `Owner_of_vehicle`, `Area_accident_occured`, `Lanes_or_Medians`, `Road_allignment`, `Types_of_Junction`
  - `Road_surface_type`, `Road_surface_conditions`, `Light_conditions`, `Weather_conditions`

## 3. Class Imbalance Mitigation & Evaluation Standards
- Standard accuracy is deceptive due to 84% majority class dominance.
- Models must be evaluated using **Macro-Precision, Macro-Recall, Macro-F1, and Balanced Accuracy**.
- Class weighting (`class_weight='balanced'`) and stratified train-test splitting (80/20) are strictly enforced.
