# Dataset Card: CCTV Surveillance Road Accident Dataset

## 1. Overview & Image Manifest
- **Dataset Name**: CCTV Highway Incident Surveillance Dataset
- **Directory Layout**: `data/train/`, `data/val/`, `data/test/`
- **Class Labels**: Binary classification — `Accident` (Class 0) vs `Non Accident` (Class 1).
- **Volume**: 990 total indexed frames:
  - `train`: 369 Accident, 422 Non Accident (791 frames)
  - `val`: 46 Accident, 52 Non Accident (98 frames)
  - `test`: 47 Accident, 54 Non Accident (101 frames)

## 2. Sequence Leakage Audit & Heuristic Grouping
- **Audit Findings**: In the naive raw distribution, **52 out of 58 video sequence groups (928/989 frames = 93.83%)** cross split boundaries, counted with the composite `(class_name, video_group)` key. Frames from identical video clips (e.g., `acc1 (x).jpg`, `test10_x.jpg`) appeared in both training and test sets.
- **Grouping Heuristic**: Extracted via prefix pattern `r"^([a-zA-Z0-9_\-]+?)(?:[\s\(]|_\d+$)"`.
- **Leakage-Free Protocol**: Generates `data/processed/cctv_grouped_manifest.csv` using GroupShuffleSplit to enforce zero video-group overlap across splits (0 leaking groups; train 769 / test 127 / val 93).

## 3. Modeling Benchmarks
- **Baseline Model**: Color Histograms + 32x32 Thumbnail Downsampling + Logistic Regression (Accuracy: ~65.0%, Macro-F1: 0.6457).
- **Deep Architecture**: MobileNetV2 (ImageNet-1K transfer learning, Dropout 0.3, linear classification head).
  - Test Accuracy (naive split): 81.0% | Macro-F1: 0.8091 | ROC-AUC: 0.8266
  - Edge Inference Latency: ~35–50 ms on CPU (~20–28 FPS).
