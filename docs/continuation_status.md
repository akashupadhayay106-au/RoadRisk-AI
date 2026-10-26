# RoadRisk AI — Project Continuation & Final Milestone Status

**Status Date**: October 24, 2026  
**Environment**: Python 3.12.7 (Windows x86_64, PowerShell)  
**Total Tests**: **167 Passed** / 0 Failed (100% Pass Rate across 13 Phases)

---

## 1. Milestone Status Summary (Milestones 0 – 12)

| Milestone | Title | Status | Summary & Verification |
| :---: | :--- | :---: | :--- |
| **M-00** | Project Scoping & Problem Framing | **COMPLETE** | Identified educational problem statement, target users, multi-source data boundaries. |
| **M-01** | Architecture & Repo Initialization | **COMPLETE** | Established directory tree, `config.py`, `utils.py`, dark UI design system, Streamlit skeleton (`app.py`). |
| **M-02** | Empirical Data Audit & Bounding Checks | **COMPLETE** | Audited Indian Roads, Kaggle Severity, UrbanGB, OpenCity, and CCTV datasets; established coordinate boundaries. |
| **M-03** | Data Cleaning & Relational SQLite ETL | **COMPLETE** | Reproducible ETL pipeline in `src/data_cleaning.py`; normalized `data/processed/roadrisk.db` with indexed tables and read-only query wrappers. |
| **M-04** | Executive Dashboard & Interactive Analytics | **COMPLETE** | Multi-tab executive dashboard (`1_Dashboard.py`), faceted search explorer (`2_Data_Explorer.py`), and parameterized SQL studio (`3_SQL_Analytics.py`). |
| **M-05** | Accident Severity ML Classification | **COMPLETE** | Implemented balanced Random Forest pipeline (`models/severity_pipeline.joblib`); eliminated 7 post-crash target leakage fields; evaluated on macro-F1. |
| **M-06** | Geospatial Hotspot Analysis | **COMPLETE** | Implemented DBSCAN with Haversine distance metric (`eps=1.5km`); interactive Mapbox density heatmaps (`5_Hotspot_Map.py`); UrbanGB UK benchmark. |
| **M-07** | Historical Weather Risk Intelligence | **COMPLETE** | Open-Meteo API client with SQLite caching (`weather_cache` table); WMO weather interpretation; weather risk multipliers (`6_Weather_Analysis.py`). |
| **M-08** | CCTV Computer Vision & Sequence Audit | **COMPLETE** | Fine-tuned MobileNetV2 (`models/cctv_mobilenet_v2.pth`); Grad-CAM overlays; audited sequence split leakage (**52/58 sequence groups, 928/989 frames = 93.83% leaking** under the composite `(class_name, video_group)` key); generated `cctv_grouped_manifest.csv` (0 leaking groups; train 769 / test 127 / val 93); baseline model benchmark; video stream frame sampling (`7_CCTV_Detection.py`). |
| **M-09** | Grounded GenAI Safety Assistant | **COMPLETE** | Dual-engine assistant (`src/ai_assistant.py`, `src/rag_pipeline.py`); pluggable provider layer — Gemini (current `google-genai` SDK), Groq, Hugging Face — with timeouts, error isolation and automatic fallback to an offline BM25 RAG engine; prompt injection defense; parameterized SQL tools; secret-safe provider status indicators (`8_AI_Assistant.py`). |
| **M-10** | Complete QA & Security Hardening | **COMPLETE** | Clean `compileall` (zero warnings); explainability suite (`src/explainability.py`, `9_Model_Explainability.py`); 167 passing pytest tests; `docs/qa_report.md`. |
| **M-11** | GitHub & Streamlit Cloud Deployment Prep | **COMPLETE** | Clean `.gitignore`; zero secret exposure; `.env.example`; `docs/deployment_checklist.md` with step-by-step push/deploy commands; local commit staged. |
| **M-12** | Final Portfolio Documentation & Packaging | **COMPLETE** | Full documentation package (`README.md`, `architecture.md`, `discovery_report.md`, data cards, model card, risk register, demo script, resume & viva defense); clean portfolio ZIP built and verified. |

---

## 2. Actual Model Training Status & Real Metrics

1. **Severity Classification Model**:
   - **Algorithm**: Balanced Random Forest with Scikit-learn Pipeline (`models/severity_pipeline.joblib`).
   - **Target**: `Accident_severity` (3 classes: Slight, Serious, Fatal).
   - **Exclusions**: 7 post-event outcome attributes strictly omitted (no casualty injury severity leakage).
   - **Performance**: Balanced Accuracy ~73.5%, Macro-F1 ~0.704.

2. **CCTV Accident Detection Model**:
   - **Algorithm**: MobileNetV2 with ImageNet-1K transfer learning (`models/cctv_mobilenet_v2.pth`).
   - **Naive Test Set (101 frames)**: Accuracy: **81.0%**, Macro-F1: **0.8091**, ROC-AUC: **0.8266**, Latency: **~35–50 ms** on CPU.
   - **Baseline Benchmark**: Color Histograms + 32x32 Spatial Thumbnails + Logistic Regression achieved **65.0%** Accuracy and **0.6457** Macro-F1 (`models/cctv_baseline_metrics.json`).
   - **Sequence Leakage Audit**: Empirical audit confirmed **52/58 sequence groups (928/989 frames = 93.83%)** cross naive splits under the composite `(class_name, video_group)` key; generated `data/processed/cctv_grouped_manifest.csv` for leakage-free grouped evaluation (0 leaking groups).

---

## 3. Deployment & Git Status
- **Local Working Tree**: Cleanly tracked and committed with descriptive commit messages.
- **Git Remote**: Configured — `https://github.com/akashupadhayay106-au/RoadRisk-AI.git`.
- **History**: Pre-existing remote history reconciled via `git merge origin/main --allow-unrelated-histories` (merge commit `1266026`); no force-push, no rewrite, original `LICENSE` attribution preserved.
- **GitHub Push**: Requires this machine to be authenticated with the owner's GitHub account.
- **Live Cloud URL**: Not attempted (pending an authenticated push and the Streamlit Cloud deploy).
- **Next User Action**: Follow `docs/deployment_checklist.md` to push and deploy on Streamlit Community Cloud.
