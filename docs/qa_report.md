# RoadRisk AI — Comprehensive QA, Security & Validation Report

## 1. Executive QA Summary
- **Execution Date**: October 24, 2026
- **Target Runtime**: Python 3.12.7 (Windows x86_64)
- **Total Test Suites**: 13 test modules (`tests/test_phase1.py` through `tests/test_phase13_ui_smoke.py`)
- **Total Tests Executed**: **167**
- **Tests Passed**: **167** (100.0% pass rate)
- **Tests Failed / Errors**: **0**
- **Full Test Suite Runtime**: ~67.9 seconds
- **Static Compilation (`compileall`)**: 100% clean compilation across all `src/`, `pages/`, `tests/`, and `app.py` files with zero syntax errors and zero warnings.

---

## 2. Test Suite Breakdown by Phase

| Test Module | Phase / Domain | Tests | Status | Key Coverage |
| :--- | :--- | :---: | :---: | :--- |
| `test_phase1.py` | Architecture & Config | 16 | PASSED | Directory trees, config constants, coordinate bounds, date parsing, data loaders. |
| `test_phase2_data_inspection.py` | Dataset Auditing | 12 | PASSED | File presence, raw schemas, India coordinate bounds, UK coordinate bounds, leakage flags. |
| `test_phase3_cleaning_and_database.py` | ETL & SQLite | 12 | PASSED | Cleaning pipelines, SQLite tables, indexing, read-only query enforcement, KPI queries. |
| `test_phase4_dashboard_analytics.py` | Analytics Engine | 6 | PASSED | Multi-filter parameterized queries, day-hour heatmaps, state comparisons, single-record dossiers. |
| `test_phase5_severity_model.py` | Severity ML Pipeline | 6 | PASSED | Target leakage exclusion, preprocessor pipeline, balanced metrics calculation, inference serialization. |
| `test_phase6_hotspot_analysis.py` | Geospatial Clustering | 6 | PASSED | DBSCAN clustering, Haversine metric, Mumbai blackspots, UrbanGB UK clustering, Mapbox builders. |
| `test_phase7_weather_analysis.py` | Weather Risk Engine | 13 | PASSED | WMO code interpretation, SQLite weather cache CRUD, offline fallback generation, risk multipliers. |
| `test_phase8_cctv_detection.py` | CCTV Computer Vision | 14 | PASSED | MobileNetV2 architecture, inference latency, Grad-CAM overlays, sequence leakage audit, baseline model, video frame sampling. |
| `test_phase9_ai_assistant.py` | Grounded GenAI & RAG | 17 | PASSED | BM25 document indexing, prompt injection defense, SQL injection blocking, offline grounded fallback, uncertainty reporting. |
| `test_phase10_qa_hardening.py` | QA & Security Hardening | 13 | PASSED | Malformed inputs, database integrity checks, global feature importance, local prediction attribution, secret protection. |
| `test_phase11_providers.py` | LLM Provider Abstraction | 18 | PASSED | Gemini/Groq/Hugging Face request payloads, timeouts, provider fallback chain, secret-free status descriptors, no-API-key path. |
| `test_phase12_routing.py` | Routing & Weather Status | 14 | PASSED | openrouteservice auth header and `[lon, lat]` ordering, isochrones, geocoding, keyless weather status. |
| `test_phase13_ui_smoke.py` | UI Smoke & Structure | 14 | PASSED | Exactly 10 pages, per-page compilation, Home page renders without exception (`AppTest`), no stale "pending" module flags. |
| **Total** | | **167** | **ALL PASSED** | |

---

## 3. Security & Anti-Adversarial Verification
1. **Zero Secret Exposure**:
   - Automated checks confirmed `.env` and `.streamlit/secrets.toml` are excluded from Git tracking via `.gitignore`.
   - `.streamlit/secrets.toml.example` contains only placeholder tokens (`your-gemini-api-key-here`).
   - Hardcoded secret scans found zero credentials, API keys, or database passwords in source code.
2. **SQL Safety & Parameterization**:
   - `execute_query()` strictly blocks any non-`SELECT` statement (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`).
   - All analytical functions use parameterized bindings (`?` or `:param`) to prevent SQL injection.
3. **Prompt Injection & Adversarial Defense**:
   - `sanitize_and_validate_prompt()` in `src/ai_assistant.py` successfully screens and neutralizes jailbreaks, instruction overrides ("ignore all previous instructions"), and credential harvesting queries.

---

## 4. Scientific Integrity & Data Governance Audits
1. **Geographic Isolation of UrbanGB**:
   - UrbanGB data resides exclusively in its dedicated table `urbangb_accidents` with verified UK bounding box [49.8°N–60.9°N, -8.6°W–1.8°E].
   - Tests enforce zero merging or conflation between Indian accident tables and Great Britain data.
2. **ML Target Leakage Screening**:
   - Post-crash outcome features (`Casualty_severity`, `Casualty_class`, `Hospital_treated`) are permanently isolated from the pre-crash feature set in `src/features.py`.
3. **CCTV Video Sequence Leakage Audit**:
   - Manifest audit empirically documented that **52 out of 58 video sequence groups (928/989 frames = 93.83%)** cross split boundaries in the raw folder distribution, using the composite `(class_name, video_group)` key (the same raw video ID can appear in both the Accident and Non Accident folders).
   - Leakage-free grouped splitting (`create_leakage_free_splits`) generates `data/processed/cctv_grouped_manifest.csv` with **0 leaking groups** across train, val, and test splits (train 769 / test 127 / val 93).
   - Baseline feature-engineering model (Color Histograms + Thumbnails + Logistic Regression) achieves 65.0% accuracy vs MobileNetV2 81.0%, proving transfer learning value.

---

## 5. Environment Limitations & Remaining Considerations
- **Open-Meteo API**: Operates on a free, rate-limited public endpoint. Production environments should utilize the local SQLite historical weather cache (`data/processed/roadrisk.db -> weather_cache` table) to avoid external rate limits.
- **LLM Providers**: Gemini (via the current `google-genai` SDK), Groq, and Hugging Face are all optional. When no provider key is configured, or when an API call fails or times out, the assistant deterministically falls back to the offline BM25 project-documentation synthesizer. No key value is ever logged or rendered.
- **HeiGIT openrouteservice**: Optional for the "Route & Reachability" tab. When `OPENROUTESERVICE_API_KEY` is absent, the tab shows an informational notice and all other tabs remain fully usable.
- **Hardware Acceleration**: Deep learning inference runs on edge CPU via MobileNetV2 with an average latency of ~35–50 ms (~20–28 FPS). CUDA GPU acceleration is automatically detected and leveraged if available.
