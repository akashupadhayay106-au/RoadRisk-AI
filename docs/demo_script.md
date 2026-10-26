# RoadRisk AI — 5-Minute Portfolio Walkthrough & Demo Script

## Persona & Setup
- **Presenter**: MCA First-Year Data Science Student / ML Engineer
- **Audience**: Technical Hiring Managers, Data Science Mentors, Viva Examiners
- **Prerequisite**: Start local server with `streamlit run app.py`

---

## Demo Script Walkthrough

### 0:00 – 0:45 | Introduction & Executive Vision (Home & Dashboard)
- **Visual**: Open `app.py` Home Page, then navigate to `1_Dashboard.py`.
- **Talking Points**:
  - *"Welcome! RoadRisk AI is an end-to-end accident intelligence and decision-support platform designed to address road mortality across Indian highways."*
  - *"Instead of a static notebook, I engineered a full relational data pipeline in SQLite holding multi-source accident logs, weather telemetry, and computer vision feeds."*
  - *"Notice the executive KPI ribbon: total incidents, casualty severity, and our normalized risk score."*

### 0:45 – 1:45 | SQL Analytics & Data Governance (SQL Studio & Hotspot Map)
- **Visual**: Navigate to `3_SQL_Analytics.py` and run a parameterized query; then show `5_Hotspot_Map.py`.
- **Talking Points**:
  - *"Here in SQL Analytics, every query uses parameterized bindings with a strict read-only allowlist preventing SQL injection."*
  - *"In Hotspot Maps, I applied DBSCAN clustering with the Haversine metric to detect accident blackspots in Mumbai and Delhi without arbitrary grid distortion."*
  - *"Crucially, I maintained strict scientific integrity: UrbanGB data is from Great Britain and is strictly benchmarked separately from Indian data."*

### 1:45 – 2:45 | Machine Learning & Causal Explainability (Severity & XAI)
- **Visual**: Navigate to `4_Severity_Prediction.py`, adjust sliders, then click `9_Model_Explainability.py`.
- **Talking Points**:
  - *"In Severity Prediction, I built a balanced Random Forest pipeline. A critical engineering triumph was target leakage prevention: I excluded post-event attributes like casualty injury severity and hospital codes to prevent synthetic metric inflation."*
  - *"In Explainability Studio, we inspect global Gini feature importances. However, notice my explicit causal disclaimer: Feature importance represents statistical correlation in police logs, not physical causality."*

### 2:45 – 3:45 | CCTV Computer Vision & Sequence Leakage Audit
- **Visual**: Navigate to `7_CCTV_Detection.py`, select an accident frame, show Grad-CAM, then show Tab 2 (Sequence Leakage Audit).
- **Talking Points**:
  - *"In the CCTV module, I fine-tuned MobileNetV2 with ImageNet transfer learning, achieving sub-50ms CPU edge inference and Grad-CAM visual attention overlays."*
  - *"Most importantly, I audited the Kaggle dataset and discovered a major flaw: 93.83% of all frames (52 of 58 sequence groups) belonged to video sequences that leaked across train, val, and test splits! I implemented a sequence grouping heuristic and generated a leakage-free grouped manifest to establish an honest generalization benchmark."*

### 3:45 – 4:30 | Grounded GenAI Assistant
- **Visual**: Navigate to `8_AI_Assistant.py`, click 'Top 5 States by Accidents' or 'Explain CCTV Model'.
- **Talking Points**:
  - *"The conversational assistant runs on a pluggable LLM provider layer — Google Gemini, Groq, or Hugging Face — paired with an offline BM25 RAG engine. The status pills show only whether each provider is configured and available, never the key itself."*
  - *"Even without any API key, the offline engine answers questions deterministically using our project documentation and parameterized SQL tools, completely preventing hallucination and blocking prompt injection."*

### 4:30 – 5:00 | QA & Engineering Rigor Conclusion
- **Talking Points**:
  - *"The codebase is backed by 167 passing unit, integration, UI-smoke and API-mocking tests across 13 phases, with zero warnings on clean Python compilation."*
  - *"Thank you! I welcome any questions regarding the architecture, ETL, or modeling choices."*
