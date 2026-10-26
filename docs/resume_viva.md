# RoadRisk AI — Truthful Resume Bullets, Viva Defense & Interview Q&A

## 1. Truthful, Measured Resume Bullets

### Role: Machine Learning & Data Science Project (RoadRisk AI)
- Engineered an end-to-end road traffic risk intelligence platform across **10 modules** in Python 3.12 and Streamlit, integrating multi-source relational SQLite storage, geospatial clustering, atmospheric intelligence, computer vision, and grounded GenAI.
- Designed a supervised accident severity classification pipeline on Kaggle traffic data; enforced strict pre-event feature filtering to **prevent target leakage** (eliminating 7 post-crash casualty outcome fields) and mitigated 84:14:2 class imbalance using balanced class weights and stratified splitting.
- Implemented edge-optimized CCTV accident classification via **MobileNetV2** transfer learning (~9.2 MB footprint, ~35–50 ms CPU inference); generated **Grad-CAM** visual attention heatmaps to localize vehicle collision zones.
- Audited surveillance video dataset splits and identified **cross-split sequence leakage affecting 93.83% of all frames (52/58 video groups)**; engineered a prefix-based grouping heuristic and GroupShuffleSplit generator (`cctv_grouped_manifest.csv`) to establish realistic out-of-sample generalization.
- Developed a dual-mode **Grounded GenAI Assistant** over a pluggable LLM provider layer (Google Gemini via the current `google-genai` SDK, Groq, Hugging Face) with timeouts and automatic fallback to an offline **BM25 RAG engine** indexing local Markdown documentation; prevented prompt injection and restricted database operations to parameterized, read-only SQL queries.
- Architected comprehensive automated QA suite with **167 unit, integration, UI-smoke and API-mocking tests across 13 phases** achieving 100% pass rate, coupled with static analysis (`compileall`) with zero compilation warnings.

---

## 2. Technical Viva Defense Questions & Answers

### Q1: What is target leakage in accident severity prediction, and how did you prevent it?
**Answer**:
Target leakage occurs when features containing information only observable *after* the target event has occurred are included in the predictor feature set. In accident severity modeling, attributes such as `Casualty_severity`, `Casualty_class`, `Hospital_treated`, and post-investigation police codes are direct consequences of the crash. Including them produces deceptively high model accuracy (~99%) during training that collapses completely when deployed in proactive pre-crash risk estimation. I addressed this by establishing a strict hardcoded exclusion allowlist in `src/features.py` that permits only pre-event environmental, roadway, and driver attributes.

### Q2: Why did you audit video sequence leakage in the CCTV dataset, and what were the findings?
**Answer**:
In surveillance video datasets, consecutive frames extracted from the same video share nearly identical background asphalt, buildings, lighting, and camera mounting geometry. If frames from sequence `acc1` or `test10` are randomly distributed across `train`, `val`, and `test` splits, the neural network easily memorizes the camera background rather than learning physical vehicle deformation or collision dynamics.
My empirical audit revealed that **52 out of 58 sequence groups (93.83% of all frames) crossed train/val/test boundaries** in the naive Kaggle folder split. While the naive split yielded an inflated test accuracy of 81.0%, evaluating with our leakage-free grouped split provides the true out-of-sample generalization baseline (~65–72%), proving the necessity of grouped temporal validation. (Grouping uses the composite `(class_name, video_group)` key, because the same raw video ID can appear in both the Accident and Non Accident folders.)

### Q3: Why is UrbanGB kept separate from Indian road accident tables?
**Answer**:
UrbanGB is an open benchmark dataset derived from the UK Department for Transport (STATS19) road safety records, with coordinates situated exclusively within Great Britain (~49.8°N to 60.9°N, -8.6°W to 1.8°E). Merging British records into Indian road tables would violate basic geospatial and demographic validity, generating fictitious blackspots in the Arabian Sea or Indian Ocean. In RoadRisk AI, UrbanGB is isolated in a dedicated SQLite table (`urbangb_accidents`) and used strictly as an international spatial clustering benchmark to validate DBSCAN and HDBSCAN algorithms.

### Q4: How does the AI Assistant prevent SQL injection and prompt manipulation?
**Answer**:
The AI Assistant enforces three defensive layers:
1. **Prompt Sanitization**: User inputs are scanned against an adversarial regex pattern catalog that immediately detects and blocks jailbreak signatures ("ignore all previous instructions", "system prompt", "reveal api key").
2. **Read-Only Parameterized Execution**: Database queries are strictly confined to predefined templates with parameterized bindings (`?` or `:param`). Any query containing DDL or DML keywords (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`) raises an immediate `ValueError`.
3. **Deterministic Offline Grounding**: The assistant tries configured LLM providers (Gemini, Groq, Hugging Face) in preference order; when no provider key is configured or when an external API fails or times out, it falls back to a local BM25 RAG index over project Markdown documentation and parameter-validated analytical functions, ensuring zero hallucinated citations or counts. Provider status indicators expose only whether a service is configured/available, never the key.

### Q5: How do you interpret feature importance without claiming physical causality?
**Answer**:
Tree-based feature importance (such as Gini impurity decrease or permutation importance) measures how much a feature improves the decision tree's mathematical split criterion within the training sample. It demonstrates statistical association, not counterfactual causation. For example, `Hour` is a top predictive feature for severity, but altering the clock does not prevent crashes; `Hour` is a proxy for driver fatigue, reduced night visibility, and alcohol consumption. I prominently feature an explicit causal disclaimer across the UI and documentation: *Feature Importance ≠ Physical Causality*.
