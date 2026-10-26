# RoadRisk AI — Deployment Checklist & Hosting Guide

## 1. Pre-Deployment Secret & Hygiene Audit

Before pushing to any public git forge (GitHub, GitLab), verify that no credentials or private data are committed:

- [x] **Virtual Environment Ignored**: `.venv/` is excluded by `.gitignore`.
- [x] **Streamlit Secrets Ignored**: `.streamlit/secrets.toml` is excluded by `.gitignore`.
- [x] **Local Databases Ignored**: `*.db`, `*.sqlite` are excluded by `.gitignore`.
- [x] **Large CCTV Datasets Ignored**: `data/train/`, `data/val/`, `data/test/` are excluded by `.gitignore`.
- [x] **Model Weights**: un-trained/optional weights stay local, but the two required runtime
      artifacts are deliberately tracked so the deployed app works out of the box:
      `models/severity_pipeline.joblib` (page 4) and `models/cctv_mobilenet_v2.pth` (page 7).
- [x] **No Hardcoded Tokens**: Scanned all `.py` files for API key literals (zero findings).
- [x] **Example Template Provided**: `.streamlit/secrets.toml.example` contains placeholders only (`your-gemini-api-key-here`).

---

## 2. GitHub Repository & Push

The remote is already configured for this project:

```
origin  https://github.com/akashupadhayay106-au/RoadRisk-AI.git
```

Local `main` was safely reconciled with the pre-existing remote history (which contained only a
`LICENSE`) using `git merge origin/main --allow-unrelated-histories`; the original owner's
copyright attribution in `LICENSE` is preserved. No force-push or history rewrite was used.

To publish later changes:

```powershell
git push origin main
```

If the push is rejected for authentication, either sign in through Git Credential Manager when
prompted in the terminal, or configure an SSH remote / a credential helper. Never paste a token
into a chat or commit it to the repository.

---

## 3. Streamlit Community Cloud Deployment

Streamlit Community Cloud provides 100% free hosting for portfolio data apps with zero AWS/cloud costs.

### Step-by-Step Instructions:
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Fill in the repository configuration:
   - **Repository**: `akashupadhayay106-au/RoadRisk-AI`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `roadrisk-ai.streamlit.app` (or customizable subdomain).
4. Click **Advanced settings...**:
   - In **Python version**, select **3.12** (matches local development).
   - In **Secrets**, paste your optional configuration (all keys are optional — the app runs fully in offline/demo mode without them):
     ```toml
     # Optional: choose a preferred provider (gemini | groq | huggingface)
     LLM_PROVIDER = "gemini"

     # Optional: Google Gemini (current google-genai SDK)
     GEMINI_API_KEY = "your-actual-gemini-api-key"
     GEMINI_MODEL = "gemini-2.0-flash"

     # Optional: Groq Cloud
     # GROQ_API_KEY = "your-actual-groq-api-key"
     # GROQ_MODEL = "llama-3.3-70b-versatile"

     # Optional: Hugging Face inference
     # HF_TOKEN = "your-actual-hf-token"

     # Optional: HeiGIT openrouteservice (route & reachability)
     # OPENROUTESERVICE_API_KEY = "your-actual-ors-api-key"
     ```
   *(The app will run smoothly in grounded offline mode even without any key.)*
5. Click **Deploy!**
6. Streamlit Cloud will run `pip install -r requirements.txt` and launch the interactive dashboard within 2–3 minutes.

---

## 4. Data Availability on a Public Clone

The source datasets are intentionally **not** committed — their licences (Kaggle competition
rules, UCI OGL/ODbL terms) do not permit redistribution. On a public clone / cloud deployment the
application therefore starts in **empty/demo mode**: every page renders a clear empty state that
names the dataset it needs and where to obtain it, instead of raising an exception. Add the
datasets to `data/raw/` and rebuild the database from the Home page to enable full analytics.
The two required model artifacts (see §1) *are* shipped, so the severity and CCTV pages run
immediately.

---

## 5. Current Deployment Verification Gate Status
- **Local Working Tree**: Reviewed; changes committed after passing gates.
- **Git Remote**: Configured — `https://github.com/akashupadhayay106-au/RoadRisk-AI.git`.
- **GitHub Push**: Requires this machine to be authenticated with the owner's GitHub account.
- **Live Cloud URL**: Not attempted (pending an authenticated push + Streamlit Cloud deploy).
- **Public URL Policy**: Never invented or assumed.
