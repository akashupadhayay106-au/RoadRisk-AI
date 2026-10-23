# RoadRisk AI — Deployment Checklist & Hosting Guide

## 1. Pre-Deployment Secret & Hygiene Audit

Before pushing to any public git forge (GitHub, GitLab), verify that no credentials or private data are committed:

- [x] **Virtual Environment Ignored**: `.venv/` is excluded by `.gitignore`.
- [x] **Streamlit Secrets Ignored**: `.streamlit/secrets.toml` is excluded by `.gitignore`.
- [x] **Local Databases Ignored**: `*.db`, `*.sqlite` are excluded by `.gitignore`.
- [x] **Large CCTV Datasets Ignored**: `data/train/`, `data/val/`, `data/test/` are excluded by `.gitignore`.
- [x] **Model Weights Ignored**: `models/*.pth`, `models/*.joblib` are excluded by `.gitignore`.
- [x] **No Hardcoded Tokens**: Scanned all `.py` files for API key literals (zero findings).
- [x] **Example Template Provided**: `.streamlit/secrets.toml.example` contains placeholders only (`your-gemini-api-key-here`).

---

## 2. GitHub Repository Initialization & Push

Because this machine is currently not authenticated with a personal GitHub account or SSH key, follow these exact terminal steps:

### Step A: Create a New GitHub Repository
1. Log in to [GitHub](https://github.com/) in your browser.
2. Click **New Repository**.
3. Name it: `RoadRisk-AI`.
4. Choose **Public** (or Private).
5. Do **NOT** check "Initialize with README" (we already have a complete, production-grade repository).
6. Click **Create repository**.

### Step B: Connect Local Git & Push
In your PowerShell terminal in the project directory (`c:\Users\Dell\OneDrive\Desktop\MACHINE_LEARNING_PROJECTS\RoadRisk AI`), run:

```powershell
# 1. Add your newly created remote URL (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/RoadRisk-AI.git

# 2. Rename branch to main if needed
git branch -M main

# 3. Push all commits to GitHub
git push -u origin main
```

---

## 3. Streamlit Community Cloud Deployment

Streamlit Community Cloud provides 100% free hosting for portfolio data apps with zero AWS/cloud costs.

### Step-by-Step Instructions:
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Fill in the repository configuration:
   - **Repository**: `YOUR_USERNAME/RoadRisk-AI`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `roadrisk-ai.streamlit.app` (or customizable subdomain).
4. Click **Advanced settings...**:
   - In **Python version**, select **3.12** (matches local development).
   - In **Secrets**, paste your optional configuration:
     ```toml
     GEMINI_API_KEY = "your-actual-gemini-api-key"
     ```
   *(Note: The app will run smoothly in grounded offline mode even without a key!)*
5. Click **Deploy!**
6. Streamlit Cloud will run `pip install -r requirements.txt` and launch the interactive dashboard within 2–3 minutes.

---

## 4. Current Deployment Verification Gate Status
- **Local Working Tree**: Cleanly staged and committed.
- **Git Remote**: Not configured (requires user's GitHub username).
- **GitHub Push**: Blocked pending user account authentication.
- **Live Cloud URL**: Not attempted (pending GitHub remote creation).
- **Public URL Policy**: Never invented or assumed.
