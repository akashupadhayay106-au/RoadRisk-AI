"""
🤖 AI Assistant — Gemini-Powered & Offline-Grounded Road Safety Q&A
====================================================================
Module I: Dual-engine conversational road safety assistant. Features Google
Gemini integration with graceful, deterministic offline RAG fallback.
Strictly blocks prompt injection, forbids arbitrary code/SQL execution,
cites project documents, and honestly communicates uncertainty.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import streamlit as st
import pandas as pd

from src.config import PAGE_CONFIG, get_secret
from src.utils import inject_custom_css, setup_sidebar, section_header, info_card, status_badge
from src.ai_assistant import ask_roadrisk_assistant

# ── Streamlit Page Setup ───────────────────────────────────────────────────────
st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.2rem;">
    <h1 style="
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(135deg, #14B8A6 0%, #4DA8FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    ">🤖 Grounded Road Safety & Analytics Assistant</h1>
    <p style="color: #8899A6; font-size: 0.95rem; line-height: 1.5;">
        Ask questions about Indian accident statistics, ML severity classification,
        CCTV crash detection, weather risk factors, and project documentation.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Check Provider & Configuration Status ────────────────────────────────────
gemini_key = get_secret("GEMINI_API_KEY")
has_gemini = bool(gemini_key and gemini_key != "your-gemini-api-key-here" and len(gemini_key) > 10)

col_s1, col_s2 = st.columns([3, 1])
with col_s1:
    if has_gemini:
        st.markdown("""
        <div style="background: rgba(0, 200, 83, 0.1); border: 1px solid rgba(0, 200, 83, 0.3); border-left: 4px solid #00C853; border-radius: 6px; padding: 0.6rem 0.9rem; margin-bottom: 1rem;">
            <span style="color: #00C853; font-weight: 700; font-size: 0.85rem;">🟢 Active Engine: Google Gemini 1.5 Flash</span>
            <span style="color: #8899A6; font-size: 0.8rem; margin-left: 8px;">(Live LLM with Grounded Project Context)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(77, 168, 255, 0.1); border: 1px solid rgba(77, 168, 255, 0.3); border-left: 4px solid #4DA8FF; border-radius: 6px; padding: 0.6rem 0.9rem; margin-bottom: 1rem;">
            <span style="color: #4DA8FF; font-weight: 700; font-size: 0.85rem;">🔵 Active Engine: Grounded Offline Knowledge Engine</span>
            <span style="color: #8899A6; font-size: 0.8rem; margin-left: 8px;">(No API key required • 100% deterministic & private)</span>
        </div>
        """, unsafe_allow_html=True)

with col_s2:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Initialize Session State Messages ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am the **RoadRisk AI Safety Assistant**.\n\n"
                "I can help you explore:\n"
                "- 🇮🇳 **Indian Accident Analytics**: State rankings, fatality rates, and time-of-day risks.\n"
                "- 🤖 **Severity ML Model**: Feature importance and how target leakage was prevented.\n"
                "- 📹 **CCTV Vision Model**: MobileNetV2 architecture, latency, and the video sequence leakage audit.\n"
                "- 🌦️ **Atmospheric Risk**: How rainfall and fog impact road accident severity.\n"
                "- 🗺️ **Geographic Boundaries**: Why UrbanGB (UK) data is strictly isolated from Indian data.\n\n"
                "What would you like to know?"
            ),
            "mode": "system",
            "sources": [],
        }
    ]

# ── Suggested Questions Carousel ──────────────────────────────────────────────
st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #8899A6; margin-bottom: 0.4rem;'>💡 QUICK INQUIRIES:</div>", unsafe_allow_html=True)
q_cols = st.columns(4)

suggested_queries = [
    "What are the top 5 states by accidents?",
    "How does rain affect accident severity?",
    "Explain CCTV model & leakage audit",
    "Is UrbanGB data part of Indian roads?",
]

selected_suggestion = None
for i, prompt_text in enumerate(suggested_queries):
    with q_cols[i]:
        if st.button(prompt_text, key=f"sugg_{i}", use_container_width=True):
            selected_suggestion = prompt_text

# ── Render Conversation History ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Verified Document Sources Cited", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

# ── Handle Input ──────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a road safety or project analytics question...")
active_prompt = selected_suggestion if selected_suggestion else user_input

if active_prompt:
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing project knowledge base and verified databases..."):
            result = ask_roadrisk_assistant(active_prompt)
            resp_content = result["response"]
            resp_mode = result["mode"]
            resp_sources = result["sources"]

        st.markdown(resp_content)

        if resp_sources:
            with st.expander("📚 Verified Document Sources Cited", expanded=False):
                for s in resp_sources:
                    st.markdown(f"- `{s}`")

        # Mode Badge Indicator
        if resp_mode == "gemini":
            st.caption("⚡ Powered by Google Gemini 1.5 Flash (Grounded with Local Context)")
        elif resp_mode == "offline":
            st.caption("🔒 Powered by Grounded Offline Knowledge & SQLite Parameterized Queries")
        elif resp_mode == "safety_block":
            st.caption("🛡️ Security Filter Enforced")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": resp_content,
        "mode": resp_mode,
        "sources": resp_sources,
    })

# ── Governance & Safety Notice ────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size: 0.78rem; color: #8899A6; line-height: 1.4;">
    <strong>Safety & Governance Architecture:</strong>
    All user queries are screened through anti-injection guardrails. The assistant operates on a read-only database
    connection with parameterized SQL queries and does not execute arbitrary code or shell commands.
    In the absence of an API key, the system seamlessly uses local deterministic BM25 documentation retrieval.
</div>
""", unsafe_allow_html=True)
