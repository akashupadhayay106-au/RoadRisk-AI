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
from src.ai_assistant import ask_roadrisk_assistant, get_provider_status

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
# provider_statuses() exposes ONLY configured/available booleans — never keys.
provider_rows = get_provider_status()
live_row = next((p for p in provider_rows if p["kind"] == "llm" and p["configured"]), None)

_PILL_PALETTE = {
    "ok": ("rgba(0,200,83,0.15)", "rgba(0,200,83,0.45)", "#00C853"),
    "idle": ("rgba(77,168,255,0.12)", "rgba(77,168,255,0.4)", "#4DA8FF"),
    "off": ("rgba(136,153,166,0.12)", "rgba(136,153,166,0.35)", "#8899A6"),
}
_PILL_DOT = {"ok": "🟢", "idle": "🔵", "off": "⚪"}


def _pill(label: str, state: str) -> str:
    """Render a small status chip. Shows only configuration state, never keys."""
    bg, border, fg = _PILL_PALETTE[state]
    return (
        f"<span style=\"display:inline-block;padding:2px 10px;border-radius:999px;"
        f"font-size:0.72rem;font-weight:600;margin:0 6px 6px 0;"
        f"border:1px solid {border};background:{bg};color:{fg};\">"
        f"{_PILL_DOT[state]} {label}</span>"
    )


def _pill_state(row: dict) -> str:
    if row["name"] == "offline":
        return "idle"
    if not row["configured"]:
        return "off"
    if row["available"] is False:
        return "off"
    return "ok"


col_s1, col_s2 = st.columns([3, 1])
with col_s1:
    if live_row:
        st.markdown(
            f"""
            <div style="background: rgba(0, 200, 83, 0.1); border: 1px solid rgba(0, 200, 83, 0.3); border-left: 4px solid #00C853; border-radius: 6px; padding: 0.6rem 0.9rem; margin-bottom: 1rem;">
                <span style="color: #00C853; font-weight: 700; font-size: 0.85rem;">🟢 Active Engine: {live_row['label']} ({live_row['model']})</span>
                <span style="color: #8899A6; font-size: 0.8rem; margin-left: 8px;">(Live LLM grounded with project context)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(77, 168, 255, 0.1); border: 1px solid rgba(77, 168, 255, 0.3); border-left: 4px solid #4DA8FF; border-radius: 6px; padding: 0.6rem 0.9rem; margin-bottom: 1rem;">
                <span style="color: #4DA8FF; font-weight: 700; font-size: 0.85rem;">🔵 Active Engine: Grounded Offline Knowledge Engine</span>
                <span style="color: #8899A6; font-size: 0.8rem; margin-left: 8px;">(No API key required • 100% deterministic & private)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    pills = "".join(
        _pill(
            f"{row['label']} — {'configured' if row['configured'] else 'not configured'}",
            _pill_state(row),
        )
        for row in provider_rows
    )
    st.markdown(
        "<div style='margin-bottom:0.8rem;'>"
        "<span style='font-size:0.72rem;color:#8899A6;font-weight:600;'>PROVIDER STATUS (keys are never displayed):</span><br>"
        f"{pills}</div>",
        unsafe_allow_html=True,
    )

with col_s2:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("🔌 Test Providers", use_container_width=True):
        from src.providers import provider_statuses as _probe_statuses
        with st.spinner("Probing configured providers..."):
            st.session_state.provider_report = _probe_statuses(probe=True)

if st.session_state.get("provider_report"):
    with st.expander("🔍 Provider health probe (non-sensitive)", expanded=False):
        for row in st.session_state["provider_report"]:
            st.markdown(f"- **{row['label']}** — {row['detail']}")

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
        _MODE_LABELS = {
            "gemini": "⚡ Powered by Google Gemini (grounded with local context)",
            "groq": "⚡ Powered by Groq Cloud (grounded with local context)",
            "huggingface": "⚡ Powered by Hugging Face Inference (grounded with local context)",
            "offline": "🔒 Powered by Grounded Offline Knowledge & SQLite Parameterized Queries",
            "safety_block": "🛡️ Security Filter Enforced",
        }
        st.caption(_MODE_LABELS.get(resp_mode, f"⚡ Powered by {resp_mode}"))

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
