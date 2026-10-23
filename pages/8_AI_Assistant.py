"""
🤖 AI Assistant — Gemini-Powered Road Safety Q&A
==================================================
Module I: Chat interface for road safety questions.
Will be fully implemented in Phase 9.
"""

import streamlit as st
from src.config import PAGE_CONFIG, get_secret
from src.utils import inject_custom_css, setup_sidebar, section_header, info_card

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

st.markdown("""
<h1 style="
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #14B8A6, #4DA8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🤖 AI Road Safety Assistant</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Ask questions about road safety, accident data, and model results.
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Check API Key ────────────────────────────────────────────────────────────
gemini_key = get_secret("GEMINI_API_KEY")

if not gemini_key or gemini_key == "your-gemini-api-key-here":
    st.warning(
        "⚠️ **Gemini API Key not configured.**\n\n"
        "To use the AI Assistant:\n"
        "1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)\n"
        "2. Add it to `.streamlit/secrets.toml`:\n"
        "   ```\n"
        '   GEMINI_API_KEY = "your-actual-key"\n'
        "   ```\n"
        "3. Restart the app."
    )
else:
    st.success("✅ Gemini API key detected.")

info_card(
    "🚧 <strong>Coming in Phase 9</strong><br>"
    "This module will include:<br>"
    "• Chat interface powered by Google Gemini<br>"
    "• Grounded answers using dashboard data and SQL queries<br>"
    "• Road safety Q&A with source citations<br>"
    "• Model performance and prediction explanations<br>"
    "• Conversation history management<br>"
    "• RAG integration for project documentation"
)

section_header("Sample Questions (Preview)", "💬")
st.markdown("""
*Once implemented, you'll be able to ask:*
- "What are the top 5 states with the most accidents?"
- "How does weather affect accident severity?"
- "Explain the model's prediction for this input"
- "What does the hotspot analysis show for urban areas?"
- "What are the main causes of road accidents in India?"
""")
