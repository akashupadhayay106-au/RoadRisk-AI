"""
RoadRisk AI — Premium Design System (theme)
============================================
Centralized CSS tokens and component styles for the RoadRisk AI interface.

Visual direction: deep midnight/navy surfaces, restrained cyan/blue accents,
amber/red reserved for meaningful risk/severity states, layered glass cards,
subtle ambient lighting, gentle 3D perspective on hover, and full
reduced-motion / keyboard-focus / responsive support.

The single exported constant ``PAGE_CSS`` is injected once by
``src.utils.inject_custom_css`` at the top of every page.
"""

# ---------------------------------------------------------------------------
# Base + ambient shell
# ---------------------------------------------------------------------------
_CSS_BASE = r"""
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --rdx-bg: #060B17;
  --rdx-surface: #0C1424;
  --rdx-surface-2: #111B31;
  --rdx-line: rgba(148, 184, 255, 0.10);
  --rdx-line-strong: rgba(148, 184, 255, 0.20);
  --rdx-cyan: #38BDF8;
  --rdx-cyan-bright: #22D3EE;
  --rdx-blue: #60A5FA;
  --rdx-indigo: #818CF8;
  --rdx-amber: #FBBF24;
  --rdx-red: #FB7185;
  --rdx-green: #34D399;
  --rdx-text: #E6EDF7;
  --rdx-muted: #8CA3C3;
  --rdx-faint: #5B6E8C;
  --rdx-radius: 16px;
}

html, body, [class*="css"], .stApp {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  color: var(--rdx-text);
}

.stApp {
  background:
    radial-gradient(1150px 640px at 12% -8%, rgba(56,189,248,0.10), transparent 60%),
    radial-gradient(900px 560px at 106% 4%, rgba(99,102,241,0.10), transparent 58%),
    radial-gradient(780px 500px at 50% 118%, rgba(34,211,238,0.07), transparent 60%),
    var(--rdx-bg);
}

/* Ambient pointer-following light layer (driven by DOM variables --px/--py,
   set by the tiny theme helper in inject_custom_css; static fallback below). */
.rdx-glow {
  position: fixed; inset: 0; z-index: 1; pointer-events: none;
  background: radial-gradient(560px circle at var(--px, 78%) var(--py, 16%),
    rgba(56,189,248,0.05), transparent 70%);
}

.block-container { padding-top: 1.2rem; padding-bottom: 4.5rem; max-width: 1220px; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #1E2A44; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #2A3A5E; }
::-webkit-scrollbar-track { background: transparent; }
"""

# ---------------------------------------------------------------------------
# Sidebar shell
# ---------------------------------------------------------------------------
_CSS_SIDEBAR = r"""
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A1120 0%, #0C1424 100%);
  border-right: 1px solid var(--rdx-line);
}
.sidebar-brand {
  display: flex; flex-direction: column; align-items: center; gap: 8px;
  padding: 30px 14px 10px; text-align: center;
}
.sidebar-logo {
  width: 52px; height: 52px; border-radius: 16px;
  display: grid; place-items: center; font-size: 28px;
  background: linear-gradient(145deg, rgba(56,189,248,0.20), rgba(99,102,241,0.10));
  border: 1px solid rgba(56,189,248,0.35);
  box-shadow: 0 10px 30px -8px rgba(56,189,248,0.45);
}
.sidebar-title {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 1.22rem; font-weight: 700; line-height: 1.2;
}
.sidebar-title .grad {
  background: linear-gradient(92deg, #7DD3FC, #818CF8, #22D3EE);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-subtitle {
  font-size: 0.68rem; color: var(--rdx-faint);
  letter-spacing: 0.16em; text-transform: uppercase;
}
.version-chip {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.66rem; font-weight: 700; color: var(--rdx-cyan);
  background: rgba(56,189,248,0.09);
  border: 1px solid rgba(56,189,248,0.25);
  padding: 3px 10px; border-radius: 999px; letter-spacing: 0.06em;
}
[data-testid="stSidebarNav"], [data-testid="stSidebarNav"] ul { padding-top: 0.25rem; }
[data-testid="stSidebarNav"] span { font-weight: 500; color: var(--rdx-muted); }
[data-testid="stSidebarNav"] a {
  border-radius: 10px; margin: 1px 0;
  transition: background 0.18s ease, color 0.18s ease;
}
[data-testid="stSidebarNav"] a:hover { background: rgba(56,189,248,0.08); color: var(--rdx-text); }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: linear-gradient(90deg, rgba(56,189,248,0.17), rgba(99,102,241,0.10));
  color: #7DD3FC;
}
"""

# ---------------------------------------------------------------------------
# Page header / hero
# ---------------------------------------------------------------------------
_CSS_HERO = r"""
.page-hero { margin: 0.1rem 0 1.1rem; }
.crumb {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  font-size: 0.72rem; color: var(--rdx-faint); letter-spacing: 0.09em;
  text-transform: uppercase; font-weight: 700; margin-bottom: 8px;
}
.crumb .cred { color: var(--rdx-cyan); }
.crumb .sep { opacity: 0.55; }
.page-hero h1 {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: clamp(1.7rem, 3.1vw, 2.5rem); font-weight: 800;
  letter-spacing: -0.02em; margin: 0 0 6px; line-height: 1.12;
}
.grad-cyan { background: linear-gradient(92deg, #7DD3FC 0%, #818CF8 55%, #22D3EE 110%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.grad-amber { background: linear-gradient(92deg, #FDE68A, #FB923C 90%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.grad-red { background: linear-gradient(92deg, #FDA4AF, #FB7185 70%, #F59E0B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.grad-teal { background: linear-gradient(92deg, #5EEAD4, #38BDF8 90%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.grad-indigo { background: linear-gradient(92deg, #C4B5FD, #818CF8 60%, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.grad-mono { background: linear-gradient(92deg, #F8FAFC, #A5B4CB); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.page-subtitle { color: var(--rdx-muted); font-size: 0.95rem; line-height: 1.6; max-width: 78ch; margin: 0; }
.hero-kicker {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.22);
  color: #7DD3FC; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 4px 12px; border-radius: 999px; margin-bottom: 10px;
}
.hero-kicker .pulse-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--rdx-cyan-bright);
  box-shadow: 0 0 0 0 rgba(34,211,238,0.6); animation: rdx-pulse 2.4s infinite;
}
@keyframes rdx-pulse {
  0% { box-shadow: 0 0 0 0 rgba(34,211,238,0.45); }
  70% { box-shadow: 0 0 0 8px rgba(34,211,238,0); }
  100% { box-shadow: 0 0 0 0 rgba(34,211,238,0); }
}
.status-strip {
  display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
  margin: 0.9rem 0 0.4rem;
}
.chip {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 0.32rem 0.8rem; border-radius: 999px; font-size: 0.74rem;
  font-weight: 600; border: 1px solid transparent; white-space: nowrap;
}
.chip .dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; background: currentColor; flex: none; }
.chip-ok   { background: rgba(52,211,153,0.12); color: var(--rdx-green); border-color: rgba(52,211,153,0.30); }
.chip-warn { background: rgba(251,191,36,0.11); color: var(--rdx-amber); border-color: rgba(251,191,36,0.28); }
.chip-err  { background: rgba(251,113,133,0.12); color: var(--rdx-red); border-color: rgba(251,113,133,0.30); }
.chip-info { background: rgba(96,165,250,0.11); color: var(--rdx-blue); border-color: rgba(96,165,250,0.28); }
"""

# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
_CSS_CARDS = r"""
.kpi-card {
  position: relative; overflow: hidden;
  background: linear-gradient(160deg, rgba(20,30,54,0.92), rgba(12,20,36,0.88));
  border: 1px solid var(--rdx-line);
  border-radius: var(--rdx-radius);
  padding: 1.35rem 1.2rem; text-align: center; height: 100%;
  box-shadow: 0 18px 40px -22px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.04);
  transition: transform 0.25s cubic-bezier(.2,.7,.3,1), box-shadow 0.25s ease, border-color 0.25s ease;
}
.kpi-card::before {
  content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none; opacity: 0;
  background: radial-gradient(420px circle at var(--px, 50%) var(--py, 0%), rgba(56,189,248,0.15), transparent 70%);
  transition: opacity 0.3s ease;
}
.kpi-card:hover {
  transform: translateY(-5px) perspective(900px) rotateX(2.2deg) rotateY(-1.4deg);
  border-color: rgba(56,189,248,0.35);
  box-shadow: 0 26px 54px -24px rgba(56,189,248,0.33), inset 0 1px 0 rgba(255,255,255,0.05);
}
.kpi-card:hover::before { opacity: 1; }
.kpi-icon { font-size: 1.55rem; line-height: 1; margin-bottom: 0.45rem; }
.kpi-value {
  font-family: 'Space Grotesk', sans-serif; font-size: 2.05rem; font-weight: 800;
  letter-spacing: -0.02em; margin: 0.25rem 0 0.18rem;
  background: linear-gradient(90deg, #F8FAFC, #C7D6F5);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.kpi-label {
  font-size: 0.7rem; color: var(--rdx-muted); text-transform: uppercase;
  letter-spacing: 0.1em; font-weight: 700;
}

.metric-card {
  position: relative; overflow: hidden;
  background: linear-gradient(160deg, rgba(20,30,54,0.9), rgba(12,20,36,0.86));
  border: 1px solid var(--rdx-line); border-left: 4px solid var(--rdx-blue);
  border-radius: 14px; padding: 0.95rem 1.05rem; height: 100%;
  box-shadow: 0 14px 34px -22px rgba(0,0,0,0.75);
  transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.metric-card:hover {
  transform: translateY(-3px) perspective(800px) rotateX(1.6deg);
  border-color: var(--rdx-line-strong);
  box-shadow: 0 20px 40px -22px rgba(56,189,248,0.28);
}
.metric-label { font-size: 0.67rem; color: var(--rdx-muted); text-transform: uppercase; letter-spacing: 0.09em; font-weight: 700; }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 800; color: var(--rdx-text); margin: 0.18rem 0; line-height: 1.15; }
.metric-sub { font-size: 0.74rem; color: var(--rdx-faint); }

.info-card {
  background: linear-gradient(160deg, rgba(20,30,54,0.8), rgba(12,20,36,0.75));
  border: 1px solid var(--rdx-line); border-left: 4px solid var(--rdx-blue);
  border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0;
  color: var(--rdx-text); line-height: 1.6;
}

.section-header {
  display: inline-block;
  font-size: 1.2rem; font-weight: 800; color: var(--rdx-text);
  margin: 1.6rem 0 0.9rem; padding-bottom: 0.45rem;
  border-bottom: 2px solid rgba(56,189,248,0.45);
}
"""

# ---------------------------------------------------------------------------
# States, badges, skeleton
# ---------------------------------------------------------------------------
_CSS_STATES = r"""
.status-badge {
  display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
  font-size: 0.72rem; font-weight: 700;
}
.status-ready  { background: rgba(52,211,153,0.13); color: var(--rdx-green); border: 1px solid rgba(52,211,153,0.3); }
.status-pending{ background: rgba(251,191,36,0.12); color: var(--rdx-amber); border: 1px solid rgba(251,191,36,0.3); }
.status-error { background: rgba(251,113,133,0.13); color: var(--rdx-red); border: 1px solid rgba(251,113,133,0.3); }

.state-box {
  display: flex; flex-direction: column; align-items: center; text-align: center;
  gap: 0.4rem; padding: 2.1rem 1.6rem; margin: 0.8rem 0;
  border-radius: 16px;
  background: linear-gradient(160deg, rgba(20,30,54,0.78), rgba(12,20,36,0.72));
  border: 1px dashed var(--rdx-line-strong);
}
.state-box.state-error { border-color: rgba(251,113,133,0.45); border-left: 4px solid var(--rdx-red); }
.state-box.state-empty { border-color: rgba(96,165,250,0.35); border-left: 4px solid var(--rdx-blue); }
.state-icon { font-size: 1.9rem; line-height: 1; }
.state-title { font-size: 1.02rem; font-weight: 700; color: var(--rdx-text); }
.state-msg { font-size: 0.87rem; color: var(--rdx-muted); max-width: 62ch; line-height: 1.5; }
.state-hint { font-size: 0.76rem; color: var(--rdx-faint); margin-top: 0.35rem; border-top: 1px solid var(--rdx-line); padding-top: 0.5rem; }

.svc-row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 0.35rem 0 0.75rem; }
.svc-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.3rem 0.75rem; border-radius: 999px; font-size: 0.73rem;
  font-weight: 600; border: 1px solid transparent; white-space: nowrap;
}
.svc-dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; background: currentColor; flex: none; }
.svc-ok  { background: rgba(52,211,153,0.12); color: var(--rdx-green); border-color: rgba(52,211,153,0.32); }
.svc-off { background: rgba(251,191,36,0.10); color: var(--rdx-amber); border-color: rgba(251,191,36,0.28); }
.svc-bad { background: rgba(251,113,133,0.12); color: var(--rdx-red); border-color: rgba(251,113,133,0.32); }

@keyframes rdx-shimmer { 0% { background-position: -420px 0; } 100% { background-position: 420px 0; } }
.rdx-skeleton {
  border-radius: 14px; height: 4.6rem; margin: 0.5rem 0;
  background: linear-gradient(90deg, rgba(20,30,54,0.6) 25%, rgba(56,189,248,0.12) 50%, rgba(20,30,54,0.6) 75%);
  background-size: 840px 100%;
  animation: rdx-shimmer 1.5s infinite linear;
  border: 1px solid var(--rdx-line);
}
"""

# ---------------------------------------------------------------------------
# Streamlit widgets
# ---------------------------------------------------------------------------
_CSS_WIDGETS = r"""
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; border: 1px solid var(--rdx-line-strong);
  background: linear-gradient(180deg, #16233D, #101A2E);
  color: var(--rdx-text); font-weight: 600;
  box-shadow: 0 10px 24px -14px rgba(0,0,0,0.8);
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease, color 0.18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px); border-color: rgba(56,189,248,0.5);
  background: linear-gradient(180deg, #1A2947, #131F36);
  color: #EAF4FF; box-shadow: 0 14px 30px -14px rgba(56,189,248,0.35);
}
[data-testid="stBaseButton-primary"] button, .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0EA5E9, #6366F1);
  border: 1px solid rgba(125,211,252,0.4); color: #F5FBFF;
}
[data-testid="stBaseButton-primary"] button:hover, .stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #38BDF8, #818CF8);
  border-color: rgba(125,211,252,0.7); color: #FFF;
}

[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--rdx-line); padding-bottom: 2px; }
[data-baseweb="tab"] {
  border-radius: 10px 10px 0 0; color: var(--rdx-muted); font-weight: 600;
  padding: 0.55rem 0.9rem; transition: color 0.18s ease, background 0.18s ease;
}
[data-baseweb="tab"]:hover { color: var(--rdx-text); background: rgba(56,189,248,0.06); }
[data-baseweb="tab"][aria-selected="true"] { color: #7DD3FC; background: rgba(56,189,248,0.10); box-shadow: inset 0 -2px 0 var(--rdx-cyan); }

[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  background-color: rgba(9,15,28,0.72) !important;
  border-color: var(--rdx-line-strong) !important;
  border-radius: 12px !important;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within, [data-baseweb="textarea"]:focus-within {
  border-color: var(--rdx-cyan) !important;
  box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}
input, textarea { color: var(--rdx-text) !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--rdx-faint) !important; }

[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {
  background: #0D1730 !important; border: 1px solid var(--rdx-line-strong) !important; border-radius: 12px !important;
}
li[role="option"] { color: var(--rdx-muted) !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] { background: rgba(56,189,248,0.12) !important; color: var(--rdx-text) !important; }

[data-testid="stDateInput"] input, [data-testid="stNumberInput"] input, [data-testid="stSlider"] input, [data-testid="stSidebar"] input { color: var(--rdx-text) !important; }

[data-testid="stDataFrame"] {
  border: 1px solid var(--rdx-line); border-radius: 14px; overflow: hidden;
  box-shadow: 0 18px 40px -28px rgba(0,0,0,0.9);
}
[data-testid="stExpander"] {
  border: 1px solid var(--rdx-line); border-radius: 14px;
  background: rgba(13,21,38,0.5); overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600; color: var(--rdx-text); }

[data-testid="stMetric"] {
  background: linear-gradient(160deg, rgba(20,30,54,0.85), rgba(12,20,36,0.8));
  border: 1px solid var(--rdx-line); border-radius: 14px; padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--rdx-muted); }
[data-testid="stMetricValue"] { color: var(--rdx-text); }
[data-testid="stMetricDelta"] { color: var(--rdx-cyan); }

[data-testid="stChatMessage"] {
  background: rgba(13,21,38,0.55); border: 1px solid var(--rdx-line);
  border-radius: 16px; padding: 0.7rem 0.9rem;
}
[data-testid="stChatInput"] [data-baseweb="input"] { border-radius: 999px !important; }

[data-testid="stFileUploader"] section {
  background: rgba(13,21,38,0.5); border: 1px dashed rgba(56,189,248,0.35); border-radius: 14px;
}
[data-testid="stCode"] { border-radius: 12px; border: 1px solid var(--rdx-line); }
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--rdx-line); }
[data-testid="stCaptionContainer"] { color: var(--rdx-faint); }
div[data-testid="stToolbar"] { opacity: 0.35; }
"""

# ---------------------------------------------------------------------------
# Accessibility + responsive
# ---------------------------------------------------------------------------
_CSS_ACCESS = r"""
a:focus-visible, button:focus-visible, [role="button"]:focus-visible,
input:focus-visible, textarea:focus-visible, summary:focus-visible {
  outline: 2px solid var(--rdx-cyan); outline-offset: 2px; border-radius: 8px;
}

@media (max-width: 768px) {
  .kpi-card { padding: 1rem 0.8rem; }
  .kpi-value { font-size: 1.55rem; }
  .kpi-label { font-size: 0.64rem; }
  .metric-card { padding: 0.78rem 0.85rem; }
  .metric-value { font-size: 1.25rem; }
  .page-hero h1 { font-size: 1.5rem; }
  .section-header { font-size: 1.05rem; }
  .state-box { padding: 1.3rem 1rem; }
  .info-card { padding: 0.85rem 0.95rem; font-size: 0.9rem; }
  .block-container { padding-left: 1rem; padding-right: 1rem; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .kpi-card:hover, .metric-card:hover { transform: none !important; }
  .hero-kicker .pulse-dot { animation: none !important; }
}
"""

PAGE_CSS = "".join([
    _CSS_BASE,
    _CSS_SIDEBAR,
    _CSS_HERO,
    _CSS_CARDS,
    _CSS_STATES,
    _CSS_WIDGETS,
    _CSS_ACCESS,
])


