"""
RoadRisk AI — Shared UI Utilities & Components
================================================
Presentation helpers used by the entry point and every page: the premium
design-system CSS injection, KPI/metric cards, headers, state blocks,
status indicators, sidebar chrome, and lightweight data-loading helpers.

The heavy CSS lives in ``src.theme`` so pages never duplicate style logic.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import APP_VERSION, PROJECT_ROOT
from src.theme import PAGE_CSS

# A local developer checkout has a ``.git`` directory; a public deployment
# (Streamlit Community Cloud) ships tracked files only, so this is a reliable,
# deterministic way to label the running environment without any secrets.
_IS_LOCAL_CHECKOUT = (PROJECT_ROOT / ".git").exists()


def _env_label() -> str:
    """Return a short deployment-context label ('LOCAL' or 'PUBLIC')."""
    return "LOCAL" if _IS_LOCAL_CHECKOUT else "PUBLIC"


# ── CSS / Ambient Layer Injection ────────────────────────────────────────────

def _inject_pointer_spotlight() -> None:
    """Optional pointer-following light enhancer (best-effort, never fatal).

    Rendered inside a sandboxed same-origin iframe; when the browser permits
    parent access it sets ``--px/--py`` CSS variables on the app root so the
    ``.rdx-glow`` layer and KPI cards follow the cursor. When blocked, the
    static radial fallback in the stylesheet keeps the design intact.
    """
    try:
        st.components.v1.html(
            "<script>try{var d=window.parent.document;"
            "function u(e){d.documentElement.style.setProperty('--px',e.clientX+'px');"
            "d.documentElement.style.setProperty('--py',e.clientY+'px')}"
            "d.addEventListener('mousemove',u,{passive:true})}catch(e){}</script>",
            height=1,
        )
    except Exception:  # pragma: no cover - cosmetic enhancement only
        pass


def inject_custom_css() -> None:
    """Inject the premium dark-theme design system once per script run."""
    st.markdown(f"<style>{PAGE_CSS}</style>", unsafe_allow_html=True)
    st.markdown('<div class="rdx-glow" aria-hidden="true"></div>', unsafe_allow_html=True)
    _inject_pointer_spotlight()


# ── Page Header / Breadcrumb ─────────────────────────────────────────────────

def page_header(
    title: str,
    section: str = "",
    subtitle: str = "",
    icon: str = "🚦",
    accent: str = "cyan",
) -> None:
    """Render the premium page hero: breadcrumb + gradient title + subtitle.

    ``accent`` selects one of the ``grad-*`` classes (cyan, teal, amber, red,
    indigo, mono) so each module keeps its own identity within one system.
    """
    crumb_html = ""
    if section:
        crumb_html = (
            '<div class="crumb">'
            '<span class="cred">RoadRisk AI</span>'
            '<span class="sep">/</span>'
            f"<span>{section}</span>"
            "</div>"
        )
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="page-hero">{crumb_html}'
        f'<h1><span class="grad-{accent}">{icon} {title}</span></h1>'
        f"{subtitle_html}</div>",
        unsafe_allow_html=True,
    )


def status_chip(label: str, state: str = "ok") -> str:
    """Return HTML for a compact status chip. state: ok | warn | err | info."""
    return f'<span class="chip chip-{state}"><span class="dot"></span>{label}</span>'


def render_status_strip(chips: list[str]) -> None:
    """Render a row of status chips (typically deployment / data-source state)."""
    st.markdown(
        f'<div class="status-strip">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def hero_kicker(label: str, pulsing: bool = True) -> str:
    """Return HTML for the animated kicker pill above a page header."""
    dot = '<span class="pulse-dot"></span>' if pulsing else ""
    return f'<div class="hero-kicker">{dot}{label}</div>'


def skeleton(count: int = 1) -> None:
    """Render shimmer placeholders (used before heavy content resolves)."""
    st.markdown('<div class="rdx-skeleton"></div>' * count, unsafe_allow_html=True)


# ── KPI Card Widget ──────────────────────────────────────────────────────────

def kpi_card(icon: str, label: str, value: str | int | float, help_text: str | None = None) -> str:
    """Return HTML for a single KPI card. `help_text` renders as a tooltip."""
    if isinstance(value, bool):
        formatted = str(value)
    elif isinstance(value, int):
        formatted = f"{value:,.0f}"
    elif isinstance(value, float):
        formatted = f"{value:,.2f}" if abs(value) < 10 else f"{value:,.1f}"
    else:
        formatted = str(value)
    tooltip = f' title="{help_text}"' if help_text else ""
    return f"""
    <div class="kpi-card"{tooltip}>
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{formatted}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def render_kpi_row(kpis: list[tuple[str, str, str | int | float]]) -> None:
    """
    Render a row of KPI cards.
    kpis: list of (icon, label, value) tuples.
    """
    cols = st.columns(len(kpis))
    for col, (icon, label, value) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(icon, label, value), unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str | int | float,
    sublabel: str = "",
    border_color: str = "#60A5FA",
) -> None:
    """
    Render a compact metric card with a coloured left accent border.

    Used by the weather, CCTV and explainability pages for their KPI ribbons.
    ``border_color`` is applied inline so callers can convey semantic meaning
    (e.g. red for an elevated adverse-weather severe rate).
    """
    sub_html = f'<div class="metric-sub">{sublabel}</div>' if sublabel else ""
    st.markdown(
        f'<div class="metric-card" style="border-left-color: {border_color};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f"{sub_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def export_dataframe_ui(
    df: pd.DataFrame,
    filename_stem: str = "roadrisk_export",
    label: str = "⬇️ Export table as CSV",
) -> None:
    """
    Render a CSV download button for a DataFrame.
    """
    if df is None or df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{filename_stem}.csv",
        mime="text/csv",
        use_container_width=False,
    )


# ── Empty / Error / Loading States ───────────────────────────────────────────

def empty_state(title: str, message: str = "", hint: str = "", icon: str = "🔍") -> None:
    """Render a consistent empty-state block (e.g. no rows match the filters)."""
    hint_html = f'<div class="state-hint">{hint}</div>' if hint else ""
    msg_html = f'<div class="state-msg">{message}</div>' if message else ""
    st.markdown(
        f'<div class="state-box state-empty">'
        f'<div class="state-icon">{icon}</div>'
        f'<div class="state-title">{title}</div>'
        f"{msg_html}{hint_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def error_state(title: str, message: str = "", hint: str = "", icon: str = "⚠️") -> None:
    """Render a consistent error-state block for recoverable failures."""
    hint_html = f'<div class="state-hint">{hint}</div>' if hint else ""
    msg_html = f'<div class="state-msg">{message}</div>' if message else ""
    st.markdown(
        f'<div class="state-box state-error">'
        f'<div class="state-icon">{icon}</div>'
        f'<div class="state-title">{title}</div>'
        f"{msg_html}{hint_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Service / API Status Indicator ───────────────────────────────────────────

def service_status_row(services: list[dict]) -> None:
    """
    Render a row of secret-free service status pills.

    Each entry must be a dict with:
      - label (str): display name, e.g. "Weather (Open-Meteo)"
      - configured (bool): whether any credentials/endpoint are set up
      - available (bool | None): None = not probed, True/False = probe result
      - detail (str, optional): short non-sensitive explanation, shown as tooltip

    Never accepts or renders credential values.
    """
    pills = []
    for svc in services:
        configured = bool(svc.get("configured"))
        available = svc.get("available")
        if not configured:
            state_class, state_word = "svc-off", "not configured"
        elif available is True:
            state_class, state_word = "svc-ok", "available"
        elif available is False:
            state_class, state_word = "svc-bad", "unavailable"
        else:
            state_class, state_word = "svc-off", "not tested"

        detail = svc.get("detail") or ""
        tooltip = f' title="{detail}"' if detail else ""
        label = svc.get("label", "Service")
        pills.append(
            f'<span class="svc-pill {state_class}"{tooltip}>'
            f'<span class="svc-dot"></span>{label}: {state_word}</span>'
        )

    st.markdown(f'<div class="svc-row">{"".join(pills)}</div>', unsafe_allow_html=True)


# ── Section Header / Info Box / Status Badge ─────────────────────────────────

def section_header(title: str, icon: str = "") -> None:
    """Render a styled section header."""
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def info_card(text: str) -> None:
    """Render a styled information card."""
    st.markdown(
        f'<div class="info-card">{text}</div>',
        unsafe_allow_html=True,
    )


def status_badge(label: str, status: str = "ready") -> str:
    """Return HTML for a status badge. status: 'ready' | 'pending' | 'error'."""
    return f'<span class="status-badge status-{status}">{label}</span>'


# ── Sidebar Setup ────────────────────────────────────────────────────────────

def setup_sidebar() -> None:
    """Render the premium branded sidebar header (logo, title, version, env)."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-logo">🚦</div>
                <div class="sidebar-title"><span class="grad">RoadRisk AI</span></div>
                <div class="sidebar-subtitle">Road Safety Intelligence</div>
                <span class="version-chip">v{APP_VERSION} · {_env_label()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()


# ── Data Loading Helper ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_csv_cached(path: str | Path, **kwargs) -> pd.DataFrame | None:
    """Load a CSV with Streamlit caching. Returns None if the file is missing."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, **kwargs)
    except Exception as e:
        st.error(f"Failed to load {p.name}: {e}")
        return None


# ── Formatting Helpers ───────────────────────────────────────────────────────

def format_number(n: int | float) -> str:
    """Human-readable number formatting (e.g., 1,234,567)."""
    if pd.isna(n):
        return "N/A"
    return f"{n:,.0f}"


def format_percentage(n: float, decimals: int = 1) -> str:
    """Format a fraction (0–1) as a percentage string."""
    if pd.isna(n):
        return "N/A"
    return f"{n * 100:.{decimals}f}%"

