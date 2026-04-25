"""QuantPipe — unified app launcher.

Run with:  streamlit run app.py
"""

import contextlib
from datetime import datetime
from pathlib import Path

import streamlit as st

from config.settings import DATA_DIR, LOGS_DIR
from reports._theme import CSS, COLORS, badge

_LOGO = Path(__file__).parent / "assets" / "logo.png"

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuantPipe",
    page_icon=str(_LOGO) if _LOGO.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

NAV_CSS = f"""
<style>
/* ── Nav links ───────────────────────────────────────────────────────────── */
[data-testid="stSidebarNavLink"] {{
    border-radius: 6px;
    padding: 8px 12px;
    margin: 2px 0;
    transition: background 0.15s ease;
}}
[data-testid="stSidebarNavLink"]:hover {{
    background: {COLORS['card_bg']};
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: rgba(201,162,39,0.10);
    border-left: 3px solid {COLORS['gold']};
    padding-left: 9px;
}}
[data-testid="stSidebarNavSeparator"] p {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 700 !important;
    margin: 14px 0 3px !important;
    padding-left: 10px;
}}

/* ── Sidebar logo container ───────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stImage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 auto !important;
    line-height: 0;
    position: relative;
}}

/* Colour overlay: sidebar bg radiates inward from all edges,
   physically painting over the image corners with the sidebar colour */
[data-testid="stSidebar"] [data-testid="stImage"]::after {{
    content: '';
    position: absolute;
    inset: 0;
    pointer-events: none;
    border-radius: 0;
    background: radial-gradient(
        circle at 50% 50%,
        transparent           28%,
        {COLORS['surface']}1A 42%,
        {COLORS['surface']}55 54%,
        {COLORS['surface']}99 64%,
        {COLORS['surface']}CC 73%,
        {COLORS['surface']}EE 82%,
        {COLORS['surface']}   90%
    );
}}

/* ── Sidebar logo image — sharp + deep mask fade ─────────────────────────── */
[data-testid="stSidebar"] [data-testid="stImage"] img {{
    image-rendering: -webkit-optimize-contrast;
    image-rendering: high-quality;
    width: 100% !important;
    height: auto;
    display: block;
    margin: 0 auto;
    -webkit-mask-image: radial-gradient(
        circle at 50% 50%,
        black             24%,
        rgba(0,0,0,0.96)  33%,
        rgba(0,0,0,0.85)  42%,
        rgba(0,0,0,0.65)  51%,
        rgba(0,0,0,0.38)  60%,
        rgba(0,0,0,0.14)  70%,
        rgba(0,0,0,0.03)  80%,
        transparent       88%
    );
    mask-image: radial-gradient(
        circle at 50% 50%,
        black             24%,
        rgba(0,0,0,0.96)  33%,
        rgba(0,0,0,0.85)  42%,
        rgba(0,0,0,0.65)  51%,
        rgba(0,0,0,0.38)  60%,
        rgba(0,0,0,0.14)  70%,
        rgba(0,0,0,0.03)  80%,
        transparent       88%
    );
}}
</style>
"""

st.markdown(CSS + NAV_CSS, unsafe_allow_html=True)


# ── Lightweight status probe ───────────────────────────────────────────────────

def _probe() -> tuple[str, str, str]:
    tw  = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
    eq  = DATA_DIR / "bronze" / "equity" / "daily"
    log = LOGS_DIR / "pipeline.log"

    if not eq.exists() or not tw.exists():
        return "NO DATA", COLORS["negative"], "—"

    age_h = (datetime.now() - datetime.fromtimestamp(tw.stat().st_mtime)).total_seconds() / 3600

    last_run = "—"
    if log.exists():
        with contextlib.suppress(Exception):
            lines = log.read_text(errors="replace").splitlines()
            for line in reversed(lines):
                if "Pipeline complete" in line:
                    parts = line.split()
                    last_run = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else "unknown"
                    break

    if age_h > 48:
        return "STALE", COLORS["warning"], last_run
    return "LIVE", COLORS["positive"], last_run


status_label, status_color, last_run = _probe()

# ── Navigation ─────────────────────────────────────────────────────────────────

pg = st.navigation(
    {
        "Dashboards": [
            st.Page("reports/health_dashboard.py",
                    title="Pipeline Health",       icon="🔧"),
            st.Page("reports/performance_dashboard.py",
                    title="Performance",           icon="📈"),
            st.Page("reports/strategy_lab.py",
                    title="Strategy Lab",          icon="⚗️"),
            st.Page("reports/research_dashboard.py",
                    title="Research",              icon="🔬"),
        ],
        "Management": [
            st.Page("reports/portfolio_dashboard.py",
                    title="Portfolio",             icon="💼"),
        ],
        "Trading": [
            st.Page("reports/paper_trading_dashboard.py",
                    title="Paper / Live Trading",  icon="📄"),
        ],
        "Reference": [
            st.Page("reports/instructions.py",
                    title="Guide & Glossary",      icon="📖"),
        ],
    },
    position="sidebar",
)

# ── Sidebar status + footer (renders below nav links) ─────────────────────────

with st.sidebar:
    if _LOGO.exists():
        # 1/3 width, centred — columns give [1/3 gap | 1/3 image | 1/3 gap]
        _l, _c, _r = st.columns([1, 1, 1])
        with _c:
            st.image(str(_LOGO), use_container_width=True)

    st.markdown(f"""
<div style="border-top:1px solid {COLORS['border']};margin:8px 0 10px;"></div>
<div style="display:flex;align-items:center;gap:7px;padding:0 4px 4px;">
  <span style="width:7px;height:7px;border-radius:50%;
               background:{status_color};display:inline-block;flex-shrink:0;"></span>
  <span style="color:{status_color};font-size:0.72rem;font-weight:700;
               letter-spacing:0.06em;">{status_label}</span>
  <span style="color:{COLORS['text_muted']};font-size:0.67rem;margin-left:2px;">
    {last_run}
  </span>
</div>
<div style="padding:0 4px 8px;border-top:1px solid {COLORS['border']};margin-top:10px;">
  <div style="color:{COLORS['neutral']};font-size:0.66rem;font-weight:700;
              text-transform:uppercase;letter-spacing:0.09em;margin:10px 0 7px;">
    Quick Commands
  </div>
  <div style="background:{COLORS['card_bg']};border:1px solid {COLORS['border']};
              border-radius:7px;padding:9px 11px;line-height:2.1;">
    <div style="font-size:0.66rem;color:{COLORS['text_muted']};
                text-transform:uppercase;letter-spacing:0.07em;margin-bottom:1px;">
      Run pipeline
    </div>
    <code style="font-size:0.66rem;color:{COLORS['gold']};
                 font-family:monospace;">uv run python orchestration/run_pipeline.py</code>
    <div style="border-top:1px solid {COLORS['border_dim']};margin:6px 0;"></div>
    <div style="font-size:0.66rem;color:{COLORS['text_muted']};
                text-transform:uppercase;letter-spacing:0.07em;margin-bottom:1px;">
      Paper rebalance
    </div>
    <code style="font-size:0.66rem;color:{COLORS['gold']};
                 font-family:monospace;">uv run python orchestration/rebalance.py --broker ibkr</code>
  </div>
  <div style="color:{COLORS['text_muted']};font-size:0.62rem;margin-top:10px;
              text-align:center;letter-spacing:0.02em;">
    v0.1 · Paper trading only · Not investment advice
  </div>
</div>
""", unsafe_allow_html=True)

# ── Run selected page ──────────────────────────────────────────────────────────

pg.run()
