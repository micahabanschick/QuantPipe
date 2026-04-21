"""QuantPipe — unified app launcher.

Run with:  streamlit run app.py
"""

from datetime import datetime
from pathlib import Path

import streamlit as st

from config.settings import DATA_DIR, LOGS_DIR
from reports._theme import CSS, COLORS, badge

# ── Page config (called exactly once here; page files must NOT call it) ───────

st.set_page_config(
    page_title="QuantPipe",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation-link CSS overlay (on top of shared theme CSS) ──────────────────

NAV_CSS = f"""
<style>
/* Nav link items */
[data-testid="stSidebarNavLink"] {{
    border-radius: 6px;
    padding: 8px 12px;
    margin: 2px 0;
    transition: background 0.15s ease;
}}
[data-testid="stSidebarNavLink"]:hover {{
    background: {COLORS['card_bg']};
}}
/* Active page link */
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: rgba(0,212,170,0.10);
    border-left: 3px solid {COLORS['positive']};
    padding-left: 9px;
}}
/* Section label above nav group */
[data-testid="stSidebarNavSeparator"] p {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 700 !important;
    margin: 14px 0 3px !important;
    padding-left: 10px;
}}
</style>
"""

st.markdown(CSS + NAV_CSS, unsafe_allow_html=True)


# ── Lightweight status probe (file-stat only, no data load) ──────────────────

def _probe() -> tuple[str, str, str]:
    """Return (status_label, hex_color, last_run_str)."""
    tw  = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
    eq  = DATA_DIR / "bronze" / "equity" / "daily"
    log = LOGS_DIR / "pipeline.log"

    if not eq.exists() or not tw.exists():
        return "NO DATA", COLORS["negative"], "—"

    age_h = (datetime.now() - datetime.fromtimestamp(tw.stat().st_mtime)).total_seconds() / 3600

    last_run = "—"
    if log.exists():
        try:
            lines = log.read_text(errors="replace").splitlines()
            for line in reversed(lines):
                if "Pipeline complete" in line:
                    parts = line.split()
                    last_run = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else "unknown"
                    break
        except Exception:
            pass

    if age_h > 48:
        return "STALE", COLORS["warning"], last_run
    return "LIVE", COLORS["positive"], last_run


status_label, status_color, last_run = _probe()

# ── Sidebar — header (appears ABOVE nav links) ────────────────────────────────

with st.sidebar:
    st.markdown(f"""
<div style="padding:14px 4px 16px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <span style="font-size:1.7rem;line-height:1;">📊</span>
    <div>
      <div style="font-size:1.05rem;font-weight:800;color:{COLORS['text']};
                  letter-spacing:-0.03em;line-height:1.15;">QuantPipe</div>
      <div style="color:{COLORS['neutral']};font-size:0.68rem;
                  letter-spacing:0.02em;margin-top:1px;">
        Quantitative Finance Pipeline
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
    <span style="width:8px;height:8px;border-radius:50%;
                 background:{status_color};display:inline-block;
                 flex-shrink:0;"></span>
    <span style="color:{status_color};font-size:0.73rem;font-weight:700;
                 letter-spacing:0.06em;">{status_label}</span>
  </div>
  <div style="color:{COLORS['text_muted']};font-size:0.67rem;padding-left:15px;">
    Last pipeline: {last_run}
  </div>
</div>
<div style="border-top:1px solid {COLORS['border']};margin:0 0 4px;"></div>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────

pg = st.navigation(
    {
        "Dashboards": [
            st.Page("reports/health_dashboard.py",
                    title="Pipeline Health",   icon="🔧"),
            st.Page("reports/performance_dashboard.py",
                    title="Performance",       icon="📈"),
            st.Page("reports/strategy_lab.py",
                    title="Strategy Lab",      icon="⚗️"),
            st.Page("reports/research_dashboard.py",
                    title="Research",          icon="🔬"),
        ],
        "Management": [
            st.Page("reports/portfolio_dashboard.py",
                    title="Portfolio",         icon="💼"),
        ],
        "Reference": [
            st.Page("reports/instructions.py",
                    title="Guide & Glossary",  icon="📖"),
        ],
    },
    position="sidebar",
)

# ── Sidebar — footer (appears BELOW nav links) ────────────────────────────────

with st.sidebar:
    st.markdown(f"""
<div style="padding:14px 4px 8px;border-top:1px solid {COLORS['border']};margin-top:12px;">
  <div style="color:{COLORS['neutral']};font-size:0.66rem;font-weight:700;
              text-transform:uppercase;letter-spacing:0.09em;margin-bottom:7px;">
    Quick Commands
  </div>
  <div style="background:{COLORS['card_bg']};border:1px solid {COLORS['border']};
              border-radius:7px;padding:9px 11px;line-height:2.1;">
    <div style="font-size:0.66rem;color:{COLORS['text_muted']};
                text-transform:uppercase;letter-spacing:0.07em;margin-bottom:1px;">
      Launch app
    </div>
    <code style="font-size:0.68rem;color:{COLORS['blue']};
                 font-family:monospace;">streamlit run app.py</code>
    <div style="border-top:1px solid {COLORS['border_dim']};margin:6px 0;"></div>
    <div style="font-size:0.66rem;color:{COLORS['text_muted']};
                text-transform:uppercase;letter-spacing:0.07em;margin-bottom:1px;">
      Run pipeline
    </div>
    <code style="font-size:0.66rem;color:{COLORS['blue']};
                 font-family:monospace;">uv run python orchestration/run_pipeline.py</code>
    <div style="border-top:1px solid {COLORS['border_dim']};margin:6px 0;"></div>
    <div style="font-size:0.66rem;color:{COLORS['text_muted']};
                text-transform:uppercase;letter-spacing:0.07em;margin-bottom:1px;">
      Paper rebalance
    </div>
    <code style="font-size:0.66rem;color:{COLORS['blue']};
                 font-family:monospace;">uv run python orchestration/rebalance.py --broker paper</code>
  </div>
  <div style="color:{COLORS['text_muted']};font-size:0.62rem;margin-top:10px;
              text-align:center;letter-spacing:0.02em;">
    v0.1 · Paper trading only · Not investment advice
  </div>
</div>
""", unsafe_allow_html=True)

# ── Run selected page ─────────────────────────────────────────────────────────

pg.run()
