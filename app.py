"""QuantPipe — unified app launcher.

Run with:  streamlit run app.py
"""

import base64
import contextlib
import io
from datetime import datetime
from pathlib import Path

import streamlit as st

from config.settings import DATA_DIR, LOGS_DIR
from reports._theme import CSS, COLORS, badge

_LOGO       = Path(__file__).parent / "assets" / "logo.png"
_LOGO_FULL  = Path(__file__).parent / "assets" / "logo_full.png"
_LOGO_WORDS = Path(__file__).parent / "assets" / "logo_words.png"
_FAVICON    = Path(__file__).parent / "assets" / "favicon.png"

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuantPipe",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

NAV_CSS = f"""
<style>
/* ── Nav links ───────────────────────────────────────────────────────────── */
/* ── Nav links — target both the link and its inner text span ───────────── */
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavLink"] span,
[data-testid="stSidebarNavLink"] p {{
    border-radius: 7px;
    padding: 8px 12px;
    margin: 2px 4px;
    transition: background 0.15s ease, color 0.15s ease;
    color: {COLORS['neutral']} !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    text-decoration: none !important;
}}
[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebarNavLink"]:hover span,
[data-testid="stSidebarNavLink"]:hover p {{
    background: rgba(201,162,39,0.07) !important;
    color: {COLORS['text']} !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: rgba(201,162,39,0.13) !important;
    border-left: 3px solid {COLORS['gold']} !important;
    padding-left: 9px !important;
    border-radius: 0 7px 7px 0 !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] span,
[data-testid="stSidebarNavLink"][aria-current="page"] p {{
    color: {COLORS['gold']} !important;
    font-weight: 700 !important;
}}
[data-testid="stSidebarNavSeparator"] p {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.61rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.11em !important;
    font-weight: 700 !important;
    margin: 16px 0 4px !important;
    padding-left: 12px;
}}

/* ── Sidebar logo images (transparent PNGs) ──────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stImage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 auto !important;
    line-height: 0;
}}
[data-testid="stSidebar"] [data-testid="stImage"] img {{
    image-rendering: -webkit-optimize-contrast;
    image-rendering: high-quality;
    width: 100% !important;
    height: auto;
    display: block;
    margin: 0 auto;
    filter: drop-shadow(0 0 10px rgba(201,162,39,0.30))
            drop-shadow(0 0 22px rgba(201,162,39,0.12));
}}
</style>
"""

st.markdown(CSS + NAV_CSS, unsafe_allow_html=True)


# ── Lightweight status probe ───────────────────────────────────────────────────

def _et_now():
    """Return current datetime in US/Eastern."""
    from zoneinfo import ZoneInfo
    from datetime import timezone
    return datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))


def _next_run_et() -> str:
    """Return the next Mon–Fri 21:30 UTC pipeline run, displayed in Eastern Time."""
    from zoneinfo import ZoneInfo
    from datetime import timezone, timedelta
    et = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    # Pipeline fires at 21:30 UTC each weekday
    candidate = now_utc.replace(hour=21, minute=30, second=0, microsecond=0)
    for offset in range(8):
        t = candidate + timedelta(days=offset)
        if t > now_utc and t.weekday() < 5:
            t_et = t.astimezone(et)
            days = (t.date() - now_utc.date()).days
            tz_abbr = t_et.strftime("%Z")
            if days == 0:
                return f"today {t_et.strftime('%H:%M')} {tz_abbr}"
            if days == 1:
                return f"tomorrow {t_et.strftime('%H:%M')} {tz_abbr}"
            return f"{t_et.strftime('%a %H:%M')} {tz_abbr}"
    return "—"


def _probe() -> tuple[str, str, str, str, int]:
    """Return (status_label, color, last_run, next_run, n_positions)."""
    from zoneinfo import ZoneInfo
    from datetime import timezone
    et = ZoneInfo("America/New_York")

    tw  = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
    eq  = DATA_DIR / "bronze" / "equity" / "daily"

    next_run = _next_run_et()

    if not eq.exists() or not tw.exists():
        return "NO DATA", COLORS["negative"], "—", next_run, 0

    age_h = (datetime.now() - datetime.fromtimestamp(tw.stat().st_mtime)).total_seconds() / 3600

    # Prefer the heartbeat file — it's always written by the pipeline regardless
    # of where systemd redirects stdout, and is more reliable than log parsing.
    last_run = "—"
    heartbeat = PROJECT_ROOT / ".pipeline_heartbeat.json"
    with contextlib.suppress(Exception):
        import json
        hb = json.loads(heartbeat.read_text(encoding="utf-8"))
        ts = hb.get("ts_utc", "")
        if ts:
            dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            dt_et  = dt_utc.astimezone(et)
            tz_abbr = dt_et.strftime("%Z")
            last_run = dt_et.strftime(f"%Y-%m-%d %H:%M {tz_abbr}")

    # Position count — lightweight: read only the symbol column
    n_pos = 0
    with contextlib.suppress(Exception):
        import polars as pl
        n_pos = pl.read_parquet(tw, columns=["symbol"])["symbol"].n_unique()

    status = "STALE" if age_h > 48 else "LIVE"
    color  = COLORS["warning"] if age_h > 48 else COLORS["positive"]
    return status, color, last_run, next_run, n_pos


status_label, status_color, last_run, next_run, n_positions = _probe()

# ── Navigation ─────────────────────────────────────────────────────────────────

pg = st.navigation(
    {
        "Data": [
            st.Page("reports/health_dashboard.py",
                    title="Pipeline Health",       icon="🔧"),
            st.Page("reports/data_lab.py",
                    title="Data Lab",              icon="🧪"),
        ],
        "Research": [
            st.Page("reports/factor_analysis_dashboard.py",
                    title="Factor Analysis",       icon="📊"),
            st.Page("reports/signal_analysis_dashboard.py",
                    title="Signal Analysis",       icon="🔬"),
            st.Page("reports/kalman_dashboard.py",
                    title="Kalman Filter",         icon="📡"),
            st.Page("reports/time_series_dashboard.py",
                    title="Time Series",           icon="〰️"),
        ],
        "Strategy": [
            st.Page("reports/strategy_lab.py",
                    title="Strategy Lab",          icon="⚗️"),
            st.Page("reports/walk_forward_dashboard.py",
                    title="Walk-Forward",          icon="🔁"),
            st.Page("reports/monte_carlo_dashboard.py",
                    title="Monte Carlo",           icon="🎲"),
        ],
        "Portfolio": [
            st.Page("reports/multi_strategy_dashboard.py",
                    title="Multi-Strategy",        icon="💼"),
            st.Page("reports/deployment_dashboard.py",
                    title="Deployment",            icon="🚀"),
            st.Page("reports/performance_dashboard.py",
                    title="Performance",           icon="📈"),
        ],
        "Trading": [
            st.Page("reports/paper_trading_dashboard.py",
                    title="Paper Trading",         icon="📄"),
            st.Page("reports/live_trading_dashboard.py",
                    title="Live Trading",          icon="⚡"),
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
    if _LOGO.exists() and _LOGO_WORDS.exists():
        from PIL import Image
        import numpy as np

        def _crop_b64(path: Path, pad: int = 6, threshold: int = 20) -> str:
            img = Image.open(path).convert('RGBA')
            a = np.array(img)[:, :, 3]
            rows = np.where(a.max(axis=1) > threshold)[0]
            cols = np.where(a.max(axis=0) > threshold)[0]
            box = (
                max(0, int(cols[0])  - pad),
                max(0, int(rows[0])  - pad),
                min(img.width,  int(cols[-1]) + pad),
                min(img.height, int(rows[-1]) + pad),
            )
            buf = io.BytesIO()
            img.crop(box).save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()

        _c = _crop_b64(_LOGO)       # circle only
        _w = _crop_b64(_LOGO_WORDS) # words only

        # Fix the background behind the logo to the sidebar's endpoint colour
        # so the transparent PNG edges always blend into the same base colour,
        # regardless of where the sidebar gradient sits at that scroll position.
        _sb_end = COLORS['surface']   # #0F1325 — bottom of sidebar gradient
        st.markdown(f"""
<div style="text-align:center; padding:8px 0 4px;
            background:linear-gradient(180deg,transparent 0%,{_sb_end} 40%);">
  <img src="data:image/png;base64,{_c}"
       style="width:68%; display:block; margin:0 auto;
              filter:drop-shadow(0 0 12px rgba(201,162,39,0.35))
                     drop-shadow(0 0 28px rgba(201,162,39,0.15));
              -webkit-mask-image:radial-gradient(
                circle at 50% 48%,
                black 38%, rgba(0,0,0,0.95) 50%,
                rgba(0,0,0,0.7) 62%, rgba(0,0,0,0.3) 76%,
                transparent 90%);
              mask-image:radial-gradient(
                circle at 50% 48%,
                black 38%, rgba(0,0,0,0.95) 50%,
                rgba(0,0,0,0.7) 62%, rgba(0,0,0,0.3) 76%,
                transparent 90%);"/>
  <img src="data:image/png;base64,{_w}"
       style="width:84%; display:block; margin:2px auto 0;
              filter:drop-shadow(0 0 6px rgba(201,162,39,0.28));"/>
</div>
""", unsafe_allow_html=True)

    _pos_str = f"{n_positions} position{'s' if n_positions != 1 else ''}" if n_positions else "no positions"
    st.markdown(f"""
<div style="border-top:1px solid {COLORS['border']};margin:8px 4px 0;padding-top:10px;">

  <!-- Status + positions row -->
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:6px;">
    <div style="display:flex;align-items:center;gap:6px;">
      <span style="width:7px;height:7px;border-radius:50%;
                   background:{status_color};display:inline-block;flex-shrink:0;"></span>
      <span style="color:{status_color};font-size:0.72rem;font-weight:700;
                   letter-spacing:0.06em;">{status_label}</span>
    </div>
    <span style="color:{COLORS['neutral']};font-size:0.68rem;font-weight:600;">
      {_pos_str}
    </span>
  </div>

  <!-- Last run -->
  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
    <span style="color:{COLORS['text_muted']};font-size:0.63rem;
                 text-transform:uppercase;letter-spacing:0.07em;">Last run</span>
    <span style="color:{COLORS['neutral']};font-size:0.63rem;">{last_run}</span>
  </div>

  <!-- Next run -->
  <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
    <span style="color:{COLORS['text_muted']};font-size:0.63rem;
                 text-transform:uppercase;letter-spacing:0.07em;">Next run</span>
    <span style="color:{COLORS['gold_dim']};font-size:0.63rem;">{next_run}</span>
  </div>

  <div style="color:{COLORS['text_muted']};font-size:0.59rem;
              text-align:center;letter-spacing:0.02em;
              border-top:1px solid {COLORS['border_dim']};padding-top:6px;">
    v0.1 · Paper trading · Not investment advice
  </div>
</div>
""", unsafe_allow_html=True)

# ── Run selected page ──────────────────────────────────────────────────────────

pg.run()
