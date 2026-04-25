"""Shared Plotly theme, CSS, and HTML components for QuantPipe dashboards.

Brand palette extracted from the QuantPipe logo:
  Gold   #C9A227 — border ring, wordmark
  Green  #00E676 — the pipe / equity curve (positive)
  Purple #6B2FA0 — circular logo background (secondary accent)
  Navy   #0A0D1A — overall dark background
  Red    #FF4444 — bearish candles (negative)
"""

# ── Colour palette ─────────────────────────────────────────────────────────────

COLORS = {
    # Backgrounds
    "bg":         "#0A0D1A",
    "surface":    "#0F1320",
    "card_bg":    "#141928",

    # Borders
    "border":     "#1E2640",
    "border_dim": "#161C30",

    # Brand accents
    "gold":       "#C9A227",   # primary UI accent — tabs, active states, KPI borders
    "green":      "#00E676",   # positive returns, bullish signals
    "purple":     "#6B2FA0",   # secondary accent
    "gold_dim":   "#8A6B18",   # dimmed gold for hover states

    # Semantic
    "positive":   "#00E676",   # matches brand green
    "negative":   "#FF4444",   # bearish red from logo candles
    "warning":    "#C9A227",   # gold doubles as warning
    "info":       "#7B5EA7",   # muted purple

    # Text
    "text":       "#E8EDF5",
    "text_muted": "#5A6478",
    "neutral":    "#8892A4",

    # Legacy aliases (kept for backwards compat)
    "blue":       "#4A90D9",
    "orange":     "#C9A227",
    "teal":       "#00E676",

    # Chart series (gold → green → purple → … )
    "series": [
        "#C9A227",  # gold
        "#00E676",  # green
        "#6B2FA0",  # purple
        "#4A90D9",  # blue
        "#FF4444",  # red
        "#F5A623",  # amber
        "#50E3C2",  # teal
        "#BD10E0",  # magenta
    ],
}

# ── Plotly base layout ─────────────────────────────────────────────────────────

_GRID   = "#1A2038"
_LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card_bg"],
    font=dict(
        color=COLORS["text"],
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
        size=12,
    ),
    xaxis=dict(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor=COLORS["border"], tickcolor=COLORS["neutral"],
        zerolinecolor=_GRID, zerolinewidth=1,
        tickfont=dict(color=COLORS["neutral"], size=11),
    ),
    yaxis=dict(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor=COLORS["border"], tickcolor=COLORS["neutral"],
        zerolinecolor=_GRID, zerolinewidth=1,
        tickfont=dict(color=COLORS["neutral"], size=11),
    ),
    margin=dict(l=8, r=8, t=38, b=8),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=COLORS["border"],
        font=dict(color=COLORS["neutral"], size=11),
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left", x=0,
    ),
    hoverlabel=dict(
        bgcolor=COLORS["surface"],
        bordercolor=COLORS["gold"],
        font=dict(color=COLORS["text"], size=12),
    ),
    hovermode="x unified",
    dragmode="pan",
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "quantpipe"},
}


def apply_theme(fig, title: str = "", height: int | None = None,
                legend_inside: bool = False) -> None:
    layout = dict(_LAYOUT_BASE)
    layout["title"] = dict(
        text=f"<b style='color:{COLORS['neutral']};font-size:12px;'>{title}</b>",
        x=0.01, y=0.98,
        xanchor="left", yanchor="top",
        pad=dict(t=4),
    )
    if height:
        layout["height"] = height
    if legend_inside:
        layout["legend"] = dict(
            bgcolor="rgba(20,25,40,0.88)",
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["neutral"], size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        )
    fig.update_layout(**layout)


def apply_subplot_theme(fig, height: int | None = None) -> None:
    layout = {k: v for k, v in _LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")}
    layout["title"] = dict(text="", x=0)
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor=COLORS["border"], tickcolor=COLORS["neutral"],
        zerolinecolor=_GRID,
        tickfont=dict(color=COLORS["neutral"], size=11),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor=COLORS["border"], tickcolor=COLORS["neutral"],
        zerolinecolor=_GRID,
        tickfont=dict(color=COLORS["neutral"], size=11),
    )


def range_selector() -> dict:
    return dict(
        buttons=[
            dict(count=3,  label="3M",  step="month", stepmode="backward"),
            dict(count=6,  label="6M",  step="month", stepmode="backward"),
            dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
            dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor=COLORS["card_bg"],
        activecolor=COLORS["gold"],
        font=dict(color=COLORS["text"], size=11),
        bordercolor=COLORS["border"],
        borderwidth=1,
    )


# ── CSS stylesheet ─────────────────────────────────────────────────────────────

CSS = f"""
<style>
/* ── Reset / global ─────────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] {{
    background-color: {COLORS['bg']} !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}}
[data-testid="stAppViewContainer"] > .main {{
    background-color: {COLORS['bg']};
}}
[data-testid="block-container"] {{
    padding-top: 24px;
    max-width: 1400px;
}}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS['gold_dim']}; }}

/* ── st.metric cards ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: {COLORS['card_bg']};
    border: 1px solid {COLORS['border']};
    border-top: 2px solid {COLORS['gold']};
    border-radius: 2px 2px 10px 10px;
    padding: 16px 18px 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.45);
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
    min-width: 0;
    overflow: hidden;
}}
[data-testid="metric-container"]:hover {{
    border-color: {COLORS['gold']};
    box-shadow: 0 4px 20px rgba(201,162,39,0.15);
}}
[data-testid="stMetricValue"] {{
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: {COLORS['text']} !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
[data-testid="stMetricLabel"] {{
    color: {COLORS['neutral']} !important;
    font-size: 0.69rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 4px;
    line-height: 1.35;
    overflow-wrap: break-word;
}}
[data-testid="stMetricDelta"] svg {{ display: none; }}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.78rem !important;
    font-weight: 500;
    letter-spacing: 0.01em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {COLORS['surface']};
    border-bottom: 1px solid {COLORS['border']};
    gap: 0;
    padding: 0 4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    color: {COLORS['neutral']};
    font-size: 0.83rem;
    font-weight: 500;
    padding: 11px 18px;
    background: transparent;
    border-bottom: 2px solid transparent;
    transition: color 0.15s ease, border-color 0.15s ease;
    letter-spacing: 0.01em;
}}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {{ color: {COLORS['text']}; }}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {COLORS['gold']} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background-color: {COLORS['gold']} !important;
    height: 2px !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none !important; }}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {COLORS['surface']} !important;
    border-right: 1px solid {COLORS['border']};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: {COLORS['neutral']};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
    margin-bottom: 2px;
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {{
    background: transparent;
    border: 1px solid {COLORS['gold']};
    color: {COLORS['gold']};
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: background 0.15s ease, color 0.15s ease;
}}
[data-testid="stButton"] > button:hover {{
    background: {COLORS['gold']};
    color: {COLORS['bg']};
}}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {COLORS['card_bg']};
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px;
}}
[data-testid="stExpander"] summary {{
    color: {COLORS['neutral']};
    font-size: 0.83rem;
    font-weight: 500;
}}

/* ── Info / alert boxes ──────────────────────────────────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 8px;
    font-size: 0.85rem;
}}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid {COLORS['border']} !important;
    margin: 18px 0;
}}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {{ color: {COLORS['gold']}; }}

/* ── Code blocks ─────────────────────────────────────────────────────────── */
[data-testid="stCode"] pre {{
    background: {COLORS['surface']} !important;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    font-size: 0.78rem;
    color: {COLORS['text_muted']};
}}

/* ── Select / radio / toggle accent ─────────────────────────────────────── */
[data-baseweb="select"] [data-baseweb="select-control"] {{
    background: {COLORS['card_bg']} !important;
    border-color: {COLORS['border']} !important;
}}

/* ── Sidebar logo separator ─────────────────────────────────────────────── */
.qp-logo-divider {{
    height: 1px;
    background: linear-gradient(90deg, {COLORS['gold']}40, transparent);
    margin: 8px 0 18px;
}}

/* ── Animations ──────────────────────────────────────────────────────────── */
@keyframes pulse-critical {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.45; }}
}}
.qp-pulse {{ animation: pulse-critical 1.8s ease-in-out infinite; }}

/* ── Gold glow effect for critical KPIs ──────────────────────────────────── */
@keyframes gold-pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(201,162,39,0); }}
    50%      {{ box-shadow: 0 0 16px 2px rgba(201,162,39,0.25); }}
}}
.qp-gold-glow {{ animation: gold-pulse 2.5s ease-in-out infinite; }}
</style>
"""


# ── HTML component helpers ─────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "", right_text: str = "") -> str:
    right = (
        f'<div style="text-align:right;">'
        f'<div style="color:{COLORS["neutral"]};font-size:0.7rem;text-transform:uppercase;'
        f'letter-spacing:0.08em;">As of</div>'
        f'<div style="color:{COLORS["text"]};font-size:0.95rem;font-weight:600;">{right_text}</div>'
        f'</div>'
        if right_text else ""
    )
    sub = (
        f'<p style="margin:3px 0 0;color:{COLORS["neutral"]};font-size:0.82rem;">{subtitle}</p>'
        if subtitle else ""
    )
    return (
        f'<div style="display:flex;align-items:flex-end;justify-content:space-between;'
        f'padding-bottom:14px;border-bottom:1px solid {COLORS["border"]};margin-bottom:20px;">'
        f'<div>'
        f'<h1 style="margin:0;font-size:1.65rem;font-weight:800;color:{COLORS["text"]};'
        f'letter-spacing:-0.04em;">{title}</h1>'
        f'{sub}'
        f'</div>'
        f'{right}'
        f'</div>'
    )


def kpi_card(label: str, value: str, delta: str = "",
             delta_up: bool | None = None,
             accent: str | None = None) -> str:
    top_border = accent or COLORS["gold"]
    if delta and delta_up is not None:
        dcolor = COLORS["positive"] if delta_up else COLORS["negative"]
        arrow = "▲" if delta_up else "▼"
        delta_html = (
            f'<div style="color:{dcolor};font-size:0.77rem;font-weight:600;'
            f'margin-top:7px;letter-spacing:0.01em;">{arrow} {delta}</div>'
        )
    else:
        delta_html = ""
    return (
        f'<div style="background:{COLORS["card_bg"]};'
        f'border:1px solid {COLORS["border"]};'
        f'border-top:3px solid {top_border};'
        f'border-radius:2px 2px 10px 10px;'
        f'padding:16px 18px 14px;'
        f'min-width:0;overflow:hidden;'
        f'box-shadow:0 2px 10px rgba(0,0,0,0.4);">'
        f'<div style="color:{COLORS["neutral"]};font-size:0.68rem;text-transform:uppercase;'
        f'letter-spacing:0.08em;font-weight:600;margin-bottom:7px;'
        f'line-height:1.35;overflow-wrap:break-word;">{label}</div>'
        f'<div style="color:{COLORS["text"]};font-size:1.3rem;font-weight:700;'
        f'letter-spacing:-0.02em;line-height:1.2;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def section_label(text: str) -> str:
    return (
        f'<div style="color:{COLORS["neutral"]};font-size:0.68rem;text-transform:uppercase;'
        f'letter-spacing:0.09em;font-weight:700;margin:20px 0 12px;'
        f'border-left:3px solid {COLORS["gold"]};padding-left:10px;'
        f'line-height:1.4;overflow-wrap:break-word;">{text}</div>'
    )


def badge(text: str, variant: str = "neutral") -> str:
    palette = {
        "positive": ("#00E676", "#003D1A"),
        "negative": ("#FF4444", "#3D0000"),
        "warning":  ("#C9A227", "#2E2000"),
        "neutral":  ("#8892A4", "#141928"),
        "blue":     ("#4A90D9", "#0D1F3D"),
        "gold":     ("#C9A227", "#2E2000"),
        "purple":   ("#7B5EA7", "#1A0F2E"),
    }
    fg, bg = palette.get(variant, palette["neutral"])
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg};'
        f'border-radius:4px;padding:3px 12px;font-size:0.78rem;'
        f'font-weight:700;font-family:monospace;letter-spacing:0.05em;">'
        f'{text}</span>'
    )


def status_banner(text: str, color: str, animate: bool = False) -> str:
    cls = ' class="qp-pulse"' if animate else ""
    return (
        f'<div style="background:{color}18;border:1px solid {color};'
        f'border-radius:10px;padding:14px 22px;margin-bottom:22px;'
        f'display:flex;align-items:center;gap:14px;">'
        f'<span{cls} style="font-size:1rem;font-weight:700;color:{color};">● {text}</span>'
        f'</div>'
    )
