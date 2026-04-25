"""Shared Plotly theme, CSS, and HTML components for QuantPipe dashboards.

Brand palette — derived from the QuantPipe logo:
  Gold   #C9A227  ring border, wordmark — primary UI accent
  Green  #00E676  the pipe / equity curve — positive / bullish
  Purple #7B5EA7  logo interior depth — tertiary accent
  Void   #05070F  deepest background — chart paper
  Navy   #0A0D1A  app background

All dashboard files reference COLORS keys only — swap here, changes everywhere.
"""

# ── Colour palette ─────────────────────────────────────────────────────────────

COLORS = {
    # Backgrounds — 5-tier depth stack
    "bg":           "#0A0D1A",   # Streamlit app / main content
    "bg_void":      "#05070F",   # Plotly paper — deepest black
    "surface":      "#0F1325",   # Sidebar, secondary panels
    "card_bg":      "#141C35",   # Card / input surfaces
    "bg_hover":     "#1B2445",   # Hover / selected rows

    # Borders
    "border":       "rgba(201,162,39,0.20)",  # standard gold-tinted
    "border_dim":   "rgba(201,162,39,0.10)",  # subtle
    "border_focus": "#C9A227",               # active / focused

    # Gold — primary accent
    "gold":         "#C9A227",
    "gold_bright":  "#D4AE42",
    "gold_dim":     "#8A6B18",
    "gold_bg":      "rgba(201,162,39,0.10)",

    # Green — pipe / positive / bullish
    "green":        "#00E676",
    "green_bright": "#33EE8F",
    "green_dim":    "#00A854",
    "green_bg":     "rgba(0,230,118,0.07)",

    # Purple — brand depth / tertiary
    "purple":       "#7B5EA7",
    "purple_bright":"#9B7EC7",
    "purple_dim":   "#5B3E87",
    "purple_bg":    "rgba(123,94,167,0.12)",

    # Semantic
    "positive":     "#00E676",   # gains, bullish
    "negative":     "#FF4D4D",   # losses, bearish
    "warning":      "#C9A227",   # stale / caution (gold)
    "info":         "#4A90D9",   # informational blue

    # Typography
    "text":         "#EEF2FF",   # primary (blue-white)
    "text_muted":   "#5A6478",   # muted
    "neutral":      "#A8B3CC",   # secondary / labels

    # Legacy aliases — dashboards import these by name
    "blue":         "#4A90D9",
    "orange":       "#C9A227",
    "teal":         "#00E676",

    # Chart series — brand-matched 8-colour rotation
    "series": [
        "#C9A227",   # 1  Gold
        "#00E676",   # 2  Green / pipe
        "#7B5EA7",   # 3  Purple
        "#4A90D9",   # 4  Blue
        "#FF4D4D",   # 5  Red
        "#F5A623",   # 6  Amber
        "#50E3C2",   # 7  Teal
        "#BD10E0",   # 8  Magenta
    ],
}

# ── Plotly base layout ─────────────────────────────────────────────────────────

_GRID = "rgba(30,38,100,0.45)"   # indigo-tinted grid — subtle but visible

_LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg_void"],
    plot_bgcolor="#0D1128",          # slightly purple-tinted inner bg
    font=dict(
        color=COLORS["neutral"],
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
        size=12,
    ),
    xaxis=dict(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor="rgba(201,162,39,0.15)",
        tickcolor=COLORS["text_muted"],
        zerolinecolor="rgba(201,162,39,0.15)", zerolinewidth=1,
        tickfont=dict(color=COLORS["text_muted"], size=11),
    ),
    yaxis=dict(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor="rgba(201,162,39,0.15)",
        tickcolor=COLORS["text_muted"],
        zerolinecolor="rgba(201,162,39,0.15)", zerolinewidth=1,
        tickfont=dict(color=COLORS["text_muted"], size=11),
    ),
    margin=dict(l=8, r=8, t=38, b=8),
    legend=dict(
        bgcolor="rgba(5,7,15,0.70)",
        bordercolor="rgba(201,162,39,0.20)",
        borderwidth=1,
        font=dict(color=COLORS["neutral"], size=11),
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left", x=0,
    ),
    hoverlabel=dict(
        bgcolor="#141C35",
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
        text=f"<b style='color:{COLORS['text_muted']};font-size:12px;"
             f"letter-spacing:0.06em;text-transform:uppercase;'>{title}</b>",
        x=0.01, y=0.98,
        xanchor="left", yanchor="top",
        pad=dict(t=4),
    )
    if height:
        layout["height"] = height
    if legend_inside:
        layout["legend"] = dict(
            bgcolor="rgba(20,28,53,0.90)",
            bordercolor="rgba(201,162,39,0.25)",
            borderwidth=1,
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
    _ax = dict(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor="rgba(201,162,39,0.15)",
        tickcolor=COLORS["text_muted"],
        zerolinecolor="rgba(201,162,39,0.15)",
        tickfont=dict(color=COLORS["text_muted"], size=11),
    )
    fig.update_xaxes(**_ax)
    fig.update_yaxes(**_ax)


def range_selector() -> dict:
    return dict(
        buttons=[
            dict(count=3,  label="3M",  step="month", stepmode="backward"),
            dict(count=6,  label="6M",  step="month", stepmode="backward"),
            dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
            dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor="#141C35",
        activecolor=COLORS["gold"],
        font=dict(color=COLORS["neutral"], size=11),
        bordercolor="rgba(201,162,39,0.25)",
        borderwidth=1,
    )


# ── CSS stylesheet ─────────────────────────────────────────────────────────────

CSS = f"""
<style>

/* ══════════════════════════════════════════════════════════════════════════════
   GLOBAL RESET / BASE
══════════════════════════════════════════════════════════════════════════════ */
html, body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}}

/* Main app background — target every layer Streamlit uses */
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section[data-testid="stMain"] {{
    background-color: {COLORS['bg']} !important;
}}
/* Ambient brand glows — applied to the outermost layer */
.stApp,
[data-testid="stApp"] {{
    background:
        radial-gradient(ellipse 55% 35% at 12% 0%,
            rgba(107,47,160,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 45% 30% at 88% 100%,
            rgba(201,162,39,0.07) 0%, transparent 60%),
        {COLORS['bg']} !important;
}}
[data-testid="block-container"] {{
    padding-top: 24px;
    max-width: 1440px;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg_void']}; }}
::-webkit-scrollbar-thumb {{
    background: rgba(201,162,39,0.25);
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS['gold']}; }}

/* ══════════════════════════════════════════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="metric-container"] {{
    background: linear-gradient(145deg, {COLORS['card_bg']} 0%, {COLORS['surface']} 100%) !important;
    border: 1px solid rgba(201,162,39,0.20) !important;
    border-top: 2px solid {COLORS['gold']} !important;
    border-radius: 2px 2px 12px 12px !important;
    padding: 16px 18px 14px !important;
    box-shadow:
        0 4px 24px rgba(0,0,0,0.50),
        inset 0 1px 0 rgba(201,162,39,0.08) !important;
    transition: border-color 0.20s ease, box-shadow 0.20s ease !important;
    min-width: 0;
    overflow: hidden;
}}
[data-testid="metric-container"]:hover {{
    border-color: rgba(201,162,39,0.45) !important;
    border-top-color: {COLORS['gold_bright']} !important;
    box-shadow:
        0 6px 32px rgba(0,0,0,0.60),
        0 0 24px rgba(201,162,39,0.14),
        inset 0 1px 0 rgba(201,162,39,0.14) !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.40rem !important;
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
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-weight: 600;
    margin-bottom: 5px;
    line-height: 1.35;
    overflow-wrap: break-word;
}}
[data-testid="stMetricDelta"] svg {{ display: none; }}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.77rem !important;
    font-weight: 600;
    letter-spacing: 0.01em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   TABS
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: linear-gradient(180deg, #0D1020 0%, {COLORS['surface']} 100%);
    border-bottom: 1px solid rgba(201,162,39,0.15);
    gap: 0;
    padding: 0 4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    color: {COLORS['text_muted']};
    font-size: 0.82rem;
    font-weight: 500;
    padding: 11px 20px;
    background: transparent;
    border-bottom: 2px solid transparent;
    transition: color 0.15s ease, border-color 0.15s ease;
    letter-spacing: 0.02em;
}}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {{
    color: {COLORS['neutral']};
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {COLORS['gold']} !important;
    font-weight: 600 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background: linear-gradient(90deg, {COLORS['gold']}, {COLORS['gold_bright']}) !important;
    height: 2px !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none !important; }}

/* ══════════════════════════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1020 0%, {COLORS['surface']} 100%) !important;
    border-right: 1px solid rgba(201,162,39,0.12) !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: {COLORS['text_muted']};
    font-size: 0.77rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 2px;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   BUTTONS — all Streamlit button variants share the same gold style
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="stButton"] > button,
[data-testid="stDownloadButton"] > button,
[data-testid="stLinkButton"] > a {{
    background: transparent !important;
    border: 1px solid {COLORS['gold']} !important;
    color: {COLORS['gold']} !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
    padding: 7px 20px !important;
    transition: background 0.18s ease, box-shadow 0.18s ease, color 0.18s ease !important;
    box-shadow: none !important;
    text-decoration: none !important;
}}
[data-testid="stButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover,
[data-testid="stLinkButton"] > a:hover {{
    background: {COLORS['gold']} !important;
    color: {COLORS['bg_void']} !important;
    box-shadow: 0 0 20px rgba(201,162,39,0.28) !important;
}}
[data-testid="stButton"] > button:disabled,
[data-testid="stDownloadButton"] > button:disabled {{
    opacity: 0.35 !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   INPUTS — select, text, number, slider
══════════════════════════════════════════════════════════════════════════════ */
[data-baseweb="select"] [data-baseweb="select-control"] {{
    background: {COLORS['card_bg']} !important;
    border-color: rgba(201,162,39,0.22) !important;
    transition: border-color 0.15s ease !important;
}}
[data-baseweb="select"] [data-baseweb="select-control"]:hover,
[data-baseweb="select"] [data-baseweb="select-control"]:focus-within {{
    border-color: {COLORS['gold']} !important;
}}
[data-baseweb="select"] [data-baseweb="menu"] {{
    background: {COLORS['card_bg']} !important;
    border: 1px solid rgba(201,162,39,0.22) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.60) !important;
}}
[data-baseweb="select"] [role="option"]:hover {{
    background: {COLORS['gold_bg']} !important;
}}
[data-baseweb="select"] [aria-selected="true"] {{
    background: rgba(201,162,39,0.15) !important;
    color: {COLORS['gold']} !important;
}}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
input[type="number"] {{
    background: {COLORS['card_bg']} !important;
    border-color: rgba(201,162,39,0.22) !important;
    color: {COLORS['text']} !important;
    border-radius: 6px !important;
}}
[data-baseweb="input"] input:focus,
[data-baseweb="textarea"] textarea:focus {{
    border-color: {COLORS['gold']} !important;
    box-shadow: 0 0 0 2px rgba(201,162,39,0.15) !important;
    outline: none !important;
}}
/* Slider thumb + fill */
[data-testid="stSlider"] [role="slider"] {{
    background: {COLORS['gold']} !important;
    border-color: {COLORS['gold']} !important;
    box-shadow: 0 0 8px rgba(201,162,39,0.45) !important;
}}
[data-testid="stSlider"] [data-testid="stTickBar"] {{
    background: {COLORS['gold_dim']} !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   CHECKBOXES / RADIOS / TOGGLES
══════════════════════════════════════════════════════════════════════════════ */
/* Checkbox — aria-checked is standard and stable */
[data-baseweb="checkbox"] [aria-checked="true"] {{
    background: {COLORS['gold']} !important;
    border-color: {COLORS['gold']} !important;
}}
[data-baseweb="radio"] [aria-checked="true"] {{
    border-color: {COLORS['gold']} !important;
    background: {COLORS['gold']} !important;
}}
/* Toggle — role="switch" is the stable ARIA pattern */
[role="switch"][aria-checked="true"] {{
    background: {COLORS['gold']} !important;
    border-color: {COLORS['gold_dim']} !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   DATAFRAME / TABLE
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {{
    border: 1px solid rgba(201,162,39,0.18) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
/* Column headers */
[data-testid="stDataFrame"] th {{
    background: {COLORS['card_bg']} !important;
    color: {COLORS['gold']} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    border-bottom: 1px solid rgba(201,162,39,0.20) !important;
}}
[data-testid="stDataFrame"] td {{
    color: {COLORS['neutral']} !important;
    font-size: 0.83rem !important;
    border-color: rgba(201,162,39,0.08) !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {{
    background: linear-gradient(145deg, {COLORS['card_bg']} 0%, {COLORS['surface']} 100%) !important;
    border: 1px solid rgba(201,162,39,0.18) !important;
    border-radius: 10px !important;
    transition: border-color 0.15s ease !important;
}}
[data-testid="stExpander"]:hover {{
    border-color: rgba(201,162,39,0.35) !important;
}}
[data-testid="stExpander"] summary {{
    color: {COLORS['neutral']};
    font-size: 0.83rem;
    font-weight: 500;
    padding: 12px 16px;
}}
[data-testid="stExpander"] summary:hover {{
    color: {COLORS['text']};
}}

/* ══════════════════════════════════════════════════════════════════════════════
   ALERT BOXES
══════════════════════════════════════════════════════════════════════════════ */
/* Alerts — Streamlit exposes type via data-testid on the inner notification */
[data-testid="stAlert"] {{
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}}
[data-testid="stNotification"] {{
    border-radius: 8px !important;
    border-width: 1px !important;
    border-style: solid !important;
}}
/* Info */
[data-testid="stNotification"][data-type="info"],
div[data-testid="stAlert"] div[role="alert"][class*="info"] {{
    background: rgba(74,144,217,0.09) !important;
    border-color: rgba(74,144,217,0.38) !important;
}}
/* Warning */
[data-testid="stNotification"][data-type="warning"],
div[data-testid="stAlert"] div[role="alert"][class*="warning"],
div[data-testid="stAlert"] div[class*="Warning"] {{
    background: rgba(201,162,39,0.09) !important;
    border-color: rgba(201,162,39,0.38) !important;
}}
/* Success */
[data-testid="stNotification"][data-type="success"],
div[data-testid="stAlert"] div[role="alert"][class*="success"],
div[data-testid="stAlert"] div[class*="Success"] {{
    background: rgba(0,230,118,0.09) !important;
    border-color: rgba(0,230,118,0.38) !important;
}}
/* Error */
[data-testid="stNotification"][data-type="error"],
div[data-testid="stAlert"] div[role="alert"][class*="error"],
div[data-testid="stAlert"] div[class*="Error"] {{
    background: rgba(255,77,77,0.09) !important;
    border-color: rgba(255,77,77,0.38) !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   DIVIDER / HR
══════════════════════════════════════════════════════════════════════════════ */
hr {{
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg, transparent 0%, rgba(201,162,39,0.30) 20%,
        rgba(201,162,39,0.30) 80%, transparent 100%) !important;
    margin: 20px 0 !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   SPINNER
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="stSpinner"] {{ color: {COLORS['gold']}; }}

/* ══════════════════════════════════════════════════════════════════════════════
   CODE BLOCKS
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="stCode"] pre {{
    background: linear-gradient(145deg, #0D1128, {COLORS['surface']}) !important;
    border: 1px solid rgba(201,162,39,0.18) !important;
    border-radius: 8px !important;
    font-size: 0.79rem !important;
    color: {COLORS['neutral']} !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   PROGRESS BAR
══════════════════════════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div > div > div {{
    background: linear-gradient(90deg, {COLORS['gold_dim']}, {COLORS['gold']}) !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   SIDEBAR LOGO / DIVIDER
══════════════════════════════════════════════════════════════════════════════ */
.qp-logo-divider {{
    height: 1px;
    background: linear-gradient(90deg, {COLORS['gold']}50, transparent);
    margin: 8px 0 16px;
}}
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
}}

/* ══════════════════════════════════════════════════════════════════════════════
   ANIMATIONS
══════════════════════════════════════════════════════════════════════════════ */
@keyframes pulse-critical {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.40; }}
}}
@keyframes gold-pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(201,162,39,0); }}
    50%      {{ box-shadow: 0 0 20px 4px rgba(201,162,39,0.22); }}
}}
@keyframes green-pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(0,230,118,0); }}
    50%      {{ box-shadow: 0 0 16px 3px rgba(0,230,118,0.20); }}
}}
.qp-pulse      {{ animation: pulse-critical 1.8s ease-in-out infinite; }}
.qp-gold-glow  {{ animation: gold-pulse 2.5s ease-in-out infinite; }}
.qp-green-glow {{ animation: green-pulse 2.5s ease-in-out infinite; }}

</style>
"""


# ── HTML component helpers ─────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "", right_text: str = "") -> str:
    right = (
        f'<div style="text-align:right;flex-shrink:0;">'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;text-transform:uppercase;'
        f'letter-spacing:0.09em;font-weight:600;">As of</div>'
        f'<div style="color:{COLORS["text"]};font-size:0.95rem;font-weight:700;'
        f'margin-top:1px;">{right_text}</div>'
        f'</div>'
        if right_text else ""
    )
    sub = (
        f'<p style="margin:4px 0 0;color:{COLORS["neutral"]};font-size:0.82rem;'
        f'letter-spacing:0.01em;">{subtitle}</p>'
        if subtitle else ""
    )
    return (
        f'<div style="margin-bottom:22px;">'
        f'<div style="display:flex;align-items:flex-end;justify-content:space-between;'
        f'padding-bottom:14px;">'
        f'<div>'
        f'<h1 style="margin:0;font-size:1.70rem;font-weight:800;color:{COLORS["text"]};'
        f'letter-spacing:-0.04em;line-height:1.15;">{title}</h1>'
        f'{sub}'
        f'</div>'
        f'{right}'
        f'</div>'
        f'<div style="height:1px;background:linear-gradient(90deg,'
        f'{COLORS["gold"]}70,rgba(201,162,39,0.35) 40%,transparent);"></div>'
        f'</div>'
    )


def kpi_card(label: str, value: str, delta: str = "",
             delta_up: bool | None = None,
             accent: str | None = None) -> str:
    top = accent or COLORS["gold"]
    if delta and delta_up is not None:
        dcol = COLORS["positive"] if delta_up else COLORS["negative"]
        arrow = "▲" if delta_up else "▼"
        d_html = (
            f'<div style="color:{dcol};font-size:0.76rem;font-weight:700;'
            f'margin-top:8px;letter-spacing:0.01em;">{arrow} {delta}</div>'
        )
    else:
        d_html = ""
    return (
        f'<div style="'
        f'background:linear-gradient(145deg,{COLORS["card_bg"]} 0%,{COLORS["surface"]} 100%);'
        f'border:1px solid rgba(201,162,39,0.20);'
        f'border-top:2px solid {top};'
        f'border-radius:2px 2px 12px 12px;'
        f'padding:16px 18px 14px;'
        f'min-width:0;overflow:hidden;'
        f'box-shadow:0 4px 24px rgba(0,0,0,0.50),inset 0 1px 0 rgba(201,162,39,0.07);">'
        f'<div style="color:{COLORS["neutral"]};font-size:0.67rem;text-transform:uppercase;'
        f'letter-spacing:0.10em;font-weight:600;margin-bottom:8px;'
        f'line-height:1.35;overflow-wrap:break-word;">{label}</div>'
        f'<div style="color:{COLORS["text"]};font-size:1.32rem;font-weight:700;'
        f'letter-spacing:-0.02em;line-height:1.2;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{value}</div>'
        f'{d_html}'
        f'</div>'
    )


def section_label(text: str) -> str:
    return (
        f'<div style="'
        f'color:{COLORS["neutral"]};font-size:0.67rem;text-transform:uppercase;'
        f'letter-spacing:0.10em;font-weight:700;margin:22px 0 13px;'
        f'border-left:3px solid {COLORS["gold"]};'
        f'padding-left:10px;'
        f'background:linear-gradient(90deg,{COLORS["gold_bg"]},transparent);'
        f'line-height:1.8;overflow-wrap:break-word;">{text}</div>'
    )


def badge(text: str, variant: str = "neutral") -> str:
    palette = {
        "positive": (COLORS["green"],        "rgba(0,230,118,0.12)"),
        "negative": (COLORS["negative"],     "rgba(255,77,77,0.12)"),
        "warning":  (COLORS["gold"],         "rgba(201,162,39,0.12)"),
        "neutral":  (COLORS["neutral"],      COLORS["card_bg"]),
        "blue":     (COLORS["blue"],         "rgba(74,144,217,0.12)"),
        "gold":     (COLORS["gold"],         "rgba(201,162,39,0.12)"),
        "purple":   (COLORS["purple"],       COLORS["purple_bg"]),
    }
    fg, bg = palette.get(variant, palette["neutral"])
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg}44;'
        f'border-radius:4px;padding:3px 12px;font-size:0.76rem;'
        f'font-weight:700;font-family:monospace;letter-spacing:0.06em;">'
        f'{text}</span>'
    )


def status_banner(text: str, color: str, animate: bool = False) -> str:
    cls = ' class="qp-pulse"' if animate else ""
    return (
        f'<div style="'
        f'background:{color}12;'
        f'border:1px solid {color}60;'
        f'border-left:3px solid {color};'
        f'border-radius:8px;'
        f'padding:13px 20px;margin-bottom:20px;'
        f'display:flex;align-items:center;gap:12px;">'
        f'<span{cls} style="font-size:0.95rem;font-weight:700;color:{color};">'
        f'● {text}</span>'
        f'</div>'
    )
