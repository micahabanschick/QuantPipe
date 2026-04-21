"""Shared Plotly theme, CSS, and HTML components for QuantPipe dashboards."""

# ── Colour palette ─────────────────────────────────────────────────────────────

COLORS = {
    "bg":         "#0e1117",
    "surface":    "#161b27",
    "card_bg":    "#1a1f2e",
    "border":     "#2d3748",
    "border_dim": "#1e2636",
    "positive":   "#00d4aa",
    "negative":   "#ff4b4b",
    "warning":    "#f6ad55",
    "info":       "#4299e1",
    "neutral":    "#8892a4",
    "text":       "#e2e8f0",
    "text_muted": "#64748b",
    "blue":       "#4299e1",
    "purple":     "#9f7aea",
    "orange":     "#ed8936",
    "teal":       "#00d4aa",
    "series": [
        "#00d4aa", "#4299e1", "#f6ad55", "#9f7aea",
        "#ed8936", "#68d391", "#fc8181", "#76e4f7",
    ],
}

# ── Plotly base layout ─────────────────────────────────────────────────────────

_LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card_bg"],
    font=dict(
        color=COLORS["text"],
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
        size=12,
    ),
    xaxis=dict(
        showgrid=True, gridcolor=COLORS["border"], gridwidth=1,
        linecolor=COLORS["border"], tickcolor=COLORS["neutral"],
        zerolinecolor=COLORS["border"], zerolinewidth=1,
        tickfont=dict(color=COLORS["neutral"], size=11),
    ),
    yaxis=dict(
        showgrid=True, gridcolor=COLORS["border"], gridwidth=1,
        linecolor=COLORS["border"], tickcolor=COLORS["neutral"],
        zerolinecolor=COLORS["border"], zerolinewidth=1,
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
        bordercolor=COLORS["border"],
        font=dict(color=COLORS["text"], size=12),
    ),
    hovermode="x unified",
    dragmode="pan",
)

# Plotly chart config (passed to st.plotly_chart)
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "quantpipe"},
}


def apply_theme(fig, title: str = "", height: int | None = None,
                legend_inside: bool = False) -> None:
    """Apply shared dark theme to a Plotly figure in-place."""
    layout = dict(_LAYOUT_BASE)
    layout["title"] = dict(
        text=f"<b style='color:{COLORS['neutral']};font-size:12px;'>{title}</b>",
        x=0.01,
        y=0.98,
        xanchor="left",
        yanchor="top",
        pad=dict(t=4),
    )
    if height:
        layout["height"] = height
    if legend_inside:
        layout["legend"] = dict(
            bgcolor="rgba(26,31,46,0.85)",
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["neutral"], size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        )
    fig.update_layout(**layout)


def apply_subplot_theme(fig, height: int | None = None) -> None:
    """Apply theme to a subplot figure (skips per-axis overrides)."""
    layout = {
        k: v for k, v in _LAYOUT_BASE.items()
        if k not in ("xaxis", "yaxis")
    }
    layout["title"] = dict(text="", x=0)
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(
        showgrid=True, gridcolor=COLORS["border"], gridwidth=1,
        linecolor=COLORS["border"], tickcolor=COLORS["neutral"],
        zerolinecolor=COLORS["border"],
        tickfont=dict(color=COLORS["neutral"], size=11),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=COLORS["border"], gridwidth=1,
        linecolor=COLORS["border"], tickcolor=COLORS["neutral"],
        zerolinecolor=COLORS["border"],
        tickfont=dict(color=COLORS["neutral"], size=11),
    )


def range_selector() -> dict:
    """Return a Plotly xaxis range-selector config."""
    return dict(
        buttons=[
            dict(count=3,  label="3M",  step="month", stepmode="backward"),
            dict(count=6,  label="6M",  step="month", stepmode="backward"),
            dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
            dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor=COLORS["card_bg"],
        activecolor=COLORS["positive"],
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
::-webkit-scrollbar-thumb:hover {{ background: #3d4f6e; }}

/* ── st.metric cards ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: {COLORS['card_bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 18px 20px 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    transition: border-color 0.18s ease;
}}
[data-testid="metric-container"]:hover {{ border-color: #3d4f6e; }}
[data-testid="stMetricValue"] {{
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: {COLORS['text']} !important;
    letter-spacing: -0.025em;
    line-height: 1.15;
}}
[data-testid="stMetricLabel"] {{
    color: {COLORS['neutral']} !important;
    font-size: 0.70rem !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 600;
    margin-bottom: 4px;
}}
[data-testid="stMetricDelta"] svg {{ display: none; }}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.80rem !important;
    font-weight: 500;
    letter-spacing: 0.01em;
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
    color: {COLORS['positive']} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background-color: {COLORS['positive']} !important;
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
hr {{ border: none !important; border-top: 1px solid {COLORS['border']} !important; margin: 6px 0; }}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {{ color: {COLORS['positive']}; }}

/* ── Code blocks ─────────────────────────────────────────────────────────── */
[data-testid="stCode"] pre {{
    background: {COLORS['surface']} !important;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    font-size: 0.78rem;
    color: {COLORS['text_muted']};
}}

/* ── Animations ──────────────────────────────────────────────────────────── */
@keyframes pulse-critical {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.45; }}
}}
.qp-pulse {{ animation: pulse-critical 1.8s ease-in-out infinite; }}
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
    """Full-control KPI card returned as HTML."""
    top_border = accent or COLORS["border"]
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
        f'box-shadow:0 2px 8px rgba(0,0,0,0.3);">'
        f'<div style="color:{COLORS["neutral"]};font-size:0.68rem;text-transform:uppercase;'
        f'letter-spacing:0.09em;font-weight:600;margin-bottom:7px;">{label}</div>'
        f'<div style="color:{COLORS["text"]};font-size:1.5rem;font-weight:700;'
        f'letter-spacing:-0.025em;line-height:1.1;">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def section_label(text: str) -> str:
    return (
        f'<div style="color:{COLORS["neutral"]};font-size:0.68rem;text-transform:uppercase;'
        f'letter-spacing:0.1em;font-weight:700;margin:18px 0 10px;'
        f'border-left:3px solid {COLORS["positive"]};padding-left:10px;">{text}</div>'
    )


def badge(text: str, variant: str = "neutral") -> str:
    palette = {
        "positive": ("#00d4aa", "#003d2e"),
        "negative": ("#ff4b4b", "#3d0000"),
        "warning":  ("#f6ad55", "#3d2800"),
        "neutral":  ("#8892a4", "#1a1f2e"),
        "blue":     ("#4299e1", "#0d2040"),
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
        f'<div style="background:{color}16;border:1px solid {color};'
        f'border-radius:10px;padding:14px 22px;margin-bottom:22px;'
        f'display:flex;align-items:center;gap:14px;">'
        f'<span{cls} style="font-size:1rem;font-weight:700;color:{color};">● {text}</span>'
        f'</div>'
    )
