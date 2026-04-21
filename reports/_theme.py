"""Shared Plotly theme and CSS for QuantPipe dashboards."""

COLORS = {
    "bg": "#0e1117",
    "card_bg": "#1a1f2e",
    "border": "#2d3748",
    "positive": "#00d4aa",
    "negative": "#ff4b4b",
    "warning": "#f6ad55",
    "neutral": "#8892a4",
    "text": "#e2e8f0",
    "blue": "#4299e1",
    "purple": "#9f7aea",
}

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card_bg"],
    font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif", size=12),
    xaxis=dict(
        showgrid=True,
        gridcolor=COLORS["border"],
        linecolor=COLORS["border"],
        tickcolor=COLORS["neutral"],
        zerolinecolor=COLORS["border"],
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor=COLORS["border"],
        linecolor=COLORS["border"],
        tickcolor=COLORS["neutral"],
        zerolinecolor=COLORS["border"],
    ),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=COLORS["border"],
        font=dict(color=COLORS["neutral"]),
    ),
)


def apply_theme(fig, title: str = "") -> None:
    """Apply shared dark theme to a Plotly figure in-place."""
    layout = dict(_PLOTLY_LAYOUT)
    layout["title"] = dict(
        text=title,
        font=dict(size=13, color=COLORS["neutral"]),
        x=0,
        pad=dict(l=4),
    )
    fig.update_layout(**layout)


CSS = """
<style>
/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
}
[data-testid="stMetricLabel"] {
    color: #8892a4;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1f2e;
    border-right: 1px solid #2d3748;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2d3748;
    border-radius: 8px;
    overflow: hidden;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #8892a4;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #00d4aa;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background: #00d4aa !important;
}

/* Divider */
hr { border-color: #2d3748 !important; }
</style>
"""


def badge(text: str, variant: str = "neutral") -> str:
    """Return an HTML status badge pill."""
    palette = {
        "positive": ("#00d4aa", "#003d2e"),
        "negative": ("#ff4b4b", "#3d0000"),
        "warning":  ("#f6ad55", "#3d2800"),
        "neutral":  ("#8892a4", "#1a1f2e"),
        "blue":     ("#4299e1", "#0d2040"),
    }
    fg, bg = palette.get(variant, palette["neutral"])
    return (
        f'<span style="background:{bg}; color:{fg}; border:1px solid {fg}; '
        f'border-radius:4px; padding:3px 12px; font-size:0.8rem; '
        f'font-weight:700; font-family:monospace; letter-spacing:0.05em;">'
        f'{text}</span>'
    )
