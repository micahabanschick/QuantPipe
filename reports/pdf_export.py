"""PDF report builder for the QuantPipe Performance Dashboard.

Sections:
  1. Executive Summary  — KPI grid, trailing returns table, risk metrics
  2. Performance Charts — equity + drawdown, trailing-returns bar, stress bar
  3. Portfolio          — holdings table, sector allocation
  4. Risk Analysis      — drawdowns table, stress scenarios table
  5. Analytics Charts   — monthly heatmap, return distribution, rolling Sharpe

All chart figures are built internally from serialised JSON data so this
module has no Streamlit / session-state dependencies and is fully cacheable.
"""

import io
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Brand palette (print-safe, white background) ──────────────────────────────

_TEAL   = colors.HexColor("#007a63")
_NAVY   = colors.HexColor("#1a1f2e")
_BLUE   = colors.HexColor("#2563eb")
_RED    = colors.HexColor("#b91c1c")
_ORANGE = colors.HexColor("#d97706")
_PURPLE = colors.HexColor("#7c3aed")
_GRAY   = colors.HexColor("#64748b")
_LGRAY  = colors.HexColor("#f8fafc")
_MGRAY  = colors.HexColor("#e2e8f0")
_WHITE  = colors.white
_BLACK  = colors.HexColor("#1e293b")
_GREEN  = colors.HexColor("#15803d")

_PAGE_W, _PAGE_H = letter
_MARGIN    = 0.65 * inch
_CONTENT_W = _PAGE_W - 2 * _MARGIN   # 7.2 inches
_HALF_W    = (_CONTENT_W - 0.15 * inch) / 2  # ~3.525 inches per panel

# ── Plotly chart theme for white/print background ─────────────────────────────

_CL = dict(            # chart layout base
    paper_bgcolor="white",
    plot_bgcolor="#f9fafb",
    font=dict(color="#374151", family="Helvetica", size=10),
    xaxis=dict(
        showgrid=True, gridcolor="#e5e7eb", linecolor="#d1d5db",
        tickfont=dict(color="#6b7280", size=8),
        zerolinecolor="#d1d5db",
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#e5e7eb", linecolor="#d1d5db",
        tickfont=dict(color="#6b7280", size=8),
        zerolinecolor="#d1d5db",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=9, color="#4b5563"),
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
    ),
    margin=dict(l=55, r=12, t=32, b=40),
    hovermode=False,
)


# ── Style helpers ──────────────────────────────────────────────────────────────

def _style(name="", **kw) -> ParagraphStyle:
    defaults = dict(fontName="Helvetica", fontSize=9, textColor=_BLACK, leading=13)
    defaults.update(kw)
    return ParagraphStyle(name or "s", **defaults)


_S_H2      = _style("h2",  fontSize=11, fontName="Helvetica-Bold", textColor=_NAVY,
                    spaceBefore=10, spaceAfter=4)
_S_BODY    = _style("body")
_S_CAPTION = _style("cap", fontSize=7.5, fontName="Helvetica-Oblique", textColor=_GRAY)
_S_SUB     = _style("sub", fontSize=10, textColor=_GRAY, leading=14)


# ── Reusable flowable builders ─────────────────────────────────────────────────

def _sp(h: float = 0.1) -> Spacer:
    return Spacer(1, h * inch)

def _hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5, color=_MGRAY, spaceAfter=6, spaceBefore=6)

def _section_header(title: str):
    bar = Table([[""]], colWidths=[0.06 * inch], rowHeights=[0.32 * inch])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _TEAL)]))
    lbl = Table([[Paragraph(title, _S_H2)]], colWidths=[_CONTENT_W - 0.14 * inch],
                rowHeights=[0.32 * inch])
    lbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _LGRAY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    outer = Table([[bar, lbl]], colWidths=[0.08 * inch, _CONTENT_W - 0.08 * inch])
    outer.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return outer


def _data_table(data: list[list], col_widths: list[float]) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",    (0, 0), (-1,  0), _NAVY),
        ("TEXTCOLOR",     (0, 0), (-1,  0), _WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("LINEBELOW",     (0, 0), (-1,  0), 0.5, _TEAL),
        ("LINEBELOW",     (0, 1), (-1, -1), 0.3, _MGRAY),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _LGRAY]),
    ]))
    return t


def _kpi_row(kpis: list[tuple[str, str, str]]) -> Table:
    """Row of KPI cards. kpis = [(label, value, hex_color), ...]"""
    n     = len(kpis)
    cell_w = _CONTENT_W / n

    def _cell(label, value, color):
        c = colors.HexColor(color if color.startswith("#") else f"#{color}")
        inner = Table(
            [[Paragraph(label, _style(fontSize=7, textColor=_GRAY))],
             [Paragraph(value, _style(fontSize=13, textColor=_BLACK,
                                      fontName="Helvetica-Bold", leading=16))]],
            colWidths=[cell_w - 0.16 * inch],
        )
        inner.setStyle(TableStyle([
            ("LEFTPADDING",  (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
            ("TOPPADDING",   (0,0),(-1,-1), 1),
            ("BOTTOMPADDING",(0,0),(-1,-1), 1),
        ]))
        outer = Table([[inner]], colWidths=[cell_w])
        outer.setStyle(TableStyle([
            ("BOX",         (0,0),(-1,-1), 0.8, _MGRAY),
            ("LINEABOVE",   (0,0),(-1, 0), 2.5, c),
            ("BACKGROUND",  (0,0),(-1,-1), _WHITE),
            ("LEFTPADDING", (0,0),(-1,-1), 9),
            ("TOPPADDING",  (0,0),(-1,-1), 7),
            ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ]))
        return outer

    row   = [_cell(l, v, c) for l, v, c in kpis]
    t     = Table([row], colWidths=[cell_w] * n)
    t.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0),(-1,-1), 3),
        ("RIGHTPADDING", (0,0),(-1,-1), 3),
        ("TOPPADDING",   (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    return t


def _render_chart(fig, width_in: float, height_in: float) -> Image | None:
    """Render a Plotly figure to a PNG Image flowable via kaleido."""
    try:
        png = fig.to_image(
            format="png",
            width=int(width_in * 100),
            height=int(height_in * 100),
            scale=2,
        )
        return Image(io.BytesIO(png), width=width_in * inch, height=height_in * inch)
    except Exception:
        return None


def _side_by_side(img_l: Image | None, img_r: Image | None,
                  cap_l: str = "", cap_r: str = "") -> Table | None:
    """Place two chart images side-by-side with optional captions."""
    if img_l is None and img_r is None:
        return None
    cap_style = _S_CAPTION

    def _col(img, cap):
        if img is None:
            return Paragraph("Chart unavailable", cap_style)
        items: list = [img]
        if cap:
            items.append(Paragraph(cap, cap_style))
        inner = Table([[x] for x in items], colWidths=[_HALF_W])
        inner.setStyle(TableStyle([
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 2),
        ]))
        return inner

    t = Table([[_col(img_l, cap_l), _col(img_r, cap_r)]],
              colWidths=[_HALF_W, _CONTENT_W - _HALF_W])
    t.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]))
    return t


# ── Chart builders (all return go.Figure) ────────────────────────────────────

def _fig_equity_drawdown(eq_dates, eq_vals, bench_dates=None, bench_vals=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
    )

    if bench_dates and bench_vals and eq_vals:
        scale = eq_vals[0] / bench_vals[0] if bench_vals[0] else 1
        fig.add_trace(go.Scatter(
            x=bench_dates, y=[v * scale for v in bench_vals],
            name="Benchmark", mode="lines",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=eq_dates, y=eq_vals,
        name="Portfolio", mode="lines",
        line=dict(color="#007a63", width=2),
        fill="tozeroy", fillcolor="rgba(0,122,99,0.07)",
    ), row=1, col=1)

    # Drawdown from equity
    eq_arr = np.array(eq_vals, dtype=float)
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak
    fig.add_trace(go.Scatter(
        x=eq_dates, y=dd,
        name="Drawdown", mode="lines",
        line=dict(color="#b91c1c", width=1.5),
        fill="tozeroy", fillcolor="rgba(185,28,28,0.1)",
        showlegend=True,
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#d1d5db", width=0.8), row=2, col=1)

    layout = dict(_CL)
    layout.update(height=320, showlegend=True)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", linecolor="#d1d5db",
                     tickfont=dict(color="#6b7280", size=8))
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", linecolor="#d1d5db",
                     tickfont=dict(color="#6b7280", size=8))
    fig.update_yaxes(tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(tickformat=".1%",   row=2, col=1)
    return fig


def _fig_trailing_bar(trailing: dict):
    import plotly.graph_objects as go
    labels = [k for k, v in trailing.items() if v is not None]
    vals   = [v for v in trailing.values() if v is not None]
    clrs   = ["#007a63" if v >= 0 else "#b91c1c" for v in vals]
    fig    = go.Figure(go.Bar(
        x=labels, y=vals,
        marker=dict(color=clrs, line=dict(width=0)),
        text=[f"{v:+.1%}" for v in vals],
        textposition="outside",
        textfont=dict(size=9, color="#374151"),
    ))
    fig.add_hline(y=0, line=dict(color="#d1d5db", width=0.8))
    layout = dict(_CL)
    layout.update(showlegend=False, title_text="Trailing Returns")
    fig.update_layout(**layout)
    fig.update_layout(yaxis=dict(tickformat=".0%", showgrid=False),
                      xaxis=dict(showgrid=False))
    return fig


def _fig_stress_bar(stress: dict):
    import plotly.graph_objects as go
    names  = list(stress.keys())
    vals   = list(stress.values())
    clrs   = ["#007a63" if v >= 0 else "#b91c1c" for v in vals]
    fig    = go.Figure(go.Bar(
        x=names, y=vals,
        marker=dict(color=clrs, line=dict(width=0)),
        text=[f"{v:+.1%}" for v in vals],
        textposition="outside",
        textfont=dict(size=8.5, color="#374151"),
    ))
    fig.add_hline(y=0, line=dict(color="#d1d5db", width=0.8))
    layout = dict(_CL)
    layout.update(showlegend=False, title_text="Stress Scenarios (Est. P&L)")
    fig.update_layout(**layout)
    fig.update_layout(
        yaxis=dict(tickformat=".0%", showgrid=False),
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=7.5)),
    )
    return fig


def _fig_monthly_heatmap(monthly_pivot: pd.DataFrame):
    import plotly.graph_objects as go
    mo   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = [m for m in mo if m in monthly_pivot.columns]
    pv   = monthly_pivot.reindex(columns=cols)
    z    = pv.values
    text = [[f"{v:+.1%}" if not np.isnan(v) else "" for v in row] for row in z]
    fig  = go.Figure(go.Heatmap(
        z=z, x=cols,
        y=[str(y) for y in pv.index.tolist()],
        text=text, texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale="RdYlGn", zmid=0,
        showscale=True,
        colorbar=dict(tickformat=".0%", len=0.8, thickness=10,
                      tickfont=dict(size=8)),
        hoverinfo="skip",
    ))
    layout = dict(_CL)
    layout.update(
        height=max(160, 36 * len(pv) + 60),
        title_text="Monthly Returns",
        margin=dict(l=45, r=55, t=35, b=30),
    )
    fig.update_layout(**layout)
    fig.update_yaxes(autorange="reversed")
    return fig


def _fig_return_dist(daily_rets: np.ndarray, var95: float, var99: float):
    import plotly.graph_objects as go
    from scipy.stats import norm
    r      = daily_rets
    x_grid = np.linspace(r.min(), r.max(), 300)
    pdf_n  = norm.pdf(x_grid, r.mean(), r.std())
    fig    = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, histnorm="probability density", nbinsx=70,
        marker=dict(color="#2563eb", opacity=0.55, line=dict(width=0)),
        name="Daily returns",
    ))
    fig.add_trace(go.Scatter(
        x=x_grid, y=pdf_n,
        line=dict(color="#d97706", width=2),
        name="Normal fit",
    ))
    for v, lbl, clr in [(var95, "VaR 95%", "#b91c1c"), (var99, "VaR 99%", "#7c3aed")]:
        fig.add_vline(x=v, line=dict(color=clr, dash="dot", width=1.4),
                      annotation_text=lbl,
                      annotation_font=dict(color=clr, size=8),
                      annotation_position="top")
    layout = dict(_CL)
    layout.update(showlegend=True, title_text="Return Distribution")
    fig.update_layout(**layout)
    fig.update_layout(xaxis=dict(tickformat=".1%"),
                      yaxis=dict(title="Density", showgrid=False))
    return fig


def _fig_rolling_sharpe(daily_rets_dates, daily_rets_vals, window: int = 63):
    import plotly.graph_objects as go
    s = pd.Series(daily_rets_vals, index=pd.to_datetime(daily_rets_dates))

    def _sharpe(x):
        sd = x.std()
        return x.mean() / sd * np.sqrt(252) if sd > 1e-10 else np.nan

    def _sortino(x):
        d = x[x < 0]
        return x.mean() / d.std() * np.sqrt(252) if len(d) > 0 and d.std() > 1e-10 else np.nan

    rs = s.rolling(window).apply(_sharpe,  raw=False)
    so = s.rolling(window).apply(_sortino, raw=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.values, name=f"{window}d Sharpe",
        line=dict(color="#007a63", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=so.index, y=so.values, name=f"{window}d Sortino",
        line=dict(color="#7c3aed", width=1.8, dash="dot"),
    ))
    fig.add_hline(y=0, line=dict(color="#d1d5db", width=0.8))
    fig.add_hline(y=1, line=dict(color="#007a63", width=0.7, dash="dot"))
    layout = dict(_CL)
    layout.update(showlegend=True, title_text=f"Rolling {window}d Sharpe / Sortino")
    fig.update_layout(**layout)
    fig.update_layout(yaxis=dict(showgrid=False))
    return fig


# ── Cover ──────────────────────────────────────────────────────────────────────

def _cover(story, as_of: date, lookback_years: int, benchmark_sym: str) -> None:
    bar = Table([[""]], colWidths=[_CONTENT_W], rowHeights=[0.08 * inch])
    bar.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), _TEAL)]))
    story.append(bar)
    story.append(_sp(0.35))
    story.append(Paragraph("QuantPipe",
                 _style(fontSize=28, fontName="Helvetica-Bold", textColor=_TEAL)))
    story.append(Paragraph("Performance Report",
                 _style(fontSize=18, fontName="Helvetica-Bold", textColor=_NAVY, spaceBefore=2)))
    story.append(_sp(0.1))
    story.append(Paragraph(
        "Cross-sectional 12-1 Momentum  ·  Top-5 ETFs  ·  Equal Weight  ·  Monthly Rebalance",
        _S_SUB))
    story.append(_sp(0.08))
    meta = [
        ["As of",     as_of.strftime("%B %d, %Y")],
        ["Lookback",  f"{lookback_years} years"],
        ["Benchmark", benchmark_sym if benchmark_sym != "None" else "None"],
        ["Generated", date.today().strftime("%B %d, %Y")],
    ]
    mt = Table(meta, colWidths=[1.1 * inch, 2.5 * inch])
    mt.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("TEXTCOLOR",    (0,0),(0,-1), _GRAY),
        ("TEXTCOLOR",    (1,0),(1,-1), _BLACK),
        ("TOPPADDING",   (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(mt)
    story.append(_sp(0.18))
    story.append(_hr())


# ── Section 1: Executive Summary ──────────────────────────────────────────────

def _section_summary(story, stats: dict, trailing: dict) -> None:
    story.append(_section_header("1 · Executive Summary"))
    story.append(_sp(0.12))

    def _pct(v): return f"{v:+.1%}" if isinstance(v, (int, float)) else "—"
    def _f(v):   return f"{v:.2f}"  if isinstance(v, (int, float)) else "—"
    def _pp(v):  return f"{v:.1%}"  if isinstance(v, (int, float)) else "—"

    # Row 1
    story.append(_kpi_row([
        ("Total Return",  _pct(stats.get("total")),  "#007a63"),
        ("CAGR",          _pct(stats.get("cagr")),   "#007a63"),
        ("Sharpe Ratio",  _f(stats.get("sharpe")),   "#2563eb"),
        ("Sortino Ratio", _f(stats.get("sortino")),  "#7c3aed"),
    ]))
    story.append(_sp(0.07))
    # Row 2
    story.append(_kpi_row([
        ("Max Drawdown",    _pct(stats.get("max_dd")),   "#b91c1c"),
        ("Calmar Ratio",    _f(stats.get("calmar")),     "#d97706"),
        ("Ann. Volatility", _pp(stats.get("vol")),       "#d97706"),
        ("Win Rate",        _pp(stats.get("win_rate")),  "#64748b"),
    ]))
    story.append(_sp(0.14))

    # Trailing returns + Risk metrics side-by-side
    tr_rows = [["Period", "Return"]]
    for period, val in trailing.items():
        v_str = f"{val:+.2%}" if val is not None else "—"
        tr_rows.append([period, v_str])

    tr_t = Table(tr_rows, colWidths=[1.3 * inch, 1.4 * inch])
    tr_t.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 8.5),
        ("BACKGROUND",   (0,0),(-1, 0), _NAVY),
        ("TEXTCOLOR",    (0,0),(-1, 0), _WHITE),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("LINEBELOW",    (0,0),(-1, 0), 0.5, _TEAL),
        ("LINEBELOW",    (0,1),(-1,-1), 0.3, _MGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [_WHITE, _LGRAY]),
        *[
            ("TEXTCOLOR", (1, i+1), (1, i+1),
             _GREEN if (trailing.get(k) or 0) >= 0 else _RED)
            for i, k in enumerate(trailing.keys())
            if trailing.get(k) is not None
        ],
    ]))

    risk_rows = [
        ["Metric", "Value"],
        ["1-Day VaR 95%",  f"{stats.get('var95', 0):.3%}"],
        ["1-Day VaR 99%",  f"{stats.get('var99', 0):.3%}"],
        ["CVaR 95% (ES)",  f"{stats.get('cvar95', 0):.3%}"],
        ["Best Day",       f"{stats.get('best_day', 0):+.2%}"],
        ["Worst Day",      f"{stats.get('worst_day', 0):+.2%}"],
    ]
    rk_t = _data_table(risk_rows, [2.0 * inch, 1.3 * inch])

    side = Table([[tr_t, rk_t]],
                 colWidths=[2.9 * inch, _CONTENT_W - 2.9 * inch])
    side.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    story.append(side)


# ── Section 2: Performance Charts ─────────────────────────────────────────────

def _section_perf_charts(story, eq_dates, eq_vals, bench_dates, bench_vals,
                          trailing: dict, stress: dict) -> None:
    story.append(PageBreak())
    story.append(_section_header("2 · Performance Charts"))
    story.append(_sp(0.12))

    # Equity + Drawdown (full width)
    fig_ed = _fig_equity_drawdown(eq_dates, eq_vals, bench_dates, bench_vals)
    img_ed = _render_chart(fig_ed, width_in=_CONTENT_W / inch, height_in=3.2)
    if img_ed:
        story.append(img_ed)
        story.append(Paragraph(
            "Top panel: portfolio equity curve (normalised to $10,000). "
            "Bottom panel: underwater drawdown from peak.",
            _S_CAPTION,
        ))
    story.append(_sp(0.18))

    # Trailing returns | Stress scenarios (side by side)
    img_tr = img_st = None
    if trailing:
        img_tr = _render_chart(_fig_trailing_bar(trailing), _HALF_W / inch, 2.0)
    if stress:
        img_st = _render_chart(_fig_stress_bar(stress),    _HALF_W / inch, 2.0)

    pair = _side_by_side(img_tr, img_st)
    if pair:
        story.append(pair)


# ── Section 3: Portfolio ───────────────────────────────────────────────────────

def _section_portfolio(story, current_weights: dict, exposures, sector_map: dict) -> None:
    story.append(PageBreak())
    story.append(_section_header("3 · Portfolio Composition"))
    story.append(_sp(0.12))

    if not current_weights:
        story.append(Paragraph("No current portfolio weights available.", _S_BODY))
        return

    if exposures is not None:
        story.append(_kpi_row([
            ("Positions",           str(exposures.n_positions),               "#64748b"),
            ("Gross Exposure",      f"{exposures.gross_exposure:.1%}",        "#2563eb"),
            ("Top-5 Concentration", f"{exposures.top_5_concentration:.1%}",   "#d97706"),
            ("Largest Name",
             f"{exposures.largest_position[0]} {exposures.largest_position[1]:.1%}",
             "#007a63"),
        ]))
        story.append(_sp(0.14))

    # Holdings + Sector side-by-side
    h_rows = [["Symbol", "Weight", "Sector"]]
    for sym, wt in sorted(current_weights.items(), key=lambda x: -x[1]):
        h_rows.append([sym, f"{wt:.1%}", sector_map.get(sym, "Other")])
    hold_t = _data_table(h_rows, [0.9*inch, 0.8*inch, 1.5*inch])

    if exposures is not None and exposures.sector_exposures:
        s_rows = [["Sector", "Alloc"]]
        for sec, wt in sorted(exposures.sector_exposures.items(), key=lambda x: -x[1]):
            s_rows.append([sec, f"{wt:.1%}"])
        sec_t = _data_table(s_rows, [2.4*inch, 0.8*inch])
        side = Table([[hold_t, sec_t]],
                     colWidths=[3.3 * inch, _CONTENT_W - 3.3 * inch])
        side.setStyle(TableStyle([
            ("LEFTPADDING",  (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
            ("TOPPADDING",   (0,0),(-1,-1), 0),
            ("BOTTOMPADDING",(0,0),(-1,-1), 0),
            ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ]))
        story.append(Paragraph("Holdings &amp; Sector Allocation", _S_H2))
        story.append(side)
    else:
        story.append(Paragraph("Holdings", _S_H2))
        story.append(hold_t)


# ── Section 4: Risk Analysis ───────────────────────────────────────────────────

def _section_risk(story, drawdowns: list[dict], stress: dict) -> None:
    story.append(_sp(0.2))
    story.append(_section_header("4 · Risk Analysis"))
    story.append(_sp(0.12))

    if stress:
        story.append(Paragraph("Stress Scenarios", _S_H2))
        sc_rows = [["Scenario", "Estimated P&L"]]
        for name, val in sorted(stress.items(), key=lambda x: x[1]):
            sc_rows.append([name, f"{val:+.1%}"])
        t = Table(sc_rows, colWidths=[3.8 * inch, 1.5 * inch])
        t.setStyle(TableStyle([
            ("FONTNAME",     (0,0),(-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0),(-1,-1), 8.5),
            ("BACKGROUND",   (0,0),(-1, 0), _NAVY),
            ("TEXTCOLOR",    (0,0),(-1, 0), _WHITE),
            ("TOPPADDING",   (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING",  (0,0),(-1,-1), 8),
            ("LINEBELOW",    (0,0),(-1, 0), 0.5, _TEAL),
            ("LINEBELOW",    (0,1),(-1,-1), 0.3, _MGRAY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[_WHITE, _LGRAY]),
            *[("TEXTCOLOR", (1,i+1),(1,i+1),
               _GREEN if list(stress.values())[i] >= 0 else _RED)
              for i in range(len(stress))],
        ]))
        story.append(t)
        story.append(_sp(0.14))

    if drawdowns:
        story.append(Paragraph("Top Drawdown Periods", _S_H2))
        dd_keys = ["Start", "Trough", "Recovery", "Drawdown", "Depth (d)", "Recov (d)"]
        dd_rows = [dd_keys] + [[d.get(k, "—") for k in dd_keys] for d in drawdowns]
        col_ws  = [1.1*inch, 1.1*inch, 1.1*inch, 0.9*inch, 0.75*inch, 0.75*inch]
        t = _data_table(dd_rows, col_ws)
        t.setStyle(TableStyle([
            ("TEXTCOLOR", (3,1),(3,-1), _RED),
            ("FONTNAME",  (3,1),(3,-1), "Helvetica-Bold"),
        ]))
        story.append(t)


# ── Section 5: Analytics Charts ───────────────────────────────────────────────

def _section_analytics(story, monthly_pivot: pd.DataFrame | None,
                        daily_rets_dates, daily_rets_vals,
                        var95: float, var99: float) -> None:
    story.append(PageBreak())
    story.append(_section_header("5 · Analytics"))
    story.append(_sp(0.12))

    # Monthly returns heatmap (full width)
    if monthly_pivot is not None and not monthly_pivot.empty:
        n_years    = len(monthly_pivot)
        h_heatmap  = max(1.6, 0.28 * n_years + 0.8)
        img_hm = _render_chart(
            _fig_monthly_heatmap(monthly_pivot),
            width_in=_CONTENT_W / inch,
            height_in=min(h_heatmap, 3.8),
        )
        if img_hm:
            story.append(img_hm)
            story.append(Paragraph(
                "Green = positive month, red = negative. "
                "Each cell shows arithmetic return for that calendar month.",
                _S_CAPTION,
            ))
            story.append(_sp(0.18))

    # Monthly returns table (Year × Month grid)
    if monthly_pivot is not None and not monthly_pivot.empty:
        mo   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        cols = [m for m in mo if m in monthly_pivot.columns]
        pv   = monthly_pivot.reindex(columns=cols)
        header_row = ["Year"] + cols
        rows = [header_row]
        row_styles = []
        for yr_idx, year in enumerate(pv.index):
            row = [str(year)]
            for c in cols:
                val = pv.loc[year, c]
                if pd.isna(val):
                    row.append("")
                else:
                    row.append(f"{val:+.1%}")
                    row_styles.append(
                        ("TEXTCOLOR", (cols.index(c)+1, yr_idx+1),
                         (cols.index(c)+1, yr_idx+1),
                         _GREEN if val >= 0 else _RED)
                    )
            rows.append(row)

        col_w = _CONTENT_W / len(header_row)
        t = Table(rows, colWidths=[col_w] * len(header_row), repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("BACKGROUND",    (0,0),(-1, 0), _NAVY),
            ("TEXTCOLOR",     (0,0),(-1, 0), _WHITE),
            ("BACKGROUND",    (0,1),(0,-1),  _LGRAY),
            ("FONTNAME",      (0,1),(0,-1),  "Helvetica-Bold"),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
            ("ALIGN",         (1,0),(-1,-1), "RIGHT"),
            ("LINEBELOW",     (0,0),(-1, 0), 0.5, _TEAL),
            ("LINEBELOW",     (0,1),(-1,-1), 0.3, _MGRAY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_WHITE, _LGRAY]),
            *row_styles,
        ]))
        story.append(t)
        story.append(_sp(0.18))

    # Return distribution | Rolling Sharpe (side by side)
    if daily_rets_vals:
        dr_arr = np.array(daily_rets_vals, dtype=float)
        img_dist = _render_chart(
            _fig_return_dist(dr_arr, var95, var99),
            _HALF_W / inch, 2.5,
        )
        img_rs = _render_chart(
            _fig_rolling_sharpe(daily_rets_dates, daily_rets_vals, window=63),
            _HALF_W / inch, 2.5,
        )
        pair = _side_by_side(
            img_dist, img_rs,
            cap_l="Daily return histogram with normal-distribution fit and VaR reference lines.",
            cap_r="Rolling 63-day Sharpe (teal) and Sortino (purple) ratios. "
                  "Dashed line marks the 1.0 target.",
        )
        if pair:
            story.append(pair)


# ── Footer ─────────────────────────────────────────────────────────────────────

def _footer_canvas(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_GRAY)
    canvas.drawCentredString(
        _PAGE_W / 2, 0.4 * inch,
        f"QuantPipe — For research and paper trading only. Not investment advice.  ·  Page {doc.page}",
    )
    canvas.restoreState()


# ── Public entry point ─────────────────────────────────────────────────────────

def build_performance_pdf(
    *,
    stats: dict,
    trailing: dict,
    drawdowns: list[dict],
    current_weights: dict,
    exposures,
    sector_map: dict,
    stress: dict,
    monthly_pivot: pd.DataFrame | None,
    lookback_years: int,
    benchmark_sym: str,
    as_of: date,
    # Serialised time-series data for chart generation
    eq_dates: list[str],
    eq_vals: list[float],
    bench_dates: list[str] | None = None,
    bench_vals: list[float] | None = None,
    daily_rets_dates: list[str] | None = None,
    daily_rets_vals: list[float] | None = None,
) -> bytes:
    """Build and return a complete PDF performance report as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_MARGIN,  bottomMargin=0.8 * inch,
        title="QuantPipe Performance Report",
        author="QuantPipe",
    )

    story: list = []

    _cover(story, as_of=as_of, lookback_years=lookback_years, benchmark_sym=benchmark_sym)

    if stats:
        _section_summary(story, stats, trailing)

    if eq_vals:
        _section_perf_charts(
            story,
            eq_dates=eq_dates, eq_vals=eq_vals,
            bench_dates=bench_dates or [],
            bench_vals=bench_vals or [],
            trailing=trailing or {},
            stress=stress or {},
        )

    _section_portfolio(story, current_weights, exposures, sector_map)
    _section_risk(story, drawdowns or [], stress or {})

    _section_analytics(
        story,
        monthly_pivot=monthly_pivot,
        daily_rets_dates=daily_rets_dates or [],
        daily_rets_vals=daily_rets_vals or [],
        var95=float(stats.get("var95") or 0),
        var99=float(stats.get("var99") or 0),
    )

    # Closing disclaimer
    story.append(PageBreak())
    story.append(_sp(0.3))
    story.append(Paragraph(
        f"QuantPipe Performance Report  ·  Generated {date.today().strftime('%B %d, %Y')}",
        _style(fontSize=9, textColor=_GRAY, alignment=TA_CENTER),
    ))
    story.append(_sp(0.1))
    story.append(Paragraph(
        "This report is for research and paper trading purposes only. "
        "It does not constitute investment advice. Past performance does not guarantee future results.",
        _style(fontSize=8, textColor=_GRAY, alignment=TA_CENTER, leading=12),
    ))

    doc.build(story, onFirstPage=_footer_canvas, onLaterPages=_footer_canvas)
    return buf.getvalue()
