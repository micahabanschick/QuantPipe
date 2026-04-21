"""PDF report builder for the QuantPipe Performance Dashboard.

Generates a multi-section PDF from pre-computed dashboard data:
  1. Executive Summary — KPI grid, trailing returns
  2. Portfolio         — holdings, sector, exposure metrics
  3. Risk              — stress scenarios, VaR/CVaR, top drawdowns
  4. Analytics         — monthly returns grid
  5. Trade History     — portfolio log (if available)

Plotly figures are embedded as PNG images via kaleido where provided.
All sections degrade gracefully when data is None.
"""

import io
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand palette (readable on white) ─────────────────────────────────────────

_TEAL   = colors.HexColor("#007a63")   # darker teal for print
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
_MARGIN = 0.65 * inch
_CONTENT_W = _PAGE_W - 2 * _MARGIN


# ── Style helpers ──────────────────────────────────────────────────────────────

def _style(name, **kw) -> ParagraphStyle:
    defaults = dict(fontName="Helvetica", fontSize=9, textColor=_BLACK, leading=13)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


_S_TITLE   = _style("title",   fontSize=22, fontName="Helvetica-Bold",  textColor=_NAVY,  leading=26)
_S_SUB     = _style("sub",     fontSize=10, fontName="Helvetica",       textColor=_GRAY,  leading=14)
_S_H2      = _style("h2",      fontSize=11, fontName="Helvetica-Bold",  textColor=_NAVY,  spaceBefore=10, spaceAfter=4)
_S_BODY    = _style("body",    fontSize=9,  fontName="Helvetica",       textColor=_BLACK)
_S_CAPTION = _style("caption", fontSize=7.5, fontName="Helvetica-Oblique", textColor=_GRAY)
_S_FOOTER  = _style("footer",  fontSize=7.5, fontName="Helvetica",      textColor=_GRAY,  alignment=TA_CENTER)


# ── Reusable flowable builders ─────────────────────────────────────────────────

def _section_header(title: str):
    """A teal-bar + label section header."""
    bar  = Table([[""]],    colWidths=[0.06 * inch], rowHeights=[0.32 * inch])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _TEAL)]))
    lbl  = Table([[Paragraph(title, _S_H2)]], colWidths=[_CONTENT_W - 0.14 * inch],
                 rowHeights=[0.32 * inch])
    lbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _LGRAY),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
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


def _table(data: list[list], col_widths: list[float],
           row_colors: bool = True, header: bool = True) -> Table:
    """Styled data table with optional alternating row background."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME",     (0, 0), (-1,  0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1),  8.5),
        ("BACKGROUND",   (0, 0), (-1,  0),  _NAVY),
        ("TEXTCOLOR",    (0, 0), (-1,  0),  _WHITE),
        ("TOPPADDING",   (0, 0), (-1, -1),  4),
        ("BOTTOMPADDING",(0, 0), (-1, -1),  4),
        ("LEFTPADDING",  (0, 0), (-1, -1),  7),
        ("RIGHTPADDING", (0, 0), (-1, -1),  7),
        ("LINEBELOW",    (0, 0), (-1,  0),  0.5, _TEAL),
        ("LINEBELOW",    (0, 1), (-1, -1),  0.3, _MGRAY),
        ("VALIGN",       (0, 0), (-1, -1),  "MIDDLE"),
        ("ALIGN",        (0, 0), (-1,  0),  "LEFT"),
    ]
    if row_colors and header:
        for i in range(1, len(data)):
            bg = _LGRAY if i % 2 == 0 else _WHITE
            style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(style))
    return t


def _kpi_table(kpis: list[tuple[str, str, Any]]) -> Table:
    """2-column KPI cards in a grid. kpis = [(label, value, color), ...]"""
    # Pair them up into rows of 4 (2 per cell, 2 cells per row)
    cell_w = _CONTENT_W / 4

    def _cell(label, value, color):
        c = colors.HexColor(color if color.startswith("#") else f"#{color}") if isinstance(color, str) else (color or _TEAL)
        inner = Table(
            [[Paragraph(label, _style("kl", fontSize=7, textColor=_GRAY, fontName="Helvetica"))],
             [Paragraph(value, _style("kv", fontSize=14, textColor=_BLACK, fontName="Helvetica-Bold", leading=17))]],
            colWidths=[cell_w - 0.2 * inch],
        )
        inner.setStyle(TableStyle([
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 1),
        ]))
        outer = Table([[inner]], colWidths=[cell_w])
        outer.setStyle(TableStyle([
            ("BOX",          (0, 0), (-1, -1), 1.0, _MGRAY),
            ("LINEABOVE",    (0, 0), (-1,  0), 2.5, c),
            ("BACKGROUND",   (0, 0), (-1, -1), _WHITE),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", (0, 0), (-1, -1), 4),
        ]))
        return outer

    rows = []
    for i in range(0, len(kpis), 4):
        chunk = kpis[i:i + 4]
        while len(chunk) < 4:
            chunk.append(("", "", _GRAY))
        rows.append([_cell(l, v, c) for l, v, c in chunk])

    t = Table(rows, colWidths=[cell_w] * 4)
    t.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    return t


def _hr():
    return HRFlowable(width="100%", thickness=0.5, color=_MGRAY, spaceAfter=6, spaceBefore=6)


def _sp(h: float = 0.1):
    return Spacer(1, h * inch)


def _chart_image(fig, width_in: float = 6.6, height_in: float = 2.8) -> Image | None:
    """Render a Plotly figure to a PNG image flowable. Returns None if kaleido fails."""
    try:
        png = fig.to_image(
            format="png",
            width=int(width_in * 96),
            height=int(height_in * 96),
            scale=2,
        )
        return Image(io.BytesIO(png), width=width_in * inch, height=height_in * inch)
    except Exception:
        return None


# ── Cover page ─────────────────────────────────────────────────────────────────

def _cover(story: list, as_of: date, lookback_years: int, benchmark_sym: str) -> None:
    # Accent bar at top
    bar = Table([[""]],
                colWidths=[_CONTENT_W], rowHeights=[0.08 * inch])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _TEAL)]))
    story.append(bar)
    story.append(_sp(0.35))

    story.append(Paragraph("QuantPipe", _style("brand", fontSize=28,
                 fontName="Helvetica-Bold", textColor=_TEAL)))
    story.append(Paragraph("Performance Report", _style("rt", fontSize=18,
                 fontName="Helvetica-Bold", textColor=_NAVY, spaceBefore=2)))
    story.append(_sp(0.12))

    story.append(Paragraph(
        f"Cross-sectional 12-1 Momentum &nbsp;·&nbsp; Top-5 ETFs &nbsp;·&nbsp; "
        f"Equal Weight &nbsp;·&nbsp; Monthly Rebalance",
        _S_SUB,
    ))
    story.append(_sp(0.08))

    meta = [
        ["As of",       as_of.strftime("%B %d, %Y")],
        ["Lookback",    f"{lookback_years} years"],
        ["Benchmark",   benchmark_sym if benchmark_sym != "None" else "None"],
        ["Generated",   date.today().strftime("%B %d, %Y")],
    ]
    mt = Table(meta, colWidths=[1.1 * inch, 2.5 * inch])
    mt.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",     (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",    (0, 0), (0, -1), _GRAY),
        ("TEXTCOLOR",    (1, 0), (1, -1), _BLACK),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    story.append(mt)
    story.append(_sp(0.18))
    story.append(_hr())


# ── Section 1: Executive Summary ───────────────────────────────────────────────

def _section_summary(story: list, stats: dict, trailing: dict) -> None:
    story.append(_section_header("1 · Executive Summary"))
    story.append(_sp(0.12))

    def _pct(v, sign=True):
        if not isinstance(v, float): return "—"
        return f"{v:+.1%}" if sign else f"{v:.1%}"
    def _f(v): return f"{v:.2f}" if isinstance(v, float) else "—"
    def _p(v): return f"{v:.1%}" if isinstance(v, float) else "—"

    kpis = [
        ("Total Return",    _pct(stats.get("total")),    "#007a63"),
        ("CAGR",            _pct(stats.get("cagr")),     "#007a63"),
        ("Sharpe Ratio",    _f(stats.get("sharpe")),     "#2563eb"),
        ("Sortino Ratio",   _f(stats.get("sortino")),    "#7c3aed"),
        ("Max Drawdown",    _pct(stats.get("max_dd")),   "#b91c1c"),
        ("Calmar Ratio",    _f(stats.get("calmar")),     "#d97706"),
        ("Ann. Volatility", _p(stats.get("vol")),        "#d97706"),
        ("Win Rate",        _p(stats.get("win_rate")),   "#64748b"),
    ]
    story.append(_kpi_table(kpis))
    story.append(_sp(0.14))

    # Trailing returns
    story.append(Paragraph("Trailing Returns", _S_H2))
    tr_rows = [["Period", "Return"]]
    for period, val in trailing.items():
        v_str = f"{val:+.2%}" if val is not None else "—"
        tr_rows.append([period, v_str])

    tr_t = Table(tr_rows, colWidths=[1.5 * inch, 1.5 * inch])
    tr_t.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",   (0, 0), (-1,  0), _NAVY),
        ("TEXTCOLOR",    (0, 0), (-1,  0), _WHITE),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("LINEBELOW",    (0, 0), (-1,  0), 0.5, _TEAL),
        ("LINEBELOW",    (0, 1), (-1, -1), 0.3, _MGRAY),
        # Colour returns
        *[
            ("TEXTCOLOR", (1, i + 1), (1, i + 1),
             _GREEN if (trailing.get(k) or 0) >= 0 else _RED)
            for i, k in enumerate(trailing.keys())
            if trailing.get(k) is not None
        ],
    ]))
    story.append(tr_t)

    # Risk summary
    story.append(_sp(0.14))
    story.append(Paragraph("Risk Metrics", _S_H2))
    risk_rows = [
        ["Metric", "Value"],
        ["1-Day VaR 95%",   f"{stats.get('var95', 0):.3%}"],
        ["1-Day VaR 99%",   f"{stats.get('var99', 0):.3%}"],
        ["CVaR 95% (ES)",   f"{stats.get('cvar95', 0):.3%}"],
        ["Best Day",        f"{stats.get('best_day', 0):+.2%}"],
        ["Worst Day",       f"{stats.get('worst_day', 0):+.2%}"],
    ]
    story.append(_table(risk_rows, [2.0 * inch, 1.5 * inch]))


# ── Section 2: Equity Curve Chart ─────────────────────────────────────────────

def _section_equity(story: list, fig_equity) -> None:
    if fig_equity is None:
        return
    img = _chart_image(fig_equity, width_in=6.6, height_in=2.8)
    if img is None:
        return
    story.append(_sp(0.1))
    story.append(_section_header("2 · Equity Curve"))
    story.append(_sp(0.1))
    story.append(img)
    story.append(Paragraph(
        "Portfolio equity curve normalised to $10,000 starting value. "
        "Dotted lines indicate monthly rebalance dates.",
        _S_CAPTION,
    ))


# ── Section 3: Portfolio ───────────────────────────────────────────────────────

def _section_portfolio(story: list, current_weights: dict, exposures, sector_map: dict) -> None:
    story.append(PageBreak())
    story.append(_section_header("3 · Portfolio Composition"))
    story.append(_sp(0.12))

    if not current_weights:
        story.append(Paragraph("No current portfolio weights available.", _S_BODY))
        return

    # Exposure KPIs
    if exposures is not None:
        exp_kpis = [
            ("Positions",          str(exposures.n_positions),             "#64748b"),
            ("Gross Exposure",     f"{exposures.gross_exposure:.1%}",      "#2563eb"),
            ("Top-5 Concentration",f"{exposures.top_5_concentration:.1%}", "#d97706"),
            ("Largest Name",
             f"{exposures.largest_position[0]} {exposures.largest_position[1]:.1%}",
             "#007a63"),
        ]
        story.append(_kpi_table(exp_kpis))
        story.append(_sp(0.14))

    # Holdings table
    story.append(Paragraph("Holdings", _S_H2))
    h_rows = [["Symbol", "Weight", "Sector"]]
    for sym, wt in sorted(current_weights.items(), key=lambda x: -x[1]):
        h_rows.append([sym, f"{wt:.1%}", sector_map.get(sym, "Other")])
    story.append(_table(h_rows, [1.2 * inch, 1.0 * inch, 2.8 * inch]))

    # Sector allocation
    if exposures is not None and exposures.sector_exposures:
        story.append(_sp(0.14))
        story.append(Paragraph("Sector Allocation", _S_H2))
        s_rows = [["Sector", "Allocation"]]
        for sec, wt in sorted(exposures.sector_exposures.items(), key=lambda x: -x[1]):
            s_rows.append([sec, f"{wt:.1%}"])
        story.append(_table(s_rows, [3.0 * inch, 1.2 * inch]))


# ── Section 4: Risk ────────────────────────────────────────────────────────────

def _section_risk(story: list, drawdowns: list[dict], stress: dict) -> None:
    story.append(_sp(0.18))
    story.append(_section_header("4 · Risk Analysis"))
    story.append(_sp(0.12))

    # Stress scenarios
    if stress:
        story.append(Paragraph("Stress Scenarios", _S_H2))
        sc_rows = [["Scenario", "Estimated P&L"]]
        for name, val in sorted(stress.items(), key=lambda x: x[1]):
            sc_rows.append([name, f"{val:+.1%}"])
        t = Table(sc_rows, colWidths=[3.5 * inch, 1.5 * inch])
        t.setStyle(TableStyle([
            ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
            ("BACKGROUND",   (0, 0), (-1,  0), _NAVY),
            ("TEXTCOLOR",    (0, 0), (-1,  0), _WHITE),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("LINEBELOW",    (0, 0), (-1,  0), 0.5, _TEAL),
            ("LINEBELOW",    (0, 1), (-1, -1), 0.3, _MGRAY),
            *[
                ("TEXTCOLOR", (1, i + 1), (1, i + 1),
                 _GREEN if list(stress.values())[i] >= 0 else _RED)
                for i in range(len(stress))
            ],
        ]))
        story.append(t)
        story.append(_sp(0.14))

    # Top drawdowns
    if drawdowns:
        story.append(Paragraph("Top Drawdown Periods", _S_H2))
        dd_keys = ["Start", "Trough", "Recovery", "Drawdown", "Depth (d)", "Recov (d)"]
        dd_rows = [dd_keys] + [[d.get(k, "—") for k in dd_keys] for d in drawdowns]
        col_ws = [1.1*inch, 1.1*inch, 1.1*inch, 0.9*inch, 0.8*inch, 0.8*inch]
        t = _table(dd_rows, col_ws)
        t.setStyle(TableStyle([
            # Highlight drawdown column red
            ("TEXTCOLOR", (3, 1), (3, -1), _RED),
            ("FONTNAME",  (3, 1), (3, -1), "Helvetica-Bold"),
        ]))
        story.append(t)


# ── Section 5: Analytics ───────────────────────────────────────────────────────

def _section_analytics(story: list, monthly_pivot: pd.DataFrame | None,
                        fig_monthly=None) -> None:
    story.append(PageBreak())
    story.append(_section_header("5 · Analytics — Monthly Returns"))
    story.append(_sp(0.12))

    if monthly_pivot is None or monthly_pivot.empty:
        story.append(Paragraph("No monthly returns data available.", _S_BODY))
        return

    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = [m for m in month_order if m in monthly_pivot.columns]
    pv   = monthly_pivot.reindex(columns=cols)

    # Build table
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
                color = _GREEN if val >= 0 else _RED
                row_styles.append(
                    ("TEXTCOLOR", (cols.index(c) + 1, yr_idx + 1),
                     (cols.index(c) + 1, yr_idx + 1), color)
                )
        rows.append(row)

    n_cols = len(header_row)
    col_w  = _CONTENT_W / n_cols
    t = Table(rows, colWidths=[col_w] * n_cols, repeatRows=1)
    style = [
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 7.5),
        ("BACKGROUND",   (0, 0), (-1,  0), _NAVY),
        ("TEXTCOLOR",    (0, 0), (-1,  0), _WHITE),
        ("BACKGROUND",   (0, 1), (0, -1),  _LGRAY),
        ("FONTNAME",     (0, 1), (0, -1),  "Helvetica-Bold"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN",        (1, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW",    (0, 0), (-1,  0), 0.5, _TEAL),
        ("LINEBELOW",    (0, 1), (-1, -1), 0.3, _MGRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _LGRAY]),
        *row_styles,
    ]
    t.setStyle(TableStyle(style))
    story.append(t)

    # Annual summary
    story.append(_sp(0.14))
    story.append(Paragraph("Annual Summary", _S_H2))
    ann_header = ["Year", "Annual Return"]
    ann_rows   = [ann_header]
    for yr_idx, year in enumerate(pv.index):
        row_vals = [pv.loc[year, c] for c in cols if not pd.isna(pv.loc[year, c])]
        if not row_vals:
            continue
        # Compound monthly returns to annual
        annual = float(np.prod([1 + v for v in row_vals]) - 1)
        ann_rows.append([str(year), f"{annual:+.2%}"])

    at = Table(ann_rows, colWidths=[1.0 * inch, 1.4 * inch])
    at.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",   (0, 0), (-1, 0),  _NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  _WHITE),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("LINEBELOW",    (0, 0), (-1,  0), 0.5, _TEAL),
        ("LINEBELOW",    (0, 1), (-1, -1), 0.3, _MGRAY),
        *[
            ("TEXTCOLOR", (1, i + 1), (1, i + 1),
             _GREEN if float(ann_rows[i + 1][1].replace("%", "")) >= 0 else _RED)
            for i in range(len(ann_rows) - 1)
        ],
    ]))
    story.append(at)


# ── Section 6: Trade History ───────────────────────────────────────────────────

def _section_trade_history(story: list, portfolio_log_pd: pd.DataFrame | None) -> None:
    story.append(PageBreak())
    story.append(_section_header("6 · Trade History"))
    story.append(_sp(0.12))

    if portfolio_log_pd is None or portfolio_log_pd.empty:
        story.append(Paragraph(
            "No portfolio log available. Run a paper rebalance to generate trade history.",
            _S_BODY,
        ))
        return

    df = portfolio_log_pd.copy()

    # Normalise any timestamp columns to short date strings
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
        elif df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[c] = parsed.dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    # Format floats compactly (4 decimal places)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

    df = df.fillna("—").tail(500)

    rows = [list(df.columns)] + df.values.tolist()
    n_cols = len(rows[0])
    col_w  = _CONTENT_W / n_cols

    t = Table(rows, colWidths=[col_w] * n_cols, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 7),
        ("BACKGROUND",   (0, 0), (-1, 0),  _NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  _WHITE),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW",    (0, 0), (-1,  0), 0.5, _TEAL),
        ("LINEBELOW",    (0, 1), (-1, -1), 0.3, _MGRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _LGRAY]),
        ("WORDWRAP",     (0, 0), (-1, -1), True),
    ]))
    story.append(t)

    if len(portfolio_log_pd) > 500:
        story.append(_sp(0.06))
        story.append(Paragraph(
            f"Showing most recent 500 of {len(portfolio_log_pd)} transactions. "
            "Download the CSV for the complete history.",
            _S_CAPTION,
        ))


# ── Footer callback ────────────────────────────────────────────────────────────

def _footer_canvas(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_GRAY)
    canvas.drawCentredString(
        _PAGE_W / 2, 0.4 * inch,
        f"QuantPipe — For research and paper trading only. Not investment advice.  "
        f"·  Page {doc.page}",
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
    portfolio_log_pd: pd.DataFrame | None,
    lookback_years: int,
    benchmark_sym: str,
    as_of: date,
    fig_equity=None,
) -> bytes:
    """Build and return a PDF report as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=0.8 * inch,
        title="QuantPipe Performance Report",
        author="QuantPipe",
    )

    story: list = []

    _cover(story, as_of=as_of, lookback_years=lookback_years, benchmark_sym=benchmark_sym)

    if stats:
        _section_summary(story, stats, trailing)
        _section_equity(story, fig_equity)

    _section_portfolio(story, current_weights, exposures, sector_map)
    _section_risk(story, drawdowns or [], stress or {})
    _section_analytics(story, monthly_pivot)
    _section_trade_history(story, portfolio_log_pd)

    # Footer disclaimer
    story.append(PageBreak())
    story.append(_sp(0.3))
    story.append(Paragraph(
        "QuantPipe Performance Report &nbsp;·&nbsp; "
        f"Generated {date.today().strftime('%B %d, %Y')}",
        _style("end", fontSize=9, textColor=_GRAY, alignment=TA_CENTER),
    ))
    story.append(_sp(0.1))
    story.append(Paragraph(
        "This report is for research and paper trading purposes only. "
        "It does not constitute investment advice. Past performance does not "
        "guarantee future results.",
        _style("disc", fontSize=8, textColor=_GRAY, alignment=TA_CENTER, leading=12),
    ))

    doc.build(story, onFirstPage=_footer_canvas, onLaterPages=_footer_canvas)
    return buf.getvalue()
