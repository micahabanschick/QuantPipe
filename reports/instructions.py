"""Guide & Glossary — navigational help and metric reference (Dashboard #3).

Run via app.py (st.set_page_config is called there).
"""

import streamlit as st

from reports._theme import CSS, COLORS, badge, section_label, kpi_card, page_header

st.markdown(CSS, unsafe_allow_html=True)

# ── Callout helpers ───────────────────────────────────────────────────────────

def tip(text: str) -> str:
    c = COLORS["positive"]
    return (
        f'<div style="background:{c}12;border-left:3px solid {c};'
        f'border-radius:0 6px 6px 0;padding:10px 14px;margin:10px 0;'
        f'font-size:0.85rem;color:{COLORS["text"]};line-height:1.6;">'
        f'<span style="color:{c};font-weight:700;font-size:0.72rem;'
        f'text-transform:uppercase;letter-spacing:0.08em;">Tip &nbsp;</span>{text}</div>'
    )


def warn(text: str) -> str:
    c = COLORS["warning"]
    return (
        f'<div style="background:{c}12;border-left:3px solid {c};'
        f'border-radius:0 6px 6px 0;padding:10px 14px;margin:10px 0;'
        f'font-size:0.85rem;color:{COLORS["text"]};line-height:1.6;">'
        f'<span style="color:{c};font-weight:700;font-size:0.72rem;'
        f'text-transform:uppercase;letter-spacing:0.08em;">Note &nbsp;</span>{text}</div>'
    )


def info(text: str) -> str:
    c = COLORS["blue"]
    return (
        f'<div style="background:{c}12;border-left:3px solid {c};'
        f'border-radius:0 6px 6px 0;padding:10px 14px;margin:10px 0;'
        f'font-size:0.85rem;color:{COLORS["text"]};line-height:1.6;">'
        f'<span style="color:{c};font-weight:700;font-size:0.72rem;'
        f'text-transform:uppercase;letter-spacing:0.08em;">Info &nbsp;</span>{text}</div>'
    )


def metric_def(name: str, formula: str, definition: str,
               interpret: str = "", accent: str | None = None) -> str:
    ac = accent or COLORS["border"]
    form_html = (
        f'<div style="font-family:monospace;font-size:0.78rem;color:{COLORS["blue"]};'
        f'background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:4px;padding:4px 10px;display:inline-block;margin:5px 0 6px;">'
        f'{formula}</div>'
    ) if formula else ""
    interp_html = (
        f'<div style="color:{COLORS["neutral"]};font-size:0.8rem;margin-top:4px;">'
        f'<span style="color:{COLORS["text_muted"]};font-size:0.68rem;'
        f'text-transform:uppercase;letter-spacing:0.07em;font-weight:600;">Interpret: </span>'
        f'{interpret}</div>'
    ) if interpret else ""
    return (
        f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
        f'border-left:3px solid {ac};border-radius:2px 8px 8px 2px;'
        f'padding:13px 16px;margin:6px 0;">'
        f'<div style="color:{COLORS["text"]};font-size:0.9rem;font-weight:700;'
        f'margin-bottom:4px;">{name}</div>'
        f'{form_html}'
        f'<div style="color:{COLORS["neutral"]};font-size:0.82rem;line-height:1.55;">'
        f'{definition}</div>'
        f'{interp_html}'
        f'</div>'
    )


def cmd(text: str) -> str:
    return (
        f'<code style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:4px;padding:3px 8px;font-size:0.8rem;color:{COLORS["blue"]};'
        f'font-family:monospace;">{text}</code>'
    )


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Guide & Glossary",
        "How to navigate, interpret, and operate QuantPipe",
    ),
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_perf, tab_health, tab_lab, tab_trading, tab_glossary, tab_ops = st.tabs([
    "  Overview  ",
    "  Performance  ",
    "  Health Dashboard  ",
    "  Strategy Lab  ",
    "  Paper & Live Trading  ",
    "  Metrics Glossary  ",
    "  Pipeline & Ops  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(section_label("What is QuantPipe?"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.88rem;line-height:1.75;">
QuantPipe is a self-contained quantitative finance pipeline for <b style="color:{COLORS['text']}">
ETF rotation and cryptocurrency systematic trading</b>. It covers the full lifecycle
from raw price ingestion to paper-trade execution:
</div>
""", unsafe_allow_html=True)

        phases = [
            ("0", "Configuration", "Universe definitions, settings, broker credentials"),
            ("1", "Data Ingestion", "Daily OHLCV prices from Yahoo Finance (equity) and Kraken (crypto)"),
            ("2", "Feature Engineering", "12-month momentum, 21-day realised vol, sector mappings"),
            ("3", "Signal Generation", "Cross-sectional momentum ranking → top-N selection"),
            ("4", "Portfolio Construction", "Equal-weight, min-variance, max-Sharpe, risk-parity"),
            ("5", "Risk Management",  "Pre-trade checks, VaR, CVaR, stress scenarios, concentration limits"),
            ("6", "Execution",        "Paper broker, IBKR adapter, CCXT crypto broker, position reconciler"),
        ]
        for num, name, desc in phases:
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:7px 0;'
                f'border-bottom:1px solid {COLORS["border_dim"]};">'
                f'<div style="min-width:26px;height:26px;border-radius:50%;'
                f'background:{COLORS["positive"]}22;border:1px solid {COLORS["positive"]}44;'
                f'display:flex;align-items:center;justify-content:center;'
                f'color:{COLORS["positive"]};font-size:0.72rem;font-weight:700;'
                f'flex-shrink:0;">{num}</div>'
                f'<div><div style="color:{COLORS["text"]};font-size:0.85rem;'
                f'font-weight:600;">{name}</div>'
                f'<div style="color:{COLORS["neutral"]};font-size:0.78rem;">{desc}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown(section_label("Navigation"), unsafe_allow_html=True)
        nav_items = [
            ("🔧", "Pipeline Health",  "System status, data freshness, log viewer"),
            ("📈", "Performance",      "Backtest tearsheet, portfolio analytics, risk"),
            ("⚗️", "Strategy Lab",     "Code editor, backtester, parameter sweep, walk-forward"),
            ("🔬", "Research",         "Signal scanner, factor analysis, walk-forward folds"),
            ("💼", "Portfolio",        "Multi-strategy overview, optimizer, deployment control"),
            ("📄", "Paper Trading",    "Live equity curve and order history for the paper account"),
            ("⚡", "Live Trading",     "Real-money IBKR account monitoring"),
            ("📖", "Guide & Glossary", "This page — metric definitions and how-to"),
        ]
        for icon, title, desc in nav_items:
            st.markdown(
                f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
                f'border-radius:8px;padding:12px 14px;margin:6px 0;'
                f'display:flex;gap:12px;align-items:flex-start;">'
                f'<span style="font-size:1.3rem;line-height:1.4;">{icon}</span>'
                f'<div><div style="color:{COLORS["text"]};font-size:0.87rem;'
                f'font-weight:600;">{title}</div>'
                f'<div style="color:{COLORS["neutral"]};font-size:0.78rem;'
                f'margin-top:2px;">{desc}</div></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(section_label("Default Strategy"), unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
            f'border-radius:8px;padding:14px 16px;font-size:0.82rem;'
            f'color:{COLORS["neutral"]};line-height:1.8;">'
            f'<b style="color:{COLORS["text"]};">Canary</b> — Cross-sectional 12-1 momentum<br/>'
            f'Universe: 26 equity ETFs · Selection: Top-5<br/>'
            f'Weighting: Equal weight · Rebalance: Monthly<br/>'
            f'Cost: 5 bps/trade<br/>'
            f'<span style="font-size:0.78rem;color:{COLORS["text_muted"]};">'
            f'Backtest results visible in the Performance and Strategy Lab tabs.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(section_label("Typical Workflow"), unsafe_allow_html=True)
    steps = [
        ("1", COLORS["blue"],     "Launch", f"Run {cmd('streamlit run app.py')} and open the browser URL shown."),
        ("2", COLORS["positive"], "Check Health", "Open <b>Pipeline Health</b> → Status tab. Verify all components are green."),
        ("3", COLORS["warning"],  "Review Performance", "Open <b>Performance</b> → Overview tab. Check equity curve and KPIs against benchmarks."),
        ("4", COLORS["purple"],   "Inspect Portfolio", "Open <b>Portfolio</b> to review multi-strategy allocation, optimizer, and deployment config."),
        ("5", COLORS["neutral"],  "Research / Lab", "Use <b>Strategy Lab</b> to edit and backtest strategies. Run a parameter sweep or walk-forward to validate."),
        ("6", COLORS["teal"],     "Deploy Strategy", "In Portfolio → Deployment tab, toggle active strategies and save config."),
        ("7", COLORS["orange"],   "Paper Trade", f"Run {cmd('uv run python orchestration/rebalance.py --broker paper')} or let the scheduler handle it."),
        ("8", COLORS["blue"],     "Monitor Paper", "Open <b>Paper Trading</b> to track the live equity curve, NAV, and order history."),
        ("9", COLORS["teal"],     "Run Pipeline", f"Execute {cmd('uv run python orchestration/run_pipeline.py')} daily (or let Task Scheduler handle it)."),
    ]
    cols = st.columns(3)
    for idx, (num, color, title, desc) in enumerate(steps):
        with cols[idx % 3]:
            st.markdown(
                f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
                f'border-top:3px solid {color};border-radius:2px 2px 8px 8px;'
                f'padding:12px 14px;margin-bottom:8px;min-height:90px;">'
                f'<div style="color:{color};font-size:0.68rem;text-transform:uppercase;'
                f'letter-spacing:0.1em;font-weight:700;margin-bottom:5px;">Step {num}</div>'
                f'<div style="color:{COLORS["text"]};font-size:0.87rem;'
                f'font-weight:700;margin-bottom:5px;">{title}</div>'
                f'<div style="color:{COLORS["neutral"]};font-size:0.78rem;'
                f'line-height:1.5;">{desc}</div></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_perf:
    st.markdown(
        info("The Performance dashboard runs a live backtest each time it loads. "
             "The first load takes 5–15 seconds; subsequent loads use a 5-minute cache."),
        unsafe_allow_html=True,
    )

    subtab_ov, subtab_port, subtab_risk, subtab_analytics = st.tabs([
        "Overview Tab", "Portfolio Tab", "Risk Tab", "Analytics Tab",
    ])

    with subtab_ov:
        st.markdown(section_label("KPI Cards (top two rows)"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                metric_def("Total Return",
                           "(Final value / Initial value) − 1",
                           "Cumulative percentage gain or loss over the entire backtest period.",
                           "Higher is better. Compare against benchmark (SPY) shown on the equity curve.",
                           COLORS["positive"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_def("CAGR",
                           "(Final / Initial)^(1/years) − 1",
                           "Compound Annual Growth Rate — the smoothed annual return assuming constant compounding.",
                           "A CAGR above 10% with a Sharpe above 1 is considered strong for a long-only strategy.",
                           COLORS["teal"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_def("Sharpe Ratio",
                           "Ann. Return / Ann. Volatility",
                           "Risk-adjusted return using total volatility as the denominator. Uses daily returns, annualised by ×√252.",
                           "> 1.0 is good; > 1.5 is excellent. Below 0.5 suggests poor risk/reward.",
                           COLORS["blue"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_def("Sortino Ratio",
                           "Ann. Return / Downside Deviation",
                           "Like Sharpe, but only penalises downside volatility (negative return days). More relevant for asymmetric strategies.",
                           "Generally higher than Sharpe for strategies that limit drawdowns. > 1.5 is good.",
                           COLORS["purple"]),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                metric_def("Max Drawdown",
                           "min( (Value − Peak) / Peak )",
                           "The worst peak-to-trough decline in portfolio value over the entire history.",
                           "−15% to −20% is typical for long-only equity strategies. More negative = higher tail risk.",
                           COLORS["negative"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_def("Calmar Ratio",
                           "CAGR / |Max Drawdown|",
                           "Return per unit of max drawdown risk. Useful for comparing strategies with different risk profiles.",
                           "> 0.5 is acceptable; > 1.0 is strong. Penalises deep drawdowns heavily.",
                           COLORS["orange"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_def("Ann. Volatility",
                           "Std(daily returns) × √252",
                           "Annualised standard deviation of daily returns. Measures how much the portfolio fluctuates day-to-day.",
                           "10–15% is typical for diversified equity strategies. Higher vol = wider swings.",
                           COLORS["warning"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_def("Win Rate",
                           "# days with positive return / # total trading days",
                           "The percentage of trading days where the portfolio gained value.",
                           "Most strategies sit 50–55%. Above 55% with positive Sharpe is unusual and worth examining for bias.",
                           COLORS["neutral"]),
                unsafe_allow_html=True,
            )

        st.markdown(section_label("Equity Curve"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
The main chart shows portfolio value over time (starting at $100,000). Key features:
<ul style="margin:6px 0;padding-left:20px;">
  <li><b style="color:{COLORS['text']};">Benchmark overlay</b> — use the sidebar to select SPY, QQQ, IWM, or AGG.
      The benchmark is normalised to the same starting value so growth rates are directly comparable.</li>
  <li><b style="color:{COLORS['text']};">Rebalance markers</b> — dotted vertical lines show when the portfolio was rebalanced
      (toggle off in sidebar if distracting).</li>
  <li><b style="color:{COLORS['text']};">Range selector</b> — click 3M / 6M / 1Y / 3Y / All to zoom into a sub-period.</li>
  <li><b style="color:{COLORS['text']};">Hover</b> — hover over any point to see the exact date and value.</li>
</ul>
</div>
""", unsafe_allow_html=True)

        st.markdown(section_label("Trailing Returns Bar"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Six period returns: <b style="color:{COLORS['text']};">MTD</b> (month-to-date),
<b style="color:{COLORS['text']};">QTD</b> (quarter-to-date),
<b style="color:{COLORS['text']};">YTD</b> (year-to-date), <b style="color:{COLORS['text']};">1Y</b>,
<b style="color:{COLORS['text']};">3Y</b>, <b style="color:{COLORS['text']};">5Y</b>.
Green bars = positive. Red bars = negative. Missing bars mean insufficient history.
</div>
""", unsafe_allow_html=True)

    with subtab_port:
        st.markdown(section_label("Holdings Table"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Shows the <b style="color:{COLORS['text']};">most recent rebalance</b> target weights. Columns:
<ul style="margin:6px 0;padding-left:20px;">
  <li><b style="color:{COLORS['text']};">Symbol</b> — ETF ticker.</li>
  <li><b style="color:{COLORS['text']};">Weight</b> — target allocation as a percentage of NAV.</li>
  <li><b style="color:{COLORS['text']};">Sector</b> — GICS sector classification for the ETF.</li>
</ul>
</div>
""", unsafe_allow_html=True)
        st.markdown(
            tip("The strategy always holds exactly 5 positions at equal weight (20% each). "
                "Weights may drift slightly between rebalances as prices move."),
            unsafe_allow_html=True,
        )

        st.markdown(section_label("Sector Donut Chart"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Aggregate exposure by GICS sector. The donut shows how much of the portfolio is concentrated in
each sector. Heavy concentration in one sector (>40%) raises idiosyncratic risk.
</div>
""", unsafe_allow_html=True)

        st.markdown(section_label("Weight History"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Only visible once the signal generator has run on multiple dates. Each line represents
one symbol; the Y-axis is target weight. A line dropping to zero means the symbol
left the top-5 at that rebalance. A new line appearing means it entered.
</div>
""", unsafe_allow_html=True)

    with subtab_risk:
        st.markdown(section_label("Risk KPI Cards"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                metric_def("1-Day VaR 95%",
                           "5th percentile of daily returns",
                           "Value at Risk — the maximum daily loss you would expect to NOT exceed on 95% of trading days.",
                           "A VaR of −1.6% means on a typical bad day you lose less than 1.6% of NAV. 1 in 20 days is worse.",
                           COLORS["negative"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_def("CVaR 95% (Expected Shortfall)",
                           "Mean of returns ≤ VaR 95%",
                           "The average loss on the worst 5% of days. More conservative than VaR because it captures tail severity.",
                           "CVaR is always worse than VaR. Large gaps between them indicate fat tails or rare extreme events.",
                           COLORS["orange"]),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                metric_def("Best Day / Worst Day",
                           "max / min of daily return series",
                           "The single best and worst one-day returns in the entire backtest history.",
                           "Worst day provides a gut-check: could you emotionally handle losing that much in a single day?",
                           COLORS["blue"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_def("1-Day VaR 99%",
                           "1st percentile of daily returns",
                           "Stricter VaR: the loss threshold exceeded on only 1% of days (roughly 2–3 times per year).",
                           "Compare VaR99 to VaR95. If VaR99 >> 2×VaR95, the tail distribution is heavy.",
                           COLORS["negative"]),
                unsafe_allow_html=True,
            )

        st.markdown(section_label("Stress Scenarios"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Each bar shows the estimated P&amp;L if the <b style="color:{COLORS['text']};">current portfolio</b>
had been held during a historical crisis, using that crisis's average asset shocks:
<ul style="margin:6px 0;padding-left:20px;">
  <li><b style="color:{COLORS['text']};">2008 GFC</b> — Global Financial Crisis equity sell-off (equity −40%, bonds +8%)</li>
  <li><b style="color:{COLORS['text']};">2020 COVID</b> — March 2020 crash (equity −30% in 30 days)</li>
  <li><b style="color:{COLORS['text']};">2022 Rates</b> — Rate-hike shock (equity −20%, bonds −15%)</li>
  <li><b style="color:{COLORS['text']};">2000 Dot-com</b> — Tech bubble burst (growth −40%, value flat)</li>
</ul>
These are <b style="color:{COLORS['warning']};">illustrative</b>, not precise — they apply fixed shocks
rather than replicating intra-period dynamics.
</div>
""", unsafe_allow_html=True)

        st.markdown(section_label("Top Drawdown Periods Table"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
The five worst drawdown episodes. Columns:
<ul style="margin:6px 0;padding-left:20px;">
  <li><b style="color:{COLORS['text']};">Start</b> — date the portfolio first fell below the prior peak</li>
  <li><b style="color:{COLORS['text']};">Trough</b> — date the maximum drawdown depth was reached</li>
  <li><b style="color:{COLORS['text']};">Recovery</b> — date the portfolio recovered to a new all-time high ("Ongoing" if not yet recovered)</li>
  <li><b style="color:{COLORS['text']};">Depth (d)</b> — trading days from start to trough</li>
  <li><b style="color:{COLORS['text']};">Recov (d)</b> — trading days from trough to recovery</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with subtab_analytics:
        st.markdown(section_label("Monthly Returns Heatmap"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Each cell is the total return for that calendar month. <span style="color:{COLORS['positive']};font-weight:600;">Green</span> =
positive, <span style="color:{COLORS['negative']};font-weight:600;">red</span> = negative.
Intensity reflects magnitude. Hover for exact values. Use this to identify seasonal patterns or
periods of persistent underperformance.
</div>
""", unsafe_allow_html=True)

        st.markdown(section_label("Daily Return Distribution"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Histogram of all daily returns. Overlays:
<ul style="margin:6px 0;padding-left:20px;">
  <li><span style="color:{COLORS['warning']};font-weight:600;">Orange curve</span> — fitted normal distribution with same mean and standard deviation</li>
  <li><span style="color:{COLORS['negative']};font-weight:600;">Dotted lines</span> — VaR 95% and VaR 99% thresholds</li>
</ul>
If the actual distribution has a fatter left tail than the normal curve, the strategy has
<b style="color:{COLORS['text']};">negative skew</b> — large losses occur more often than a normal model would predict.
</div>
""", unsafe_allow_html=True)

        st.markdown(section_label("Holdings Correlation Matrix"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Pairwise 252-day rolling return correlations for the <b style="color:{COLORS['text']};">current holdings</b>.
Values range from −1 (perfect negative correlation) to +1 (perfect positive correlation).
<br/><br/>
<span style="color:{COLORS['positive']};font-weight:600;">For risk purposes</span>: high correlation among all holdings (>0.8) means the portfolio
offers limited diversification. The diagonal is always 1.0 (each asset with itself).
</div>
""", unsafe_allow_html=True)
        st.markdown(
            warn("High correlations during normal markets often spike toward 1.0 in crisis periods — "
                 "the so-called 'correlation breakdown'. Stress scenario results account for this implicitly."),
            unsafe_allow_html=True,
        )

        st.markdown(section_label("Rolling Sharpe & Sortino"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Rolling N-day (configurable in sidebar) annualised Sharpe and Sortino ratios.
The dotted line at 1.0 is a common quality threshold.
<ul style="margin:6px 0;padding-left:20px;">
  <li>Periods where the line is <b style="color:{COLORS['text']};">consistently above 1</b> indicate the strategy is working well.</li>
  <li>Periods <b style="color:{COLORS['text']};">below 0</b> are drawdown regimes.</li>
  <li>Large divergence between Sharpe and Sortino suggests asymmetric return distribution.</li>
</ul>
Use the sidebar <b style="color:{COLORS['text']};">Rolling window</b> selector (21d / 63d / 126d / 252d)
to adjust the smoothing level.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEALTH DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_health:
    st.markdown(section_label("Status Tab"), unsafe_allow_html=True)
    st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
The top banner summarises overall pipeline health:
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, label, desc, color in [
        (c1, "🟢 HEALTHY",  "All data sources fresh and pipeline completed successfully.", COLORS["positive"]),
        (c2, "🟡 WARNING",  "One or more sources are stale (>30h) or pipeline has not completed.", COLORS["warning"]),
        (c3, "🔴 CRITICAL", "A required data file is entirely missing. Action required.", COLORS["negative"]),
    ]:
        col.markdown(
            f'<div style="background:{color}12;border:1px solid {color}44;'
            f'border-radius:8px;padding:10px 12px;text-align:center;">'
            f'<div style="color:{color};font-weight:700;font-size:0.87rem;">{label}</div>'
            f'<div style="color:{COLORS["neutral"]};font-size:0.78rem;margin-top:4px;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown(section_label("Pipeline Component Cards"), unsafe_allow_html=True)
    st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Four cards show the age of each data component:
<ul style="margin:6px 0;padding-left:20px;">
  <li><b style="color:{COLORS['text']};">Equity Ingest</b> — age of most recently modified equity Parquet file. Should be &lt;30h on trading days.</li>
  <li><b style="color:{COLORS['text']};">Crypto Ingest</b> — same for crypto. Crypto trades 24/7 so &lt;30h is always expected.</li>
  <li><b style="color:{COLORS['text']};">Signal Generate</b> — age of <code>target_weights.parquet</code>. Signals run after ingest.</li>
  <li><b style="color:{COLORS['text']};">Last Pipeline Run</b> — parsed from <code>logs/pipeline.log</code>.</li>
</ul>
Accent colours: <span style="color:{COLORS['positive']};font-weight:600;">teal</span> = fresh,
<span style="color:{COLORS['warning']};font-weight:600;">amber</span> = moderately stale,
<span style="color:{COLORS['negative']};font-weight:600;">red</span> = critically stale.
</div>
""", unsafe_allow_html=True)

    st.markdown(section_label("Data Freshness Bar Chart"), unsafe_allow_html=True)
    st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Horizontal bars show the age of each source in hours. Reference lines:
<ul style="margin:6px 0;padding-left:20px;">
  <li><span style="color:{COLORS['warning']};font-weight:600;">24h amber line</span> — warning threshold</li>
  <li><span style="color:{COLORS['negative']};font-weight:600;">48h red line</span> — critical threshold</li>
</ul>
Bars capped at 72h for readability; hover shows exact age.
</div>
""", unsafe_allow_html=True)

    st.markdown(section_label("Data Quality Tab"), unsafe_allow_html=True)
    st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Shows row counts for each symbol over the last 30 days. Status column:
<ul style="margin:6px 0;padding-left:20px;">
  <li><span style="color:{COLORS['positive']};font-weight:600;">OK</span> — latest date is within 5 trading days of today</li>
  <li><span style="color:{COLORS['negative']};font-weight:600;">STALE</span> — latest date is older than 5 days; data likely missing or exchange was closed</li>
</ul>
The bar chart on the right makes it easy to spot symbols with anomalously low row counts
(missing days, API errors, corporate actions that changed the ticker).
</div>
""", unsafe_allow_html=True)

    st.markdown(section_label("Logs Tab"), unsafe_allow_html=True)
    st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
The last 80 lines of each log file. Error badges show at a glance whether any issues occurred.
The expander is auto-opened when errors are present. Look for lines containing:
<ul style="margin:6px 0;padding-left:20px;">
  <li><code style="color:{COLORS['negative']};">ERROR</code> — a step failed but execution continued</li>
  <li><code style="color:{COLORS['negative']};">FAILED</code> — a step aborted with a non-zero exit code</li>
  <li><code style="color:{COLORS['warning']};">ALERT</code> — a pre-trade limit was triggered (not necessarily an error)</li>
</ul>
</div>
""", unsafe_allow_html=True)
    st.markdown(
        tip("Log files are in the <code>logs/</code> directory. "
            "pipeline.log is the master log; ingest.log and signals.log are written by sub-processes."),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STRATEGY LAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_lab:
    st.markdown(
        info("The Strategy Lab is where you write, edit, and backtest strategy code entirely within the browser. "
             "All changes are saved to the <code>strategies/</code> directory."),
        unsafe_allow_html=True,
    )

    subtab_editor, subtab_bt, subtab_sweep, subtab_wf = st.tabs([
        "Code Editor", "Backtesting", "Parameter Sweep", "Walk-Forward",
    ])

    with subtab_editor:
        st.markdown(section_label("Strategy File Structure"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Each strategy lives in its own folder under <code>strategies/&lt;slug&gt;/</code>
and must contain at minimum a Python file with the following interface:
</div>
""", unsafe_allow_html=True)
        st.code("""NAME        = "My Strategy"
DESCRIPTION = "One-line description"
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          5,
    "cost_bps":       5.0,
    "weight_scheme":  "equal",   # or "vol_scaled"
}

def get_signal(features, rebal_dates, top_n, **kwargs):
    ...  # return a Polars DataFrame [rebalance_date, symbol, rank/score]

def get_weights(signal, weight_scheme, **kwargs):
    ...  # return a Polars DataFrame [rebalance_date, symbol, weight]""", language="python")
        st.markdown(
            tip("Click <b>➕ New</b> in the Strategy Lab to create a pre-filled template, then "
                "customise the <code>get_signal()</code> function to implement your logic."),
            unsafe_allow_html=True,
        )
        st.markdown(section_label("Validate Button"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Clicking <b style="color:{COLORS['text']};">✔ Validate</b> runs three checks without executing a full backtest:
<ul style="margin:8px 0;padding-left:20px;">
  <li><b>Syntax</b> — Python compile check; reports line number on error.</li>
  <li><b>Import</b> — Attempts to load the module; catches missing dependencies or bad imports.</li>
  <li><b>Interface</b> — Verifies that <code>get_signal</code>, <code>get_weights</code>,
      <code>NAME</code>, <code>DESCRIPTION</code>, and all required <code>DEFAULT_PARAMS</code>
      keys are present.</li>
</ul>
All checks must pass before running a backtest.
</div>
""", unsafe_allow_html=True)

    with subtab_bt:
        st.markdown(section_label("Running a Backtest"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Click <b style="color:{COLORS['text']};">▶ Run Backtest</b> with the desired configuration:
<ul style="margin:8px 0;padding-left:20px;">
  <li><b>Lookback (years)</b> — History window for the simulation. Longer windows improve statistical
      confidence but may include stale regimes.</li>
  <li><b>Top-N positions</b> — Number of ETFs held at each rebalance.</li>
  <li><b>Cost (bps)</b> — Round-trip transaction cost per trade (5 bps ≈ 0.05% each way).</li>
  <li><b>Weighting</b> — <code>equal</code> assigns uniform weight; <code>vol_scaled</code>
      gives more weight to lower-volatility holdings.</li>
</ul>
</div>
""", unsafe_allow_html=True)
        st.markdown(section_label("Result Tabs"), unsafe_allow_html=True)
        result_tabs = [
            ("Equity Curve", "Strategy NAV vs. SPY (scaled to same starting value). Hover for daily values."),
            ("Rolling Analytics", "Rolling Sharpe and drawdown charts over a sliding window. Identifies regime changes."),
            ("Monthly Returns", "Year × month heatmap. Red cells = negative months, green = positive."),
            ("Trade Log", "Every simulated order with date, symbol, quantity, price, value, cost, and direction."),
        ]
        for tab_name, desc in result_tabs:
            st.markdown(
                f'<div style="display:flex;gap:10px;padding:8px 0;'
                f'border-bottom:1px solid {COLORS["border_dim"]};">'
                f'<div style="min-width:140px;color:{COLORS["text"]};font-size:0.85rem;'
                f'font-weight:600;">{tab_name}</div>'
                f'<div style="color:{COLORS["neutral"]};font-size:0.82rem;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            tip("Use the <b>⬇ Equity Curve (CSV)</b> and <b>⬇ Trade Log (CSV)</b> download buttons "
                "to export results for further analysis in Excel or a Jupyter notebook."),
            unsafe_allow_html=True,
        )

    with subtab_sweep:
        st.markdown(section_label("Parameter Sweep"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
The parameter sweep runs a full backtest for every combination of
<b style="color:{COLORS['text']};">Top-N</b> and <b style="color:{COLORS['text']};">Lookback</b>
values you specify. Results are shown as a Sharpe ratio heatmap.
</div>
""", unsafe_allow_html=True)
        st.markdown(
            warn("Each cell is a separate backtest. A 4×3 grid = 12 backtests. Keep the grid small "
                 "(≤ 20 cells) to avoid long runtimes. The computation runs in-process, so the "
                 "browser will appear frozen until it completes."),
            unsafe_allow_html=True,
        )
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;margin-top:8px;">
<b style="color:{COLORS['text']};">Interpreting the heatmap:</b>
<ul style="margin:8px 0;padding-left:20px;">
  <li>Green cells → high Sharpe for that combination.</li>
  <li>Red cells → poor fit; avoid those parameters.</li>
  <li>Prefer parameter regions where a <i>cluster</i> of cells is green — a single bright cell
      surrounded by red suggests overfitting.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with subtab_wf:
        st.markdown(section_label("Walk-Forward Validation"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Walk-forward splits the historical data into an
<b style="color:{COLORS['text']};">In-Sample (IS)</b> period used to develop the strategy
and an <b style="color:{COLORS['text']};">Out-of-Sample (OOS)</b> period used to test whether
the strategy holds up on unseen data.
</div>
""", unsafe_allow_html=True)
        st.markdown(
            warn("If OOS Sharpe drops sharply relative to IS Sharpe, the strategy is likely overfit. "
                 "A robust strategy should show OOS Sharpe within ~30% of its IS Sharpe."),
            unsafe_allow_html=True,
        )
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;margin-top:8px;">
The <b style="color:{COLORS['text']};">OOS fraction slider</b> controls how much of the data
is held out. A value of 0.30 means the last 30% of the history is the test set.
The equity curves are indexed to 100 so both periods start at the same point for easy comparison.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PAPER & LIVE TRADING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_trading:
    st.markdown(
        info("Paper trading simulates real execution using a $100,000 virtual account. "
             "Live trading connects to IBKR TWS/Gateway for real-money execution."),
        unsafe_allow_html=True,
    )

    subtab_paper, subtab_live, subtab_setup = st.tabs([
        "Paper Trading", "Live Trading", "IBKR Setup",
    ])

    with subtab_paper:
        st.markdown(section_label("Paper Trading Dashboard"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
The Paper Trading dashboard shows the live performance of the deployed strategy configuration
using a virtual $100,000 account:
<ul style="margin:8px 0;padding-left:20px;">
  <li><b style="color:{COLORS['text']};">Equity Curve</b> — Daily NAV computed from
      target weights × live prices, anchored to actual NAV snapshots after each rebalance.</li>
  <li><b style="color:{COLORS['text']};">Rebalance dots</b> — Markers on the curve showing
      each rebalance event.</li>
  <li><b style="color:{COLORS['text']};">Deployment markers</b> — Vertical dotted lines
      whenever the active strategy configuration changed.</li>
  <li><b style="color:{COLORS['text']};">Current Positions</b> — Latest target weights with
      estimated dollar values and a pie chart.</li>
  <li><b style="color:{COLORS['text']};">Trade History</b> — All order attempts with status
      (FILLED / REJECTED / PENDING).</li>
</ul>
</div>
""", unsafe_allow_html=True)
        st.markdown(section_label("Running a Paper Rebalance"), unsafe_allow_html=True)
        st.code("uv run python orchestration/rebalance.py --broker paper", language="bash")
        st.markdown(
            tip("The Task Scheduler runs this automatically at 16:30 Mon–Fri. "
                "Click <b>Refresh</b> on the dashboard after the rebalance completes to see updated data."),
            unsafe_allow_html=True,
        )

    with subtab_live:
        st.markdown(section_label("Live Trading Dashboard"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
The Live Trading dashboard connects to IBKR TWS/Gateway and displays:
<ul style="margin:8px 0;padding-left:20px;">
  <li><b style="color:{COLORS['text']};">Connection status</b> — Shows whether TWS/Gateway
      is reachable on the configured host and port.</li>
  <li><b style="color:{COLORS['text']};">Current NAV</b> — Net Liquidation Value pulled
      directly from the IBKR account.</li>
  <li><b style="color:{COLORS['text']};">Open positions</b> — Current holdings with quantity
      and average cost.</li>
  <li><b style="color:{COLORS['text']};">NAV history</b> — Snapshots recorded after each
      live rebalance.</li>
</ul>
</div>
""", unsafe_allow_html=True)
        st.markdown(
            warn("Live trading is intentionally minimal. The paper trading tab is the primary "
                 "monitoring tool. The live tab requires TWS or IB Gateway to be running and "
                 "accessible on the network."),
            unsafe_allow_html=True,
        )

    with subtab_setup:
        st.markdown(section_label("IBKR Connection Settings"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.75;">
Configure the IBKR connection in <code>config/settings.py</code>:
</div>
""", unsafe_allow_html=True)
        st.code("""IBKR_HOST      = "127.0.0.1"   # TWS/Gateway host
IBKR_PORT      = 7497          # 7497 = paper, 7496 = live
IBKR_CLIENT_ID = 1             # Unique ID per connected client""", language="python")
        st.markdown(
            warn("Always test with paper trading (port 7497) before switching to live (port 7496). "
                 "Set <code>IBKR_PORT = 7496</code> only when you are ready for real-money execution."),
            unsafe_allow_html=True,
        )
        st.markdown(section_label("Troubleshooting IBKR"), unsafe_allow_html=True)
        ib_issues = [
            ("Connection refused / timeout",
             "Ensure TWS or IB Gateway is running and 'Allow API connections' is checked in "
             "TWS → Configuration → API → Settings."),
            ("Wrong client ID error",
             "Each client connection needs a unique clientId. If another script or the dashboard "
             "is already connected, increment IBKR_CLIENT_ID."),
            ("Market data frozen / no prices",
             "IB paper accounts require a market data subscription. Check TWS → Account → Market Data."),
            ("Rebalance fails for live broker",
             "Run with --broker paper first to verify the order flow, then switch to --broker ibkr."),
        ]
        for problem, solution in ib_issues:
            with st.expander(problem):
                st.markdown(
                    f'<div style="color:{COLORS["neutral"]};font-size:0.83rem;'
                    f'line-height:1.6;">{solution}</div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — METRICS GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_glossary:
    st.markdown(section_label("Performance Metrics"), unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    left_metrics = [
        ("CAGR", "(V_n / V_0)^(1/T) − 1",
         "Compound Annual Growth Rate. The annualised return assuming reinvestment.",
         "Target > 10% for equity-like strategies.", COLORS["teal"]),
        ("Sharpe Ratio", "μ_ann / σ_ann",
         "Annualised excess return divided by annualised volatility (risk-free rate assumed 0 for simplicity).",
         "> 1.0 good, > 1.5 excellent.", COLORS["blue"]),
        ("Sortino Ratio", "μ_ann / σ_downside",
         "Like Sharpe but uses downside deviation (standard deviation of negative returns only) as the denominator.",
         "Preferred over Sharpe for skewed distributions.", COLORS["purple"]),
        ("Calmar Ratio", "CAGR / |Max DD|",
         "Return per unit of maximum drawdown risk. Common in hedge fund analysis.",
         "> 0.5 acceptable, > 1.0 strong.", COLORS["orange"]),
        ("Win Rate", "# positive days / # total days",
         "Fraction of trading days with a positive return.",
         "50–55% is typical; above 60% warrants scrutiny.", COLORS["neutral"]),
    ]
    right_metrics = [
        ("Ann. Volatility", "σ_daily × √252",
         "Annualised standard deviation of daily returns. Primary measure of total risk.",
         "10–18% is typical for long-only equity.", COLORS["warning"]),
        ("Max Drawdown", "min((V_t − peak) / peak)",
         "Worst peak-to-trough decline over the entire history. A measure of tail risk.",
         "Deeper than −30% often causes investor redemptions.", COLORS["negative"]),
        ("VaR 95% / 99%", "5th / 1st percentile(daily returns)",
         "Value at Risk. The daily loss exceeded only 5% / 1% of the time.",
         "Multiply by √horizon to scale to multi-day VaR.", COLORS["negative"]),
        ("CVaR 95% (ES)", "E[r | r ≤ VaR 95%]",
         "Conditional VaR / Expected Shortfall. The average loss on the worst 5% of days. Coherent risk measure.",
         "Always worse than VaR; shows tail severity.", COLORS["orange"]),
        ("Trailing Returns", "V_t / V_{t−n} − 1",
         "Point-in-time returns over standard lookback windows (MTD, QTD, YTD, 1Y, 3Y, 5Y).",
         "Useful for comparing against a benchmark over the same window.", COLORS["neutral"]),
    ]
    with col_l:
        for args in left_metrics:
            st.markdown(metric_def(*args), unsafe_allow_html=True)
    with col_r:
        for args in right_metrics:
            st.markdown(metric_def(*args), unsafe_allow_html=True)

    st.markdown(section_label("Portfolio & Exposure Metrics"), unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    exp_left = [
        ("Gross Exposure", "Σ |w_i|",
         "Sum of absolute weights. For a long-only portfolio equals 1.0 (100%). Values above 1.0 imply leverage.",
         "Should equal 100% for unlevered strategies.", COLORS["blue"]),
        ("Top-5 Concentration", "Σ top-5 weights",
         "Sum of the five largest position weights. Measures portfolio concentration.",
         "The Canary strategy always equals 100% (5 positions × 20%). For larger universes, >80% is a concern.",
         COLORS["warning"]),
    ]
    exp_right = [
        ("Largest Name", "max(w_i)",
         "The single largest position weight. For Canary = 20% always (equal weight). Higher values indicate concentration risk.",
         "Pre-trade limits block positions >40% of NAV by default.", COLORS["orange"]),
        ("Rebalance Date", "—",
         "The date the target weights were last updated by the signal generator. "
         "Should align with the first trading day of each month for the Canary strategy.",
         "If today >> last rebalance, the pipeline may not have run.", COLORS["neutral"]),
    ]
    with col_l:
        for args in exp_left:
            st.markdown(metric_def(*args), unsafe_allow_html=True)
    with col_r:
        for args in exp_right:
            st.markdown(metric_def(*args), unsafe_allow_html=True)

    st.markdown(section_label("Risk Engine Limits (defaults)"), unsafe_allow_html=True)
    limits = [
        ("Max Position Size",      "40%",    "Single position weight cap"),
        ("Max Gross Exposure",     "110%",   "Allows slight leverage; blocks >110%"),
        ("Max Top-5 Concentration","80%",    "Overridden to 100% for 5-position strategies"),
        ("VaR 95% Limit",          "5%/day", "Pre-trade check fails if current VaR > 5%"),
    ]
    hdr_cols = st.columns([2, 1, 3])
    hdr_cols[0].markdown(f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;font-weight:700;padding:6px 0;">Limit</div>', unsafe_allow_html=True)
    hdr_cols[1].markdown(f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;font-weight:700;padding:6px 0;">Default</div>', unsafe_allow_html=True)
    hdr_cols[2].markdown(f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;font-weight:700;padding:6px 0;">Description</div>', unsafe_allow_html=True)
    for name, val, desc in limits:
        cols = st.columns([2, 1, 3])
        cols[0].markdown(f'<div style="color:{COLORS["text"]};font-size:0.83rem;padding:5px 0;border-top:1px solid {COLORS["border_dim"]};">{name}</div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div style="color:{COLORS["negative"]};font-family:monospace;font-size:0.83rem;font-weight:700;padding:5px 0;border-top:1px solid {COLORS["border_dim"]};">{val}</div>', unsafe_allow_html=True)
        cols[2].markdown(f'<div style="color:{COLORS["neutral"]};font-size:0.82rem;padding:5px 0;border-top:1px solid {COLORS["border_dim"]};">{desc}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PIPELINE & OPS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ops:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(section_label("Launching the App"), unsafe_allow_html=True)
        st.code("streamlit run app.py", language="bash")
        st.markdown(
            tip("Always use <code>app.py</code> as the entry point, not the individual dashboard files. "
                "The individual files lack <code>st.set_page_config()</code> when run through the navigator."),
            unsafe_allow_html=True,
        )

        st.markdown(section_label("Running the Pipeline Manually"), unsafe_allow_html=True)
        st.code("""# Full pipeline (ingest + signals)
uv run python orchestration/run_pipeline.py

# Skip ingest (signals only)
uv run python orchestration/run_pipeline.py --skip-ingest

# Override date
uv run python orchestration/run_pipeline.py --date 2025-12-31""", language="bash")

        st.markdown(section_label("Paper Rebalance"), unsafe_allow_html=True)
        st.code("""# Paper trading (default)
uv run python orchestration/rebalance.py --broker paper

# Dry run — show orders without placing them
uv run python orchestration/rebalance.py --broker paper --dry-run

# With date override
uv run python orchestration/rebalance.py --broker paper --date 2025-12-31""", language="bash")

        st.markdown(section_label("First-Time Setup"), unsafe_allow_html=True)
        st.code("""# 1 — Install dependencies
uv sync --extra backtest --extra portfolio

# 2 — Backfill historical prices (6 years)
uv run python orchestration/backfill.py

# 3 — Compute features
uv run python features/compute.py

# 4 — Generate first signals
uv run python orchestration/generate_signals.py

# 5 — Register Task Scheduler (Windows)
python orchestration/setup_scheduler.py

# 6 — Launch
streamlit run app.py""", language="bash")

    with col_b:
        st.markdown(section_label("Windows Task Scheduler"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.8;">
Two scheduled tasks are registered under the <code>QuantPipe\\</code> folder:
<ul style="margin:6px 0;padding-left:20px;">
  <li><b style="color:{COLORS['text']};">DailyPipeline</b> — runs at 06:15 Mon–Fri.
      Executes <code>run_pipeline.bat</code> which chains ingest → signal generation.
      Output appended to <code>logs/pipeline.log</code>.</li>
  <li><b style="color:{COLORS['text']};">DailyRebalance</b> — runs at 16:30 Mon–Fri.
      Executes <code>run_rebalance.bat</code> with paper broker.
      Output appended to <code>logs/rebalance.log</code>.</li>
</ul>
To manage tasks:
</div>
""", unsafe_allow_html=True)
        st.code("""# Register (run once)
python orchestration/setup_scheduler.py

# List registered tasks
python orchestration/setup_scheduler.py --list

# Remove all tasks
python orchestration/setup_scheduler.py --remove

# Verify via Windows built-in
schtasks /Query /TN "QuantPipe\\DailyPipeline" /FO LIST""", language="bash")

        st.markdown(section_label("Log Files"), unsafe_allow_html=True)
        logs = [
            ("logs/pipeline.log",   "Master log for the full ingest + signal chain"),
            ("logs/ingest.log",     "Equity and crypto price download details"),
            ("logs/signals.log",    "Signal generation and portfolio construction"),
            ("logs/rebalance.log",  "Paper trade execution and reconciliation"),
        ]
        for path, desc in logs:
            st.markdown(
                f'<div style="display:flex;gap:10px;padding:6px 0;'
                f'border-bottom:1px solid {COLORS["border_dim"]};">'
                f'<code style="font-size:0.75rem;color:{COLORS["blue"]};'
                f'flex-shrink:0;min-width:170px;">{path}</code>'
                f'<span style="color:{COLORS["neutral"]};font-size:0.8rem;">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(section_label("Troubleshooting"), unsafe_allow_html=True)
        issues = [
            ("Health dashboard shows NO DATA",
             "Run the backfill and generate_signals scripts from the First-Time Setup sequence above."),
            ("Equity curve won't load",
             "Check that prices are present in data/bronze/equity/daily/. Run the ingest script if needed."),
            ("Pre-trade check fails",
             "Check portfolio_log.parquet for the failure reason. Likely VaR or concentration limit. "
             "For the 5-position strategy, ensure RiskLimits(max_top5_concentration=1.0) is passed."),
            ("Task Scheduler tasks not running",
             "Check that the .bat files reference the correct absolute path. "
             "Run the .bat file manually to confirm it works before relying on the scheduler."),
            ("Crypto symbol not found (POL/USDT)",
             "Kraken uses MATIC/USDT for Polygon. The universe config maps this correctly."),
            ("Strategy Lab backtest fails",
             "Click ✔ Validate first to catch syntax/import errors. Ensure the strategy file exports "
             "get_signal(), get_weights(), NAME, DESCRIPTION, and DEFAULT_PARAMS."),
            ("IBKR connection refused",
             "TWS or IB Gateway must be running. Enable 'Allow API connections' in TWS Settings → API. "
             "Paper port is 7497; live port is 7496."),
        ]
        for problem, solution in issues:
            with st.expander(problem):
                st.markdown(
                    f'<div style="color:{COLORS["neutral"]};font-size:0.83rem;'
                    f'line-height:1.6;">{solution}</div>',
                    unsafe_allow_html=True,
                )
