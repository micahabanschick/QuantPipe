"""Guide & Glossary — navigational help and metric reference (Dashboard #3)."""

import os
import re

import streamlit as st

from reports._theme import CSS, COLORS, badge, section_label, kpi_card, page_header

st.markdown(CSS, unsafe_allow_html=True)


# ── Callout helpers ───────────────────────────────────────────────────────────

def tip(text):
    c = COLORS["positive"]
    return (f'<div style="background:{c}12;border-left:3px solid {c};border-radius:0 6px 6px 0;'
            f'padding:10px 14px;margin:10px 0;font-size:0.85rem;color:{COLORS["text"]};line-height:1.6;">'
            f'<span style="color:{c};font-weight:700;font-size:0.72rem;text-transform:uppercase;'
            f'letter-spacing:0.08em;">Tip &nbsp;</span>{text}</div>')


def warn(text):
    c = COLORS["warning"]
    return (f'<div style="background:{c}12;border-left:3px solid {c};border-radius:0 6px 6px 0;'
            f'padding:10px 14px;margin:10px 0;font-size:0.85rem;color:{COLORS["text"]};line-height:1.6;">'
            f'<span style="color:{c};font-weight:700;font-size:0.72rem;text-transform:uppercase;'
            f'letter-spacing:0.08em;">Note &nbsp;</span>{text}</div>')


def mdef(name, formula, definition, interpret="", accent=None):
    ac = accent or COLORS["border"]
    form = (f'<div style="font-family:monospace;font-size:0.78rem;color:{COLORS["blue"]};'
            f'background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:4px;padding:4px 10px;display:inline-block;margin:5px 0 6px;">{formula}</div>') if formula else ""
    interp = (f'<div style="color:{COLORS["neutral"]};font-size:0.8rem;margin-top:4px;">'
              f'<span style="color:{COLORS["text_muted"]};font-size:0.68rem;text-transform:uppercase;'
              f'letter-spacing:0.07em;font-weight:600;">Interpret: </span>{interpret}</div>') if interpret else ""
    return (f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid {ac};border-radius:2px 8px 8px 2px;padding:13px 16px;margin:6px 0;">'
            f'<div style="color:{COLORS["text"]};font-size:0.9rem;font-weight:700;margin-bottom:4px;">{name}</div>'
            f'{form}<div style="color:{COLORS["neutral"]};font-size:0.82rem;line-height:1.55;">{definition}</div>'
            f'{interp}</div>')


def cmd(text):
    return (f'<code style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:4px;padding:3px 8px;font-size:0.8rem;color:{COLORS["blue"]};'
            f'font-family:monospace;">{text}</code>')


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(page_header("Guide & Glossary", "How to navigate, interpret, and operate QuantPipe"), unsafe_allow_html=True)

tab_qs, tab_ref, tab_trouble = st.tabs(["  Quickstart  ", "  Metrics Reference  ", "  Troubleshooting  "])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUICKSTART
# ═══════════════════════════════════════════════════════════════════════════════

with tab_qs:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(section_label("What is QuantPipe?"), unsafe_allow_html=True)
        st.markdown(f'<div style="color:{COLORS["neutral"]};font-size:0.88rem;line-height:1.75;">QuantPipe is a self-contained quant finance pipeline for <b style="color:{COLORS["text"]};">ETF rotation and crypto systematic trading</b>. It covers raw price ingestion through paper-trade execution.</div>', unsafe_allow_html=True)

        phases = [
            ("0", "Configuration",   "Universe definitions, settings, broker credentials"),
            ("1", "Data Ingestion",  "Daily OHLCV from Yahoo Finance (equity) and Kraken (crypto)"),
            ("2", "Features",        "12-month momentum, 21-day realised vol, sector mappings"),
            ("3", "Signal",          "Cross-sectional momentum ranking → top-N selection"),
            ("4", "Portfolio",       "Equal-weight, min-variance, max-Sharpe, risk-parity"),
            ("5", "Risk",            "Pre-trade checks, VaR, CVaR, stress scenarios"),
            ("6", "Execution",       "Paper broker, IBKR adapter, CCXT crypto, reconciler"),
        ]
        for num, name, desc in phases:
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:7px 0;border-bottom:1px solid {COLORS["border_dim"]};">'
                f'<div style="min-width:26px;height:26px;border-radius:50%;background:{COLORS["positive"]}22;'
                f'border:1px solid {COLORS["positive"]}44;display:flex;align-items:center;justify-content:center;'
                f'color:{COLORS["positive"]};font-size:0.72rem;font-weight:700;flex-shrink:0;">{num}</div>'
                f'<div><div style="color:{COLORS["text"]};font-size:0.85rem;font-weight:600;">{name}</div>'
                f'<div style="color:{COLORS["neutral"]};font-size:0.78rem;">{desc}</div></div></div>',
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown(section_label("Navigation"), unsafe_allow_html=True)
        nav_items = [
            ("🔧", "Pipeline Health",  "System status, data freshness, log viewer"),
            ("🧪", "Data Lab",         "FRED connector, alt-data ingestion, tradability checks"),
            ("🔬", "Research",         "Factor analysis, signal, walk-forward, Monte Carlo, time-series"),
            ("⚗️", "Strategy Lab",     "Code editor, backtester, parameter sweep, AI assistant"),
            ("📡", "Kalman Filter",    "TVP regression — dynamic factor betas with uncertainty bands"),
            ("📈", "Performance",      "Backtest tearsheet, risk analytics, factor attribution, Q-Q plot"),
            ("💼", "Portfolio",        "Multi-strategy blending, optimizer, efficient frontier"),
            ("📄", "Paper Trading",    "Live equity curve, P&L bars, win/loss stats, order history"),
            ("📖", "Guide & Glossary", "This page"),
        ]
        for icon, title, desc in nav_items:
            st.markdown(
                f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
                f'border-radius:8px;padding:10px 12px;margin:5px 0;display:flex;gap:10px;align-items:flex-start;">'
                f'<span style="font-size:1.2rem;line-height:1.4;">{icon}</span>'
                f'<div><div style="color:{COLORS["text"]};font-size:0.85rem;font-weight:600;">{title}</div>'
                f'<div style="color:{COLORS["neutral"]};font-size:0.76rem;margin-top:2px;">{desc}</div></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(section_label("Typical Workflow"), unsafe_allow_html=True)
    steps = [
        ("1", COLORS["blue"],     "Launch",         f"Run {cmd('streamlit run app.py')} and open the browser URL."),
        ("2", COLORS["positive"], "Check Health",   "Open Pipeline Health → Status. Verify all components are green."),
        ("3", COLORS["warning"],  "Review Perf.",   "Open Performance → Overview. Check equity curve and KPIs."),
        ("4", COLORS["purple"],   "Inspect Port.",  "Open Portfolio to review allocation, optimizer, and deployment."),
        ("5", COLORS["neutral"],  "Lab / Research", "Use Strategy Lab to edit and backtest; sweep parameters."),
        ("6", COLORS["teal"],     "Deploy",         "Portfolio → Deployment: toggle active strategies and save."),
        ("7", COLORS["orange"],   "Paper Trade",    f"Run {cmd('uv run python orchestration/rebalance.py --broker paper')}"),
        ("8", COLORS["blue"],     "Monitor",        "Open Paper Trading to track live equity curve and orders."),
        ("9", COLORS["teal"],     "Automate",       f"Run {cmd('uv run python orchestration/run_pipeline.py')} daily."),
    ]
    cols = st.columns(3)
    for idx, (num, color, title, desc) in enumerate(steps):
        with cols[idx % 3]:
            st.markdown(
                f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
                f'border-top:3px solid {color};border-radius:2px 2px 8px 8px;'
                f'padding:12px 14px;margin-bottom:8px;min-height:82px;">'
                f'<div style="color:{color};font-size:0.68rem;text-transform:uppercase;'
                f'letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">Step {num}</div>'
                f'<div style="color:{COLORS["text"]};font-size:0.87rem;font-weight:700;margin-bottom:4px;">{title}</div>'
                f'<div style="color:{COLORS["neutral"]};font-size:0.78rem;line-height:1.5;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

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

    st.markdown(section_label("Common Commands"), unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.code("""# Full pipeline
uv run python orchestration/run_pipeline.py

# Signals only
uv run python orchestration/run_pipeline.py --skip-ingest

# Paper rebalance
uv run python orchestration/rebalance.py --broker paper

# Dry run (no orders)
uv run python orchestration/rebalance.py --broker paper --dry-run""", language="bash")
    with col_c2:
        st.code("""# Task Scheduler
python orchestration/setup_scheduler.py       # register
python orchestration/setup_scheduler.py --list  # list
python orchestration/setup_scheduler.py --remove # remove

# Log files
logs/pipeline.log    # master log
logs/ingest.log      # price downloads
logs/signals.log     # signal generation
logs/rebalance.log   # paper execution""", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — METRICS REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ref:
    st.markdown(section_label("Performance Metrics"), unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        for args in [
            ("Total Return",  "(V_n / V_0) − 1",
             "Cumulative % gain or loss over the entire backtest period.",
             "Higher is better. Compare against benchmark on equity curve.", COLORS["positive"]),
            ("CAGR",          "(V_n / V_0)^(1/T) − 1",
             "Compound Annual Growth Rate — smoothed annual return assuming constant compounding.",
             "> 10% for equity-like strategies.", COLORS["teal"]),
            ("Sharpe Ratio",  "μ_ann / σ_ann",
             "Annualised return / annualised volatility. Uses daily returns × √252.",
             "> 1.0 good; > 1.5 excellent; < 0.5 poor risk/reward.", COLORS["blue"]),
            ("Sortino Ratio", "μ_ann / σ_downside",
             "Like Sharpe but penalises downside volatility only — more relevant for asymmetric strategies.",
             "Generally higher than Sharpe. > 1.5 is good.", COLORS["purple"]),
            ("Win Rate",      "# positive days / # total days",
             "Fraction of trading days with positive return.",
             "50–55% typical; above 60% warrants scrutiny.", COLORS["neutral"]),
        ]:
            st.markdown(mdef(*args), unsafe_allow_html=True)

    with col_r:
        for args in [
            ("Ann. Volatility", "σ_daily × √252",
             "Annualised standard deviation of daily returns.",
             "10–18% typical for long-only equity.", COLORS["warning"]),
            ("Max Drawdown",    "min((V_t − peak) / peak)",
             "Worst peak-to-trough decline over entire history.",
             "Deeper than −30% often causes investor redemptions.", COLORS["negative"]),
            ("Calmar Ratio",    "CAGR / |Max DD|",
             "Return per unit of max drawdown risk.",
             "> 0.5 acceptable; > 1.0 strong.", COLORS["orange"]),
            ("VaR 95% / 99%",   "5th / 1st percentile(daily returns)",
             "Daily loss exceeded only 5% / 1% of trading days.",
             "Multiply by √horizon to scale to multi-day VaR.", COLORS["negative"]),
            ("CVaR 95% (ES)",   "E[r | r ≤ VaR 95%]",
             "Average loss on worst 5% of days. Coherent risk measure.",
             "Always worse than VaR; shows tail severity.", COLORS["orange"]),
        ]:
            st.markdown(mdef(*args), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Benchmark-Relative Metrics"), unsafe_allow_html=True)
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown(mdef("Information Ratio",
                         "(port_ret − bench_ret).mean() / TE × √252",
                         "Annualised active return per unit of tracking error. Measures manager skill.",
                         "> 0.5 is good; > 1.0 is excellent.", COLORS["blue"]), unsafe_allow_html=True)
    with col_r2:
        st.markdown(mdef("Tracking Error",
                         "std(port_ret − bench_ret) × √252",
                         "Annualised standard deviation of the active return (portfolio minus benchmark).",
                         "Lower = closer to benchmark. 5–10% typical for factor strategies.", COLORS["teal"]), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Portfolio & Exposure Metrics"), unsafe_allow_html=True)
    col_l3, col_r3 = st.columns(2)
    with col_l3:
        st.markdown(mdef("Gross Exposure", "Σ |w_i|",
                         "Sum of absolute weights. For long-only = 100%. Above 100% = leverage.",
                         "Should equal 100% for unlevered strategies.", COLORS["blue"]), unsafe_allow_html=True)
        st.markdown(mdef("Top-5 Concentration", "Σ top-5 weights",
                         "Sum of five largest position weights. Measures concentration.",
                         "Canary strategy = 100% (5 × 20%). > 80% is a concern for larger universes.", COLORS["warning"]), unsafe_allow_html=True)
    with col_r3:
        st.markdown(mdef("Largest Name", "max(w_i)",
                         "Single largest position weight. Pre-trade limits block positions > 40% NAV by default.",
                         "Canary = 20% (equal weight).", COLORS["orange"]), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Risk Engine Defaults"), unsafe_allow_html=True)
    limits = [
        ("Max Position Size",       "40%",    "Single position weight cap"),
        ("Max Gross Exposure",      "110%",   "Allows slight leverage; blocks > 110%"),
        ("Max Top-5 Concentration", "80%",    "Overridden to 100% for 5-position strategies"),
        ("VaR 95% Limit",           "5%/day", "Pre-trade check fails if current VaR > 5%"),
    ]
    _hc = st.columns([2, 1, 3])
    for lbl, align in [(_hc[0], "Limit"), (_hc[1], "Default"), (_hc[2], "Description")]:
        lbl.markdown(f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;font-weight:700;padding:6px 0;">{align}</div>', unsafe_allow_html=True)
    for name, val, desc in limits:
        _cols = st.columns([2, 1, 3])
        _cols[0].markdown(f'<div style="color:{COLORS["text"]};font-size:0.83rem;padding:5px 0;border-top:1px solid {COLORS["border_dim"]};">{name}</div>', unsafe_allow_html=True)
        _cols[1].markdown(f'<div style="color:{COLORS["negative"]};font-family:monospace;font-size:0.83rem;font-weight:700;padding:5px 0;border-top:1px solid {COLORS["border_dim"]};">{val}</div>', unsafe_allow_html=True)
        _cols[2].markdown(f'<div style="color:{COLORS["neutral"]};font-size:0.82rem;padding:5px 0;border-top:1px solid {COLORS["border_dim"]};">{desc}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Time-Series Analytics Metrics"), unsafe_allow_html=True)
    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        for args in [
            ("Hurst Exponent (H)", "log(R/S) / log(n)  [R/S method]",
             "Measures long-range dependence. H > 0.55 = persistent/trending; "
             "H ~ 0.50 = random walk (Brownian motion); H < 0.45 = mean-reverting.",
             "Momentum factors typically have H > 0.55. Mean-reversion factors H < 0.45.",
             COLORS["teal"]),
            ("Power Spectral Density", "Welch PSD = E[|FFT(x)|^2] / fs",
             "Decomposes a time series into its constituent frequencies. Peaks in the PSD "
             "identify dominant cycles (weekly, monthly, quarterly).",
             "A peak at period T means the series has a recurring pattern every T trading days.",
             COLORS["blue"]),
            ("FFT Filter (Trend/Cycle)", "X_filtered = IFFT(mask(FFT(x)))",
             "Brick-wall frequency filter. Low-pass extracts the slow trend; "
             "high-pass isolates fast cyclical noise.",
             "Cutoff = 20d means signals longer than 20 days are 'trend'; shorter are 'cycle'.",
             COLORS["purple"]),
        ]:
            st.markdown(mdef(*args), unsafe_allow_html=True)
    with col_ts2:
        for args in [
            ("GBM Drift (mu)", "mu = mean(log_ret) * 252 + 0.5*sigma^2",
             "Annualised drift of a Geometric Brownian Motion fitted to the price series. "
             "Includes the Ito correction for the log-price to price transformation.",
             "Positive mu = upward-trending process. Actual drift ≠ expected return due to vol drag.",
             COLORS["positive"]),
            ("GBM Volatility (sigma)", "sigma = std(log_ret) * sqrt(252)",
             "Annualised diffusion coefficient. Determines how wide the GBM fan chart spreads over time.",
             "Larger sigma = wider uncertainty fan and lower terminal median (vol drag effect).",
             COLORS["warning"]),
            ("ACF / ARCH Test", "rho_k = corr(r_t, r_{t-k})",
             "Autocorrelation of returns at lag k. ACF of squared returns tests for ARCH/GARCH effects "
             "(volatility clustering). Bars outside 95% CI (±1.96/sqrt(n)) are statistically significant.",
             "Significant ACF(r^2) = volatility clusters. Larger block size in MC bootstrap recommended.",
             COLORS["neutral"]),
        ]:
            st.markdown(mdef(*args), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Strategy Lab Interface"), unsafe_allow_html=True)
    st.code("""NAME        = "My Strategy"
DESCRIPTION = "One-line description"
DEFAULT_PARAMS = {
    "lookback_years": 6, "top_n": 5,
    "cost_bps": 5.0, "weight_scheme": "equal",
}

def get_signal(features, rebal_dates, top_n, **kwargs):
    ...  # return Polars DataFrame [rebalance_date, symbol, rank/score, selected, equity_pct]

def get_weights(signal, weight_scheme, **kwargs):
    ...  # return Polars DataFrame [rebalance_date, symbol, weight]""", language="python")
    st.markdown(tip("Use <b>✔ Validate</b> before running a backtest — it checks syntax, imports, and the required interface."), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_trouble:
    st.markdown(section_label("Common Issues"), unsafe_allow_html=True)
    issues = [
        ("Health dashboard shows NO DATA",
         "Run the backfill and generate_signals scripts: <br/><code>uv run python orchestration/backfill.py</code><br/><code>uv run python orchestration/generate_signals.py</code>"),
        ("Equity curve won't load",
         "Check that prices are present in <code>data/bronze/equity/daily/</code>. Run the ingest script if needed."),
        ("Pre-trade check fails",
         "Check <code>portfolio_log.parquet</code> for the failure reason. Likely VaR or concentration limit. For the 5-position strategy, ensure <code>RiskLimits(max_top5_concentration=1.0)</code> is passed."),
        ("Task Scheduler tasks not running",
         "Check that the .bat files reference the correct absolute path. Run the .bat file manually to confirm it works before relying on the scheduler."),
        ("Crypto symbol not found (POL/USDT)",
         "Kraken uses MATIC/USDT for Polygon. The universe config maps this correctly."),
        ("Strategy Lab backtest fails",
         "Click ✔ Validate first. Ensure the strategy exports <code>get_signal()</code>, <code>get_weights()</code>, <code>NAME</code>, <code>DESCRIPTION</code>, and <code>DEFAULT_PARAMS</code>."),
        ("Research dashboard Signal Scanner is missing",
         "Signal Scanner has been merged into the Factor Analysis tab (first expandable section)."),
        ("Live Trading page is minimal",
         "Live Trading has been consolidated into the Paper Trading page. Use the mode toggle at the top to switch to Live view."),
    ]
    for problem, solution in issues:
        with st.expander(problem):
            st.markdown(f'<div style="color:{COLORS["neutral"]};font-size:0.83rem;line-height:1.6;">{solution}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
    st.markdown(section_label("IBKR Connection"), unsafe_allow_html=True)
    st.code("""IBKR_HOST      = "127.0.0.1"
IBKR_PORT      = 7497   # paper; 7496 = live
IBKR_CLIENT_ID = 1""", language="python")
    ib_issues = [
        ("Connection refused / timeout",
         "Ensure TWS or IB Gateway is running and 'Allow API connections' is checked in TWS → Configuration → API → Settings."),
        ("Wrong client ID error",
         "Each client connection needs a unique clientId. If another script is connected, increment IBKR_CLIENT_ID."),
        ("Market data frozen / no prices",
         "IB paper accounts require a market data subscription. Check TWS → Account → Market Data."),
        ("Rebalance fails for live broker",
         "Run with <code>--broker paper</code> first to verify the order flow, then switch to <code>--broker ibkr</code>."),
    ]
    for problem, solution in ib_issues:
        with st.expander(problem):
            st.markdown(f'<div style="color:{COLORS["neutral"]};font-size:0.83rem;line-height:1.6;">{solution}</div>', unsafe_allow_html=True)

    st.markdown(warn("Always test with paper trading (port 7497) before switching to live (port 7496)."), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Windows Task Scheduler"), unsafe_allow_html=True)
    st.code("""# Register (run once)
python orchestration/setup_scheduler.py

# Verify
schtasks /Query /TN "QuantPipe\\DailyPipeline" /FO LIST

# Two tasks: DailyPipeline (06:15 Mon-Fri) and DailyRebalance (16:30 Mon-Fri)""", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(section_label("AI Assistant"), unsafe_allow_html=True)
st.markdown(
    f'<div style="color:{COLORS["text_muted"]};font-size:0.80rem;margin:-6px 0 14px;">'
    f'Ask Claude anything about QuantPipe — pipeline setup, strategy code, metrics, or debugging.'
    f'</div>',
    unsafe_allow_html=True,
)

_AI_SYSTEM = """\
You are an expert on the QuantPipe quantitative finance pipeline — a Python system for ETF rotation
and crypto systematic trading. You help users understand: the dashboard pages, metric definitions,
strategy code (get_signal/get_weights interface), pipeline commands, and troubleshooting.

Keep answers concise. Use code blocks when showing commands or Python snippets.
"""

def _ai_api_key():
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY") or None

_api_key = _ai_api_key()
if not _api_key:
    st.info("Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable the AI assistant.")
else:
    try:
        import anthropic as _anthropic

        _chat_key = "guide_ai_chat"
        if _chat_key not in st.session_state:
            st.session_state[_chat_key] = []
        _msgs = st.session_state[_chat_key]

        _hist = st.container()
        with _hist:
            for _msg in _msgs:
                with st.chat_message(_msg["role"]):
                    st.markdown(_msg["content"])

        if _msgs and st.button("Clear", key="guide_ai_clear"):
            st.session_state[_chat_key] = []
            st.rerun()

        _prompt = st.chat_input("Ask about QuantPipe…", key="guide_ai_input")
        if _prompt:
            _msgs.append({"role": "user", "content": _prompt})
            with st.chat_message("user"):
                st.markdown(_prompt)

            _client = _anthropic.Anthropic(api_key=_api_key)
            with st.chat_message("assistant"):
                _ph = st.empty()
                _full = ""
                try:
                    with _client.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=2048,
                        system=_AI_SYSTEM,
                        messages=[{"role": m["role"], "content": m["content"]} for m in _msgs],
                    ) as _stream:
                        for _text in _stream.text_stream:
                            _full += _text
                            _ph.markdown(_full + "▌")
                    _ph.markdown(_full)
                except Exception as _exc:
                    _ph.error(f"API error: {_exc}")
                    _full = ""
            if _full:
                _msgs.append({"role": "assistant", "content": _full})
            st.session_state[_chat_key] = _msgs
    except ImportError:
        st.warning("Install `anthropic` to enable the AI assistant: `pip install anthropic`")
