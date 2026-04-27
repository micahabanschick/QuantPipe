"""Generate README screenshot PNGs from live pipeline data.

Run from the project root on the server:
    uv run python tools/generate_screenshots.py

Outputs PNGs to docs/screenshots/. Requires kaleido (already in pyproject.toml).
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

from config.settings import DATA_DIR
from storage.parquet_store import load_bars

# ── Theme (mirrors _theme.py exactly) ─────────────────────────────────────────

C = {
    "bg":       "#0A0D1A", "bg_void":  "#05070F",
    "surface":  "#0F1325", "card_bg":  "#141C35",
    "gold":     "#C9A227", "gold_dim": "#8A6B18",
    "green":    "#00E676", "green_dim":"#00A854",
    "purple":   "#7B5EA7", "blue":     "#4A90D9",
    "negative": "#FF4D4D", "warning":  "#C9A227",
    "text":     "#EEF2FF", "neutral":  "#A8B3CC",
    "text_muted":"#5A6478","border":   "rgba(201,162,39,0.20)",
    "series": ["#C9A227","#00E676","#7B5EA7","#4A90D9",
               "#FF4D4D","#F5A623","#50E3C2","#BD10E0"],
}
GRID = "rgba(30,38,100,0.45)"
W, H = 1300, 620


def _base_layout(title="", height=H) -> dict:
    return dict(
        paper_bgcolor=C["bg_void"], plot_bgcolor="#0D1128",
        font=dict(color=C["neutral"], family="'Segoe UI', system-ui, sans-serif", size=12),
        title=dict(text=title, font=dict(color=C["text"], size=16), x=0.02, xanchor="left"),
        height=height, margin=dict(l=60, r=40, t=55, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["neutral"], size=11)),
        xaxis=dict(showgrid=True, gridcolor=GRID, linecolor=C["border"],
                   tickfont=dict(color=C["text_muted"], size=11),
                   zerolinecolor=C["border"]),
        yaxis=dict(showgrid=True, gridcolor=GRID, linecolor=C["border"],
                   tickfont=dict(color=C["text_muted"], size=11),
                   zerolinecolor=C["border"]),
    )


def save(fig: go.Figure, name: str, width=W, height=H) -> None:
    path = OUT / f"{name}.png"
    fig.write_image(str(path), width=width, height=height, scale=2)
    print(f"  ✓ {path.name}")


def load_prices(symbols, years=6) -> pl.DataFrame:
    from datetime import date, timedelta
    end   = date.today()
    start = end - timedelta(days=365 * years)
    return load_bars(symbols, start, end, "equity").sort("date")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Performance — equity curve + drawdown
# ═══════════════════════════════════════════════════════════════════════════════

def shot_performance():
    print("performance...")
    syms = ["SPY", "QQQ", "GLD", "AGG"]
    df = load_prices(syms)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna(how="all").ffill())

    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28],
                        shared_xaxes=True, vertical_spacing=0.04)

    colors = [C["green"], C["gold"], C["purple"], C["blue"]]
    for i, sym in enumerate(syms):
        if sym not in wide.columns:
            continue
        s = wide[sym].dropna()
        norm = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, name=sym,
            line=dict(color=colors[i], width=2),
            hovertemplate=f"{sym}: %{{y:.1f}}<extra></extra>",
        ), row=1, col=1)

    # Drawdown of SPY
    spy = wide["SPY"].dropna()
    roll_max = spy.cummax()
    dd = (spy - roll_max) / roll_max * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="SPY DD",
        fill="tozeroy", fillcolor="rgba(255,77,77,0.12)",
        line=dict(color=C["negative"], width=1),
        showlegend=False,
        hovertemplate="DD: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    layout = _base_layout("Performance — Normalised Equity Curves (base=100)")
    layout["yaxis"]["title"] = dict(text="Indexed Value", font=dict(color=C["text_muted"], size=11))
    layout.update(dict(
        yaxis2=dict(showgrid=True, gridcolor=GRID, ticksuffix="%",
                    tickfont=dict(color=C["text_muted"], size=11),
                    linecolor=C["border"],
                    title=dict(text="Drawdown", font=dict(color=C["text_muted"], size=11))),
        xaxis2=dict(showgrid=True, gridcolor=GRID, linecolor=C["border"],
                    tickfont=dict(color=C["text_muted"], size=11)),
    ))
    fig.update_layout(**layout)
    save(fig, "performance")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Pipeline Health — timeline + KPI cards (static composition)
# ═══════════════════════════════════════════════════════════════════════════════

def shot_pipeline_health():
    print("pipeline_health...")
    import json
    from datetime import datetime, timezone, timedelta

    hb_path = ROOT / ".pipeline_heartbeat.json"
    hb = json.loads(hb_path.read_text()) if hb_path.exists() else {}
    last_ts = hb.get("ts_utc", "2026-04-26T17:10:20+00:00")
    last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))

    # Simulate ~5 weeks of daily pipeline events Mon-Fri
    events = []
    t = last_dt
    for _ in range(35):
        if t.weekday() < 5:
            events.append(t)
        t -= timedelta(days=1)
    events = sorted(events)

    steps = ["Equity Ingest", "Crypto Ingest", "Signal Generate", "Pipeline Run"]
    offsets = [0, 2, 5, 0]  # minutes after base time

    fig = go.Figure()
    for i, (step, off) in enumerate(zip(steps, offsets)):
        xs = [e + timedelta(minutes=off) for e in events]
        fig.add_trace(go.Scatter(
            x=xs, y=[step] * len(xs),
            mode="markers",
            marker=dict(symbol="line-ew", size=14, line=dict(width=3, color=C["series"][i])),
            name=step,
            hovertemplate=f"{step}<br>%{{x|%Y-%m-%d %H:%M}}<extra></extra>",
        ))

    layout = _base_layout("Pipeline Health — Run Timeline (Mon–Fri 21:30 UTC)")
    fig.update_layout(**layout)
    fig.update_yaxes(showgrid=False, title=None)
    fig.update_xaxes(title=None)
    save(fig, "pipeline_health")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Data Lab — normalised multi-ETF price chart
# ═══════════════════════════════════════════════════════════════════════════════

def shot_data_lab():
    print("data_lab...")
    syms = ["XLK", "XLE", "XLF", "XLU", "XLI", "XLV"]
    df = load_prices(syms, years=3)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna(how="all").ffill())

    fig = go.Figure()
    for i, sym in enumerate(syms):
        if sym not in wide.columns:
            continue
        s = wide[sym].dropna()
        norm = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, name=sym,
            line=dict(color=C["series"][i], width=1.8),
            hovertemplate=f"{sym}: %{{y:.1f}}<extra></extra>",
        ))

    layout = _base_layout("Data Lab — Sector ETF Performance (Normalised, 3Y)")
    fig.update_layout(**layout)
    save(fig, "data_lab")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Factor Analysis — monthly IC bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def shot_factor_analysis():
    print("factor_analysis...")
    from scipy.stats import spearmanr
    import pandas as pd

    syms = ["SPY","QQQ","GLD","AGG","IWM","XLK","XLE","XLF","XLU","IWN","IWB","TLT"]
    df = load_prices(syms, years=5)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna(how="all").ffill())
    wide.index = pd.to_datetime(wide.index)

    # Compute 12-1 month momentum signal
    ret_12 = wide.pct_change(252)
    ret_1  = wide.pct_change(21)
    signal = ret_12 - ret_1

    # Monthly forward return
    fwd_1m = wide.pct_change(21).shift(-21)

    # Monthly IC
    monthly_ic = {}
    for period, grp in signal.resample("ME"):
        idx = grp.index[-1] if len(grp) else None
        if idx is None or idx not in signal.index or idx not in fwd_1m.index:
            continue
        sig_row = signal.loc[idx].dropna()
        fwd_row = fwd_1m.loc[idx].reindex(sig_row.index).dropna()
        sig_row = sig_row.reindex(fwd_row.index)
        if len(sig_row) < 5:
            continue
        ic, _ = spearmanr(sig_row, fwd_row)
        if not np.isnan(ic):
            monthly_ic[period] = ic

    dates = list(monthly_ic.keys())
    ics   = list(monthly_ic.values())
    colors = [C["green"] if v >= 0 else C["negative"] for v in ics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=ics, marker_color=colors, name="Monthly IC",
        hovertemplate="%{x|%Y-%m}: IC=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=C["border"], width=1))

    mean_ic = float(np.mean(ics))
    fig.add_hline(y=mean_ic, line=dict(color=C["gold"], width=1.5, dash="dot"),
                  annotation_text=f"Mean IC = {mean_ic:.3f}",
                  annotation_font=dict(color=C["gold"], size=11))

    layout = _base_layout("Factor Analysis — Momentum Signal IC by Month")
    layout["yaxis"]["title"] = dict(text="Spearman IC", font=dict(color=C["text_muted"], size=11))
    fig.update_layout(**layout, showlegend=False)
    save(fig, "factor_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Signal Analysis — IC decay curve
# ═══════════════════════════════════════════════════════════════════════════════

def shot_signal_analysis():
    print("signal_analysis...")
    from scipy.stats import spearmanr
    import pandas as pd

    syms = ["SPY","QQQ","GLD","AGG","IWM","XLK","XLE","XLF","XLU","IWN","IWB","TLT"]
    df = load_prices(syms, years=5)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna(how="all").ffill())
    wide.index = pd.to_datetime(wide.index)

    ret_12 = wide.pct_change(252)
    ret_1  = wide.pct_change(21)
    signal = (ret_12 - ret_1).dropna(how="all")

    horizons = [5, 10, 21, 42, 63, 84, 126]
    mean_ics, std_ics = [], []

    for h in horizons:
        fwd = wide.pct_change(h).shift(-h)
        ics = []
        for idx in signal.index[::5]:
            if idx not in fwd.index:
                continue
            sig_row = signal.loc[idx].dropna()
            fwd_row = fwd.loc[idx].reindex(sig_row.index).dropna()
            sig_row = sig_row.reindex(fwd_row.index)
            if len(sig_row) < 5:
                continue
            ic, _ = spearmanr(sig_row, fwd_row)
            if not np.isnan(ic):
                ics.append(ic)
        mean_ics.append(np.mean(ics) if ics else 0)
        std_ics.append(np.std(ics) if ics else 0)

    labels = ["1W","2W","1M","2M","3M","4M","6M"]
    upper = [m + s for m, s in zip(mean_ics, std_ics)]
    lower = [m - s for m, s in zip(mean_ics, std_ics)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(0,230,118,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=mean_ics, name="Mean IC",
        mode="lines+markers",
        line=dict(color=C["green"], width=2.5),
        marker=dict(color=C["green"], size=8, line=dict(color=C["bg_void"], width=2)),
        hovertemplate="Horizon %{x}: IC=%{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=C["border"], width=1))

    layout = _base_layout("Signal Analysis — Momentum IC Decay by Forecast Horizon")
    layout["yaxis"]["title"] = dict(text="Mean Spearman IC ± 1σ", font=dict(color=C["text_muted"], size=11))
    fig.update_layout(**layout)
    save(fig, "signal_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Walk-Forward — OOS equity curves by fold
# ═══════════════════════════════════════════════════════════════════════════════

def shot_walk_forward():
    print("walk_forward...")
    import pandas as pd
    from datetime import date, timedelta

    # Load SPY for a simple IS/OOS walk-forward demo
    df = load_prices(["SPY","QQQ","GLD","IWM","XLK","XLE"], years=6)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna(how="all").ffill())
    wide.index = pd.to_datetime(wide.index)

    # 4 folds: each IS=18mo, OOS=6mo
    oos_curves, fold_dates = [], []
    total_days = len(wide)
    is_days, oos_days = 378, 126  # ~18mo, ~6mo
    step = oos_days

    fig = go.Figure()
    fold_colors = [C["green"], C["gold"], C["purple"], C["blue"]]
    start_idx = is_days

    for fold in range(4):
        end_idx = min(start_idx + oos_days, total_days - 1)
        if end_idx >= total_days:
            break
        oos = wide.iloc[start_idx:end_idx]

        # Simple equal-weight momentum: buy top-3 by 126d return
        is_data = wide.iloc[max(0, start_idx - is_days):start_idx]
        mom = is_data.iloc[-1] / is_data.iloc[max(0, len(is_data)-126)] - 1
        top3 = mom.nlargest(3).index.tolist()
        port_ret = oos[top3].pct_change().fillna(0).mean(axis=1)
        eq = (1 + port_ret).cumprod() * 100

        fig.add_trace(go.Scatter(
            x=oos.index, y=eq.values,
            name=f"Fold {fold+1}",
            line=dict(color=fold_colors[fold], width=2),
            hovertemplate=f"Fold {fold+1}: %{{y:.1f}}<extra></extra>",
        ))
        start_idx += step

    fig.add_hline(y=100, line=dict(color=C["border"], width=1, dash="dot"))
    layout = _base_layout("Walk-Forward Validation — OOS Equity by Fold (base=100)")
    fig.update_layout(**layout)
    save(fig, "walk_forward")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Monte Carlo — fan chart
# ═══════════════════════════════════════════════════════════════════════════════

def shot_monte_carlo():
    print("monte_carlo...")
    import pandas as pd
    from research.monte_carlo import run as mc_run, MCConfig

    df = load_prices(["SPY"], years=5)
    spy = (df.to_pandas()
             .pivot(index="date", columns="symbol", values="adj_close")["SPY"]
             .dropna())
    daily_returns = spy.pct_change().dropna().values

    result = mc_run(daily_returns, MCConfig(n_simulations=2000, seed=42))

    x   = result.fan_x
    p10 = result.fan_p10
    p25 = result.fan_p25
    p50 = result.fan_p50
    p75 = result.fan_p75
    p90 = result.fan_p90

    fig = go.Figure()
    # Fan bands
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(p90) + list(p10)[::-1],
        fill="toself", fillcolor="rgba(0,230,118,0.06)",
        line=dict(width=0), name="10–90th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(p75) + list(p25)[::-1],
        fill="toself", fillcolor="rgba(0,230,118,0.12)",
        line=dict(width=0), name="25–75th pct", hoverinfo="skip",
    ))
    # Sample paths
    for path in result.sample_paths[:40]:
        fig.add_trace(go.Scatter(
            x=x, y=path,
            line=dict(color="rgba(0,230,118,0.07)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Median
    fig.add_trace(go.Scatter(
        x=x, y=p50, name="Median",
        line=dict(color=C["green"], width=2.5),
        hovertemplate="Period %{x}: $%{y:,.0f}<extra>Median</extra>",
    ))
    fig.add_hline(y=result.sample_paths[0][0] if result.sample_paths else 100_000,
                  line=dict(color=C["border"], width=1, dash="dot"))

    layout = _base_layout("Monte Carlo — 500-Path Block-Bootstrap Fan Chart (1Y Forward)")
    layout["xaxis"]["title"] = dict(text="Trading Days", font=dict(color=C["text_muted"], size=11))
    layout["yaxis"]["title"] = dict(text="Cumulative Return", font=dict(color=C["text_muted"], size=11))
    fig.update_layout(**layout)
    save(fig, "monte_carlo")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Time Series — GBM paths + Welch PSD
# ═══════════════════════════════════════════════════════════════════════════════

def shot_time_series():
    print("time_series...")
    import pandas as pd
    from research.spectral import gbm_paths, compute_psd

    df = load_prices(["SPY"], years=5)
    spy = (df.to_pandas()
             .pivot(index="date", columns="symbol", values="adj_close")["SPY"]
             .dropna())
    daily_returns = spy.pct_change().dropna().values
    mu  = float(np.mean(daily_returns)) * 252
    vol = float(np.std(daily_returns))  * np.sqrt(252)

    paths = gbm_paths(S0=float(spy.iloc[-1]), mu_ann=mu, sigma_ann=vol,
                      T_days=252, n_paths=100, seed=42)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["GBM Price Simulation (100 paths, 1Y)",
                                        "Welch Power Spectral Density — SPY Returns"])

    # GBM paths
    t = np.linspace(0, 1, paths.shape[0])
    for i in range(paths.shape[1]):
        fig.add_trace(go.Scatter(
            x=t, y=paths[:, i],
            line=dict(color=f"rgba(0,230,118,{0.12 if i < 99 else 0.9})", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # Welch PSD
    freqs, psd = compute_psd(daily_returns, fs=252)
    fig.add_trace(go.Scatter(
        x=freqs[1:], y=10 * np.log10(psd[1:]),
        line=dict(color=C["gold"], width=2), name="PSD",
        hovertemplate="%.2f Hz: %{y:.1f} dB<extra></extra>",
    ), row=1, col=2)

    layout = _base_layout("Time Series Analytics")
    layout.update(height=H, showlegend=False)
    fig.update_layout(**layout)
    for ax in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        fig.update_layout(**{ax: dict(
            showgrid=True, gridcolor=GRID, linecolor=C["border"],
            tickfont=dict(color=C["text_muted"], size=10),
        )})
    fig.update_layout(
        xaxis=dict(title=dict(text="Time (years)", font=dict(color=C["text_muted"], size=11))),
        xaxis2=dict(title=dict(text="Frequency (cycles/year)", font=dict(color=C["text_muted"], size=11))),
        yaxis=dict(title=dict(text="Price ($)", font=dict(color=C["text_muted"], size=11))),
        yaxis2=dict(title=dict(text="Power (dB)", font=dict(color=C["text_muted"], size=11))),
    )
    for ann in fig.layout.annotations:
        ann.font.color = C["neutral"]
        ann.font.size  = 13
    save(fig, "time_series")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Kalman Filter — dynamic beta vs static OLS
# ═══════════════════════════════════════════════════════════════════════════════

def shot_kalman():
    print("kalman...")
    import pandas as pd
    from research.kalman_filter import kalman_smooth_betas

    syms = ["QQQ", "SPY"]
    df = load_prices(syms, years=5)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna().ffill())
    wide.index = pd.to_datetime(wide.index)

    rets = wide.pct_change().dropna()
    # kalman_smooth_betas expects pd.Series and pd.DataFrame
    result = kalman_smooth_betas(rets["QQQ"], rets[["SPY"]])

    if result.filtered_betas.size == 0:
        print("  (kalman returned empty — skipping)")
        return

    # filtered_betas: (T, K) — col 0=alpha, col 1=SPY beta
    betas = result.filtered_betas[:, 1]
    # ±1σ CI from diagonal of posterior covariance
    std   = np.sqrt(result.filtered_vars[:, 1, 1])
    ci_lo = betas - std
    ci_hi = betas + std
    dates = pd.to_datetime(result.dates)

    # Static OLS beta
    from numpy.linalg import lstsq
    y_arr = rets["QQQ"].values
    X_arr = rets[["SPY"]].values
    ols_beta = float(lstsq(np.column_stack([np.ones(len(X_arr)), X_arr]), y_arr, rcond=None)[0][1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates.tolist() + dates.tolist()[::-1],
        y=ci_hi.tolist() + ci_lo.tolist()[::-1],
        fill="toself", fillcolor="rgba(201,162,39,0.10)",
        line=dict(width=0), name="±1σ CI", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=betas, name="Kalman β (QQQ ~ SPY)",
        line=dict(color=C["gold"], width=2.5),
        hovertemplate="β = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=ols_beta, line=dict(color=C["purple"], width=1.5, dash="dot"),
                  annotation_text=f"OLS β = {ols_beta:.3f}",
                  annotation_font=dict(color=C["purple"], size=11))

    layout = _base_layout("Kalman Filter — Time-Varying Beta: QQQ ~ SPY")
    layout["yaxis"]["title"] = dict(text="Beta", font=dict(color=C["text_muted"], size=11))
    fig.update_layout(**layout)
    save(fig, "kalman")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Strategy Lab — backtest equity curve
# ═══════════════════════════════════════════════════════════════════════════════

def shot_strategy_lab():
    print("strategy_lab...")
    import pandas as pd

    syms = ["SPY","QQQ","GLD","AGG","IWM","XLK","XLE","XLF","XLU","IWN","IWB","IWS","TLT","DIA"]
    df = load_prices(syms, years=6)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna(how="all").ffill())
    wide.index = pd.to_datetime(wide.index)

    # Simple monthly momentum_top5: hold top-5 by 12-1 month momentum, equal weight
    monthly = wide.resample("ME").last().dropna(how="all")
    nav = 100.0
    nav_series = {}
    current_syms = []

    for i in range(12, len(monthly)):
        period_end   = monthly.index[i]
        period_start = monthly.index[i - 1]

        # Compute signal on available data
        ret_12 = monthly.iloc[i-1] / monthly.iloc[max(0, i-12)] - 1
        ret_1  = monthly.iloc[i-1] / monthly.iloc[i-2] - 1
        signal = (ret_12 - ret_1).dropna()
        top5   = signal.nlargest(5).index.tolist()

        # Monthly return for previous top5
        if current_syms:
            valid = [s for s in current_syms if s in wide.columns]
            if valid:
                sub = wide.loc[period_start:period_end, valid]
                if len(sub) > 1:
                    port_ret = sub.pct_change().fillna(0).mean(axis=1)
                    nav *= float((1 + port_ret).prod())

        nav_series[period_end] = nav
        current_syms = top5

    # SPY benchmark
    spy_m = monthly["SPY"].dropna()
    spy_m = spy_m / spy_m.iloc[12] * 100

    dates = list(nav_series.keys())
    navs  = list(nav_series.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=navs, name="Momentum Top-5",
        line=dict(color=C["green"], width=2.5),
        hovertemplate="Strategy: $%{y:.1f}<extra></extra>",
    ))
    spy_aligned = spy_m.reindex(dates, method="nearest")
    fig.add_trace(go.Scatter(
        x=dates, y=spy_aligned.values, name="SPY",
        line=dict(color=C["neutral"], width=1.5, dash="dot"),
        hovertemplate="SPY: $%{y:.1f}<extra></extra>",
    ))

    layout = _base_layout("Strategy Lab — Momentum Top-5 Backtest vs SPY (base=100)")
    layout["yaxis"]["tickprefix"] = "$"
    fig.update_layout(**layout)
    save(fig, "strategy_lab")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Portfolio Management — efficient frontier + allocation pie
# ═══════════════════════════════════════════════════════════════════════════════

def shot_portfolio_mgmt():
    print("portfolio_mgmt...")
    import pandas as pd

    syms = ["SPY","QQQ","GLD","AGG","IWM","XLK","XLE","XLF","XLU","IWN"]
    df = load_prices(syms, years=3)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna().ffill())
    daily_ret = wide.pct_change().dropna()
    mu  = daily_ret.mean() * 252
    cov = daily_ret.cov()  * 252

    # Random portfolios
    rng = np.random.default_rng(42)
    n_pts = 3000
    port_ret, port_vol, port_sharpe = [], [], []
    for _ in range(n_pts):
        w = rng.dirichlet(np.ones(len(syms)))
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ cov.values @ w))
        port_ret.append(r)
        port_vol.append(v)
        port_sharpe.append(r / v if v > 0 else 0)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Efficient Frontier (3,000 Random Portfolios)",
                                        "Current Allocation"],
                        specs=[[{"type": "xy"}, {"type": "domain"}]])

    # Frontier scatter
    fig.add_trace(go.Scatter(
        x=port_vol, y=port_ret, mode="markers",
        marker=dict(color=port_sharpe, colorscale=[
            [0.0, C["purple"]], [0.5, C["gold"]], [1.0, C["green"]],
        ], size=3, opacity=0.6,
        colorbar=dict(title=dict(text="Sharpe", font=dict(color=C["neutral"], size=11)),
                      thickness=12,
                      tickfont=dict(color=C["neutral"], size=10))),
        name="Portfolio", hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
    ), row=1, col=1)

    # Allocation pie from target_weights
    tw_path = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
    if tw_path.exists():
        tw = pl.read_parquet(tw_path)
        latest = tw.filter(pl.col("date") == tw["date"].max())
        labels  = latest["symbol"].to_list()
        weights = latest["weight"].to_list()
    else:
        labels  = syms[:6]
        weights = [1/6] * 6

    fig.add_trace(go.Pie(
        labels=labels, values=weights,
        marker=dict(colors=C["series"][:len(labels)],
                    line=dict(color=C["bg_void"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color=C["text"]),
        hole=0.45,
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ), row=1, col=2)

    layout = _base_layout("Portfolio Management")
    layout.update(height=H, showlegend=False)
    fig.update_layout(**layout)
    for ax in ["xaxis", "yaxis"]:
        fig.update_layout(**{ax: dict(
            showgrid=True, gridcolor=GRID, linecolor=C["border"],
            tickfont=dict(color=C["text_muted"], size=10),
        )})
    fig.update_layout(
        xaxis=dict(title=dict(text="Volatility (Ann.)", font=dict(color=C["text_muted"], size=11)), tickformat=".0%"),
        yaxis=dict(title=dict(text="Return (Ann.)", font=dict(color=C["text_muted"], size=11)), tickformat=".0%"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = C["neutral"]
        ann.font.size  = 13
    save(fig, "portfolio_mgmt")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Paper Trading — NAV history + position pie
# ═══════════════════════════════════════════════════════════════════════════════

def shot_paper_trading():
    print("paper_trading...")
    import pandas as pd

    tw_path = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
    th_path = DATA_DIR / "gold" / "equity" / "trading_history.parquet"

    # Build equity curve from target_weights × prices
    tw = pl.read_parquet(tw_path)
    latest_tw = tw.filter(pl.col("date") == tw["date"].max())
    all_syms  = tw["symbol"].unique().to_list()

    df = load_prices(all_syms, years=1)
    wide = (df.to_pandas()
              .pivot(index="date", columns="symbol", values="adj_close")
              .dropna(how="all").ffill())
    wide.index = pd.to_datetime(wide.index)

    # Use trading_history anchors
    th = pl.read_parquet(th_path).filter(pl.col("broker") == "paper") if th_path.exists() else None

    nav = 1_000_000.0
    nav_series = {}
    latest_weights = dict(zip(latest_tw["symbol"].to_list(), latest_tw["weight"].to_list()))
    prev_ts = None

    for ts in wide.index:
        if prev_ts is None:
            nav_series[ts] = nav
            prev_ts = ts
            continue
        port_ret = sum(
            w * (wide.at[ts, sym] / wide.at[prev_ts, sym] - 1)
            for sym, w in latest_weights.items()
            if sym in wide.columns and not np.isnan(wide.at[ts, sym]) and not np.isnan(wide.at[prev_ts, sym])
        )
        nav *= (1 + port_ret)
        nav_series[ts] = nav
        prev_ts = ts

    # Anchor at trading_history if available
    if th is not None and len(th) > 0:
        for row in th.iter_rows(named=True):
            d = pd.Timestamp(row["date"])
            if d in nav_series:
                nav_series[d] = row["nav"]

    dates = list(nav_series.keys())
    navs  = list(nav_series.values())

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Paper Account NAV ($)", "Current Positions"],
                        specs=[[{"type": "xy"}, {"type": "domain"}]])

    fig.add_trace(go.Scatter(
        x=dates, y=navs, name="NAV",
        line=dict(color=C["green"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.06)",
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=list(latest_weights.keys()),
        values=list(latest_weights.values()),
        marker=dict(colors=C["series"][:len(latest_weights)],
                    line=dict(color=C["bg_void"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color=C["text"]),
        hole=0.45,
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ), row=1, col=2)

    layout = _base_layout("Paper Trading — Account NAV & Positions (DUQ368627)")
    layout.update(height=H, showlegend=False)
    fig.update_layout(**layout)
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor=GRID, linecolor=C["border"],
                   tickfont=dict(color=C["text_muted"], size=10)),
        yaxis=dict(showgrid=True, gridcolor=GRID, linecolor=C["border"],
                   tickfont=dict(color=C["text_muted"], size=10),
                   tickprefix="$", tickformat=",.0f"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = C["neutral"]
        ann.font.size  = 13
    save(fig, "paper_trading")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Deployment — strategy allocation bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def shot_deployment():
    print("deployment...")
    import json

    config_path = DATA_DIR / "gold" / "equity" / "deployment_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        strats = cfg.get("strategies", [])
    else:
        strats = []

    if not strats:
        # Fallback: show all 9 strategies with equal weight
        names = [
            "Momentum Top-5", "Aggressive Concentrated Momentum",
            "ETF Pairs Relative Momentum", "MASE",
            "Regime-Adaptive Dynamic Allocation", "Sector Rotation Momentum",
            "Shium Optimal Momentum", "Tactical Defense Momentum",
            "Volatility-Scaled Sector Momentum",
        ]
        weights = [1/9] * 9
    else:
        names   = [s.get("name", s.get("slug", "?")) for s in strats]
        weights = [s.get("allocation_weight", 0) for s in strats]

    fig = go.Figure(go.Bar(
        x=weights, y=names, orientation="h",
        marker=dict(color=C["series"][:len(names)],
                    line=dict(color=C["bg_void"], width=1)),
        text=[f"{w*100:.1f}%" for w in weights],
        textposition="outside",
        textfont=dict(color=C["neutral"], size=11),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))

    layout = _base_layout("Deployment — Active Strategy Allocation", height=460)
    layout["xaxis"]["tickformat"] = ".0%"
    layout["yaxis"]["showgrid"]   = False
    layout["yaxis"]["tickfont"]   = dict(color=C["text"], size=12)
    layout["margin"] = dict(l=240, r=80, t=55, b=40)
    fig.update_layout(**layout)
    save(fig, "deployment", height=460)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Generating screenshots → {OUT}\n")
    shot_performance()
    shot_pipeline_health()
    shot_data_lab()
    shot_factor_analysis()
    shot_signal_analysis()
    shot_walk_forward()
    shot_monte_carlo()
    shot_time_series()
    shot_kalman()
    shot_strategy_lab()
    shot_portfolio_mgmt()
    shot_paper_trading()
    shot_deployment()
    print(f"\nDone — {len(list(OUT.glob('*.png')))} screenshots in {OUT}")
