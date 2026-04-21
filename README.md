# QuantPipe

An end-to-end quantitative finance pipeline for systematic equity and crypto trading. Built around a **Sector × Size × Style ETF rotation** framework, with a clear path from free data and paper trading to live deployment.

**Stack:** Python 3.13 · Polars · DuckDB · Parquet · VectorBT · cvxpy · PyPortfolioOpt · Streamlit · streamlit-ace · reportlab · kaleido · IBKR · CCXT · `uv`

**Status:** Phases 0–6 complete. Pipeline running with live data. 145 tests passing.

**Live results (canary, 6-year backtest):** Sharpe 1.036 · CAGR 17.1% · Max DD −14.3%

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
  - [config](#config)
  - [data\_adapters](#data_adapters)
  - [storage](#storage)
  - [features](#features)
  - [signals](#signals)
  - [backtest](#backtest)
  - [portfolio](#portfolio)
  - [risk](#risk)
  - [execution](#execution)
  - [orchestration](#orchestration)
  - [reports](#reports)
  - [strategies](#strategies)
  - [tools](#tools)
  - [research](#research)
  - [tests](#tests)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Build Phases](#build-phases)
- [Design Principles](#design-principles)
- [Reference Documents](#reference-documents)

---

## Architecture Overview

Data flows in one direction through clearly separated layers. No module reaches backwards.

```
[Vendors]
    │  yfinance (equities/ETFs)
    │  CCXT (crypto)
    │  IBKR (live/paper)
    ▼
[data_adapters]   ← unified DataAdapter protocol
    ▼
[storage]         ← Parquet on disk, DuckDB queries, bronze/gold layers
    ▼
[features]        ← pure functions, point-in-time safe, snapshot-tested
    ▼
[signals]         ← cross-sectional rankings, walk-forward validated
    ▼
[backtest]        ← VectorBT integration, cost models, tearsheets
    ▼
[portfolio]       ← signals + covariance → target weights (cvxpy / PyPortfolioOpt)
    ▼
[risk]            ← pre-trade checks, VaR, exposure limits (hard blocks)
    ▼
[execution]       ← BrokerAdapter protocol → IBKR, CCXT, PaperBroker
    ▼
[reports]         ← Streamlit dashboards (ops health + P&L performance)
    │
[orchestration]   ← cron-driven pipeline, alerting via Pushover/ntfy
```

**Key invariants:**
- The data layer never talks to execution.
- Every feature is a pure function of data available on or before date T — no lookahead.
- `DataAdapter` and `BrokerAdapter` are protocols; swapping providers requires no downstream changes.
- Paper and live trading share 100% of the strategy and execution code path.
- Pre-trade checks are hard blocks, never advisory warnings.

---

## Repository Structure

```
QuantPipe/
│
├── config/
│   ├── universes.py            9-box ETFs, GICS sectors, crypto universe
│   └── settings.py             Paths, API keys, pipeline defaults (loaded from .env)
│
├── data_adapters/
│   ├── base.py                 DataAdapter protocol + OHLCV schema
│   ├── yfinance_adapter.py     Yahoo Finance (equities, ETFs) — free
│   └── ccxt_adapter.py         CCXT unified interface (crypto, any exchange)
│
├── storage/
│   ├── parquet_store.py        write_bars(), load_bars(), list_symbols()
│   ├── validators.py           Post-ingestion data quality checks
│   ├── adjustments.py          Corporate actions (splits, dividends) storage
│   └── universe.py             universe_as_of_date() — survivorship-bias-safe lookups
│
├── features/
│   ├── canonical.py            5 pure factor functions (momentum, vol, reversal, …)
│   └── compute.py              compute_and_store(), load_features() with gold-layer cache
│
├── signals/
│   └── momentum.py             cross_sectional_momentum(), momentum_weights(),
│                               get_monthly_rebalance_dates()
│
├── backtest/
│   ├── engine.py               run_backtest() — VectorBT wrapper, cost model
│   ├── tearsheet.py            print_tearsheet(), tearsheet_dict()
│   ├── walk_forward.py         walk_forward() — expanding-window OOS validation
│   └── canary.py               Canary strategy: 12-1 momentum top-5, monthly rebalance
│
├── portfolio/
│   ├── covariance.py           ledoit_wolf_cov(), sample_cov(), compute_returns()
│   └── optimizer.py            construct_portfolio() — 5 methods (equal, vol_scaled,
│                               mean_variance, min_variance, max_sharpe)
│
├── risk/
│   ├── engine.py               compute_exposures(), historical_var(), pre_trade_check(),
│   │                           generate_risk_report(), print_risk_report()
│   └── scenarios.py            4 stress scenarios (2008 GFC, 2020 COVID, 2022 rates,
│                               2000 dot-com) + run_all_scenarios()
│
├── execution/
│   ├── base.py                 BrokerAdapter protocol, Order, Fill, Position types
│   ├── paper_broker.py         In-memory paper broker — fills at mark price
│   ├── ibkr_adapter.py         IBKRAdapter via ib_insync (paper port 7497)
│   ├── ccxt_broker.py          CCXTBroker — live + paper crypto execution
│   ├── trader.py               compute_orders() — pure, idempotent weight → order calc
│   └── reconciler.py           reconcile(), has_material_drift(), write_reconcile_log()
│
├── orchestration/
│   ├── backfill_history.py     One-shot historical seed (run once)
│   ├── ingest_daily.py         Nightly incremental ingestion
│   ├── generate_signals.py     Daily signal generation + risk snapshot persistence
│   ├── rebalance.py            Daily rebalance: load weights → check → orders → reconcile
│   └── run_pipeline.py         Master cron chain: ingest → signals → alert on failure
│
├── strategies/
│   ├── __init__.py
│   └── momentum_top5.py        Cross-sectional 12-1 momentum, equal-weight top-5 (template)
│
├── tools/
│   ├── __init__.py
│   └── backtest_runner.py      Subprocess backtest runner — loads any strategy, emits JSON
│
├── reports/
│   ├── health_dashboard.py     Dashboard #1: ops health, ingestion status, log viewer
│   ├── performance_dashboard.py Dashboard #2: equity curve, drawdown, Sharpe, exposures, PDF export
│   ├── strategy_lab.py         Dashboard #3: in-browser strategy editor + backtest runner
│   └── pdf_export.py           ReportLab PDF builder — charts via kaleido, no Streamlit deps
│
├── research/                   Jupyter notebooks (exploration only, never production)
│
├── tests/
│   ├── test_adapters.py        Adapter schema + date range (hits network)
│   ├── test_storage.py         Parquet roundtrip, idempotency, list_symbols
│   ├── test_features.py        Per-feature correctness + snapshot tests
│   ├── test_signals.py         Momentum signal ranking, weights, rebalance dates
│   ├── test_portfolio.py       Covariance estimation + all 5 optimizer methods
│   ├── test_risk.py            Exposures, VaR, pre-trade checks, stress scenarios
│   ├── test_orchestration.py   Signal generation helpers, pipeline step sequencing
│   └── test_execution.py       Trader, PaperBroker, reconciler
│
├── .env.example                All required environment variables documented
├── pyproject.toml              Dependencies and tool config (uv-compatible)
└── quantfinance_*.md           Reference design documents
```

---

## Modules

### `config`

**`universes.py`** — defines the three trading universes used throughout the pipeline:

| Constant | Contents |
|---|---|
| `STYLE_SIZE_9BOX` | 9 iShares Russell ETFs — Large/Mid/Small × Growth/Blend/Value |
| `SECTOR_SPDRS` | 11 GICS sector SPDR ETFs (XLK, XLV, XLF, …) |
| `BENCHMARKS` | SPY, QQQ, AGG, TLT, GLD, DIA |
| `EQUITY_UNIVERSE` | Union of all above — 26 ETFs total |
| `CRYPTO_UNIVERSE` | BTC, ETH, SOL, AVAX, LINK, ADA, DOT, POL, BNB, XRP (USDT pairs) |

**`settings.py`** — loads all configuration from `.env`. Copy `.env.example` to `.env` and fill in your keys. Never commit `.env`.

---

### `data_adapters`

All adapters implement the `DataAdapter` protocol defined in `base.py`.

```python
from data_adapters import YFinanceAdapter, CCXTAdapter
from datetime import date

adapter = YFinanceAdapter()
df = adapter.get_bars("SPY", date(2020, 1, 1), date(2024, 12, 31))
# Returns Polars DataFrame: date, symbol, open, high, low, close, volume, adj_close
```

| Adapter | Asset class | Cost | Notes |
|---|---|---|---|
| `YFinanceAdapter` | Equities, ETFs | Free | Auto-adjusted + raw close stored |
| `CCXTAdapter` (data) | Crypto OHLCV | Free | Kraken default, pagination handled |
| `IBKRAdapter` | All | Requires account | Phase 6 — live/paper execution |
| `CCXTBroker` | Crypto | Free / keys | Phase 6 — order routing via CCXT |

---

### `storage`

Parquet files on disk, queryable via DuckDB. No server required.

**Layout:**
```
data/
├── bronze/
│   ├── equity/daily/symbol=SPY/year=2024/data.parquet
│   └── crypto/daily/symbol=BTC_USDT/year=2024/data.parquet
└── gold/
    ├── equity/features/                 Pre-computed feature Parquet
    ├── equity/target_weights.parquet    Latest rebalance weights (upserted daily)
    ├── equity/portfolio_log.parquet     Daily risk snapshots (VaR, exposures)
    └── equity/reconcile_log.parquet     Broker reconciliation history
```

```python
from storage import write_bars, load_bars, list_symbols

write_bars(df, asset_class="equity", symbol="SPY")  # idempotent upsert
df = load_bars(["SPY", "QQQ"], start=date(2020,1,1), end=date(2024,12,31))
symbols = list_symbols("equity")
```

**Validators** (`storage/validators.py`):

| Check | Catches |
|---|---|
| `check_row_counts` | Missing trading days |
| `check_null_rate` | Null rate > 1% in price columns |
| `check_price_jumps` | Daily moves > 8σ |
| `check_staleness` | Last bar older than 3 trading days |

---

### `features`

Five canonical factors in `features/canonical.py`. All backward-looking — no lookahead possible by construction.

| Function | Description |
|---|---|
| `log_return(prices, periods=1)` | Log return over N periods |
| `realized_vol(prices, window=21)` | Rolling annualised volatility |
| `momentum_12m_1m(prices)` | 12-month return skipping most recent month (Jegadeesh & Titman) |
| `dollar_volume(close, volume, window=63)` | Rolling average dollar volume — liquidity filter |
| `reversal_5d(prices)` | Negative 5-day return — short-term mean reversion signal |

Snapshot tests in `tests/test_features.py` pin output to a fixed fixture. Any accidental lookahead contamination fails immediately.

---

### `signals`

Cross-sectional momentum signal pipeline in `signals/momentum.py`.

```python
from signals.momentum import cross_sectional_momentum, momentum_weights, get_monthly_rebalance_dates

# Rank universe by 12-1 momentum at each rebalance date, select top 5
signal = cross_sectional_momentum(features_df, rebalance_dates, top_n=5)

# Convert signal to equal-weight or vol-scaled weights
weights = momentum_weights(signal, weight_scheme="equal")
weights = momentum_weights(signal, weight_scheme="vol_scaled", vol_series=vol_df)
```

---

### `backtest`

VectorBT integration with a shared cash, group-by portfolio simulation.

```python
from backtest.engine import run_backtest
from backtest.tearsheet import print_tearsheet

result = run_backtest(prices, weights, cost_bps=5.0, initial_cash=100_000)
print_tearsheet(result, title="Canary Strategy")
# result.sharpe, result.cagr, result.max_drawdown, result.equity_curve
```

**Canary validation:** `uv run python backtest/canary.py` runs the full 12-1 momentum pipeline and prints a tearsheet + risk report. Expected Sharpe: 0.5–1.5. Anything outside that range indicates a pipeline bug.

**Walk-forward validation:**

```python
from backtest.walk_forward import walk_forward

result = walk_forward(prices, features, signal_fn, weight_fn, train_years=3, test_months=12)
print(result.combined_sharpe)   # stitched OOS Sharpe
```

---

### `portfolio`

Pure function interface — no I/O, no state.

```python
from portfolio import ledoit_wolf_cov, construct_portfolio, PortfolioConstraints

cov, symbols = ledoit_wolf_cov(prices_df, lookback_days=252)
weights = construct_portfolio(
    signals,
    cov_matrix=cov,
    symbols=symbols,
    method="vol_scaled",          # or "equal", "mean_variance", "min_variance", "max_sharpe"
    constraints=PortfolioConstraints(max_position=0.40),
)
```

| Method | Description | Requires |
|---|---|---|
| `equal` | 1/N equal weight | — |
| `vol_scaled` | Inverse-volatility weight (1/σᵢ normalised) | cov matrix |
| `mean_variance` | Max μᵀw − (λ/2)wᵀΣw via cvxpy | cov + expected returns |
| `min_variance` | Minimum variance via PyPortfolioOpt | cov matrix |
| `max_sharpe` | Maximum Sharpe ratio via PyPortfolioOpt | cov + expected returns |

---

### `risk`

Pre-trade checks are **hard blocks** — `CheckResult(passed=False)` must prevent order submission. Never downgrade to a warning.

```python
from risk import pre_trade_check, generate_risk_report, print_risk_report, RiskLimits, run_all_scenarios

check = pre_trade_check(proposed_weights, RiskLimits(max_position=0.40, var_limit_pct=0.025))
if not check.passed:
    raise RuntimeError(f"Pre-trade check failed: {check.violations}")

report = generate_risk_report(weights, prices_df, stress_results=run_all_scenarios(weights))
print_risk_report(report)
```

**Default `RiskLimits`:**

| Limit | Default |
|---|---|
| `max_position` | 40% in any single name |
| `max_sector` | 60% in any one sector |
| `max_gross` | 100% gross exposure |
| `max_net` | 100% net exposure |
| `max_top5_concentration` | 80% in top-5 names |
| `var_limit_pct` | None (uncapped by default) |

**Stress scenarios** in `risk/scenarios.py`: `2008_GFC`, `2020_COVID`, `2022_RATES`, `2000_DOTCOM`.

---

### `execution`

All broker adapters implement `BrokerAdapter` from `execution/base.py` — swap providers without touching downstream code.

**Order computation (pure function):**

```python
from execution import compute_orders, nav_from_positions

nav = nav_from_positions(positions, cash, prices)
orders = compute_orders(target_weights, positions, prices, nav, min_trade_pct=0.005)
# Idempotent: same inputs always produce the same orders
# Returns integer share quantities, skips dust trades
```

**Paper broker:**

```python
from execution import PaperBroker, Order

broker = PaperBroker(initial_cash=100_000)
broker.set_prices({"SPY": 520.0, "QQQ": 445.0})
order_id = broker.place_order(Order(symbol="SPY", qty=10))
```

**IBKR (paper/live):**

```python
from execution.ibkr_adapter import IBKRAdapter

with IBKRAdapter(host="127.0.0.1", port=7497, is_paper=True) as broker:
    positions = broker.get_positions()
    orders = compute_orders(target_weights, positions, prices, nav)
    for order in orders:
        broker.place_order(order)
```

**Reconciler:**

```python
from execution import reconcile, has_material_drift, format_reconcile_report

report = reconcile(internal_positions, broker.get_positions(), prices, nav)
print(format_reconcile_report(report))
if has_material_drift(report, threshold_pct=5.0):
    raise RuntimeError("Material drift detected — investigate before next rebalance")
```

---

### `orchestration`

**Full daily pipeline:**

```bash
# Master chain: ingest → validate → features → signals → alert on failure
uv run python orchestration/run_pipeline.py

# Skip ingestion and regenerate signals only
uv run python orchestration/run_pipeline.py --skip-ingest

# Execute rebalance against paper broker
uv run python orchestration/rebalance.py --broker paper

# Dry-run: compute orders without placing them
uv run python orchestration/rebalance.py --broker paper --dry-run

# Live IBKR rebalance (requires TWS/Gateway running)
uv run python orchestration/rebalance.py --broker ibkr
```

**Windows Task Scheduler setup (run once):**

```bash
# Registers two tasks: DailyPipeline at 06:15 and DailyRebalance at 16:30, Mon–Fri
uv run python orchestration/setup_scheduler.py

# Verify
schtasks /Query /TN "QuantPipe\DailyPipeline" /FO LIST
schtasks /Query /TN "QuantPipe\DailyRebalance" /FO LIST

# Remove
uv run python orchestration/setup_scheduler.py --remove
```

The scheduler calls `orchestration/run_pipeline.bat` and `orchestration/run_rebalance.bat`,
which are thin wrappers that `cd` to the project directory and redirect output to `logs/`.

**Individual steps:**

```bash
uv run python orchestration/backfill_history.py          # one-shot historical seed
uv run python orchestration/ingest_daily.py              # incremental ingest
uv run python orchestration/generate_signals.py          # signal generation + risk snapshot
uv run python features/compute.py                        # recompute gold-layer features
```

Exit codes: `0` = success, `1` = partial failure, `2` = total failure. Pushover/ntfy alert sent on any non-zero exit.

---

### `reports`

All three dashboards are wired together in `app.py` and launched via a single command:

```bash
streamlit run app.py
```

**Dashboard #1 — Pipeline Health** (`reports/health_dashboard.py`):

Shows: last ingestion time per asset class, signal freshness, universe sizes, per-symbol row counts with staleness flags, portfolio snapshot metrics (VaR, gross exposure, pre-trade status), tabbed log viewers for pipeline/ingest/signals logs.

**Dashboard #2 — Performance** (`reports/performance_dashboard.py`):

Shows: equity curve with benchmark overlay, trailing returns bar, rolling Sharpe/Sortino, monthly returns heatmap, return distribution with VaR markers, current portfolio positions + sector breakdown, stress scenario bars, top drawdown table.

Download bar at the top of the page:
- **📄 PDF Report** — full multi-section report (executive summary, performance charts, portfolio, risk, analytics) as a print-ready PDF generated with ReportLab + kaleido.
- **📊 Trade History** — every backtest transaction (entry, exit, size, P&L) as CSV.

**Dashboard #3 — Strategy Lab** (`reports/strategy_lab.py`):

In-browser strategy development environment backed by the `strategies/` folder.

- Strategy selector dropdown auto-discovers all `strategies/*.py` files.
- **➕ New Strategy** button scaffolds a new file from the standard template.
- Ace editor (via `streamlit-ace`) for in-browser Python editing with syntax highlighting.
- Save / Discard / Reload action bar with per-file dirty state tracking.
- Backtest config panel (lookback, top-N, cost bps, weight scheme) pre-filled from `DEFAULT_PARAMS`.
- **▶ Run Backtest** fires `tools/backtest_runner.py` as a subprocess, streams progress, and displays full tearsheet results in-page.

### `strategies`

User-defined strategy files. Each file must expose:

| Attribute | Type | Description |
|---|---|---|
| `NAME` | `str` | Display name in Strategy Lab selector |
| `DESCRIPTION` | `str` | One-line summary |
| `DEFAULT_PARAMS` | `dict` | Fallback values for `lookback_years`, `top_n`, `cost_bps`, `weight_scheme` |
| `get_signal(features, rebal_dates, **kwargs)` | function | Returns signal `pl.DataFrame` |
| `get_weights(signal, **kwargs)` | function | Returns weights `pl.DataFrame` |

New strategies can be created from the Strategy Lab UI or by copying `strategies/momentum_top5.py`.

### `tools`

**`backtest_runner.py`** — invoked as a subprocess by Strategy Lab.

```bash
uv run python tools/backtest_runner.py \
    --strategy strategies/momentum_top5.py \
    --lookback-years 6 \
    --top-n 5 \
    --cost-bps 5.0 \
    --weight-scheme equal
```

Emits a single JSON payload to stdout:

```json
{ "ok": true, "strategy_name": "...", "metrics": {...}, "equity": {...}, "benchmark": {...}, "params": {...} }
```

Progress lines stream to stderr for display in the Strategy Lab console expander.

---

### `research`

Jupyter notebooks for exploratory work. Naming convention: `YYYY-MM-DD_short-description.ipynb`.

Rules:
- Notebooks are exploration only — logic that matters migrates to a module.
- Never use `data.shift(-1)` or any future-looking operation.
- Track every hypothesis in the notebook header: what you expected, what you found.

---

### `tests`

```bash
uv run pytest                                # all 145 tests
uv run pytest --ignore=tests/test_adapters.py   # skip network tests
uv run pytest tests/test_risk.py -v         # single file, verbose
```

| File | Covers |
|---|---|
| `test_adapters.py` | YFinance schema, date range, null checks (hits network) |
| `test_storage.py` | Parquet roundtrip, idempotency, list_symbols |
| `test_features.py` | Per-feature correctness, no-lookahead snapshot tests |
| `test_signals.py` | Momentum ranking, weight schemes, rebalance dates |
| `test_portfolio.py` | Ledoit-Wolf covariance, all 5 optimizer methods, position cap |
| `test_risk.py` | Exposures, VaR, pre-trade hard blocks, stress scenarios |
| `test_orchestration.py` | Upsert helpers, pipeline step sequencing |
| `test_execution.py` | Trader idempotency, PaperBroker fills, reconciler drift detection |

---

## Setup

### Prerequisites

- Python 3.13 via `uv`
- Windows, macOS, or Linux

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Clone and install

```bash
git clone git@github.com:micahabanschick/QuantPipe.git
cd QuantPipe
uv sync                           # core dependencies
uv sync --extra dev               # pytest, ruff, black
uv sync --extra backtest          # VectorBT
uv sync --extra portfolio         # cvxpy, PyPortfolioOpt
uv sync --extra execution         # ib_insync
# Install everything at once:
uv sync --extra dev --extra backtest --extra portfolio --extra execution
```

### Configure environment

```bash
cp .env.example .env
# Edit .env — fill in API keys (see .env.example for all variables)
```

### Seed historical data

```bash
uv run python orchestration/backfill_history.py
# Takes 5–15 minutes. Populates data/bronze/ for all 26 ETFs and 9 crypto symbols.
```

### Compute gold-layer features

```bash
uv run python features/compute.py
# Writes pre-computed features to data/gold/equity/features/
```

### Generate first signal snapshot

```bash
uv run python orchestration/generate_signals.py
# Creates data/gold/equity/target_weights.parquet and portfolio_log.parquet
# Required before launching the dashboards.
```

### Register automated daily tasks (Windows)

```bash
uv run python orchestration/setup_scheduler.py
# Registers QuantPipe\DailyPipeline (06:15) and QuantPipe\DailyRebalance (16:30)
```

---

## Running the Pipeline

### Validate end-to-end

```bash
uv run python backtest/canary.py
# Expected: Sharpe 0.5–1.5. Prints full tearsheet + risk report.
# Confirmed result: Sharpe 1.036, CAGR 17.1%, Max DD -14.3%
```

### Daily operations

```bash
# 1. Ingest + features + signals (automated via Task Scheduler at 06:15)
uv run python orchestration/run_pipeline.py

# 2. Paper rebalance at market close (automated via Task Scheduler at 16:30)
uv run python orchestration/rebalance.py --broker paper

# 3. All dashboards (Health · Performance · Strategy Lab)
streamlit run app.py
```

### Run tests

```bash
uv run pytest --ignore=tests/test_adapters.py   # fast, no network
```

---

## Build Phases

| Phase | Deliverable | Status |
|---|---|---|
| **0 — Setup** | Repo, dependencies, accounts | Complete |
| **1 — Data layer** | Daily bars for 26 ETFs + 10 crypto, queryable in <1s | Complete |
| **2 — Data quality + features** | Adjustments, universe-as-of-date, 5 factors, snapshot tests | Complete |
| **3 — Backtest + first signal** | Canary strategy: Sharpe ~1.0, walk-forward validated | Complete |
| **4 — Portfolio + risk** | Ledoit-Wolf, 5 optimizer methods, VaR, pre-trade checks, stress scenarios | Complete |
| **5 — Reporting + orchestration** | Two Streamlit dashboards, Task Scheduler automation, Pushover alerts | Complete |
| **6 — Paper trading** | Trader, IBKRAdapter, CCXTBroker, reconciler, rebalance script, Task Scheduler live | Complete |
| **7 — Go live** | Gate: 4 weeks green paper trading, implementation gap < 25% of Sharpe | Pending |
| **8 — Expand** | Paid data, additional strategies, scale capital | Future |

**Gate criteria before Phase 7** (all must be true):
- Paper trading reconciler green for 4+ consecutive weeks
- Implementation gap understood and within acceptable bounds
- Pre-trade checks tested — attempting to violate limits must produce a block
- Written kill-switch procedure exists
- Dashboard checked daily

---

## Design Principles

1. **Separation of concerns.** Each module does one thing and exposes a clean interface. The data layer never talks to execution.
2. **Point-in-time correctness.** No lookahead bias. Features use only data available on or before date T. Snapshot tests enforce this.
3. **Research-to-production parity.** The same code generates signals in research and runs live — no rewrite gap.
4. **Honest evaluation.** Cost models (5bps round-trip), slippage, and walk-forward validation are non-negotiable.
5. **Hard risk gates.** Pre-trade checks block orders — they are never downgraded to warnings.
6. **Incremental deployability.** Each phase delivers something real and usable before the next begins.
7. **Idempotency everywhere.** Writes upsert by key, order computation is pure, reruns produce no duplicates.

---

## Reference Documents

| Document | Contents |
|---|---|
| [`quantfinance_pipeline_outline.md`](quantfinance_pipeline_outline.md) | All 11 apps, feasibility notes, difficulty ratings, recommended build order |
| [`quantfinance_build_plan.md`](quantfinance_build_plan.md) | Constrained build plan: tooling, budget, per-phase milestones, per-app difficulty |
| [`quantfinance_kickoff_week1.md`](quantfinance_kickoff_week1.md) | Day-by-day Week 1 plan, setup checklist, strategy decisions (ETF 9-box rotation, crypto sidecar, options as live overlay) |
