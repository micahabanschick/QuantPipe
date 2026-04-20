# QuantPipe

An end-to-end quantitative finance pipeline for systematic equity and crypto trading. Built around a **Sector × Size × Style ETF rotation** framework, with a clear path from free data and paper trading to live deployment.

**Stack:** Python 3.12 · Polars · DuckDB · Parquet · VectorBT · Streamlit · IBKR · CCXT · `uv`

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
[storage]         ← Parquet on disk, DuckDB queries, bronze/silver/gold layers
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
[execution]       ← BrokerAdapter protocol → IBKR paper/live, CCXT, PaperBroker
    ▼
[reports]         ← Streamlit dashboards (ops health + P&L)
    │
[orchestration]   ← cron-driven scheduler, alerting via Pushover/ntfy
```

**Key invariants:**
- The data layer never talks to execution.
- Every feature is a pure function of data available on or before date T — no lookahead.
- `DataAdapter` and `BrokerAdapter` are protocols; swapping providers requires no downstream changes.
- Paper and live trading share 100% of the strategy/execution code path.

---

## Repository Structure

```
QuantPipe/
│
├── config/                     Universe definitions and environment settings
│   ├── universes.py            9-box ETFs, GICS sectors, crypto universe
│   └── settings.py             Paths, API keys, pipeline defaults (loaded from .env)
│
├── data_adapters/              Market data provider adapters
│   ├── base.py                 DataAdapter protocol + OHLCV schema
│   ├── yfinance_adapter.py     Yahoo Finance (equities, ETFs) — free
│   └── ccxt_adapter.py         CCXT unified interface (crypto, any exchange)
│
├── storage/                    Parquet-on-disk persistence layer
│   ├── parquet_store.py        write_bars(), load_bars(), list_symbols()
│   └── validators.py           Post-ingestion data quality checks
│
├── features/                   Point-in-time feature library
│   └── canonical.py            5 canonical factors (see below)
│
├── signals/                    [Phase 3] Cross-sectional signal generation
├── backtest/                   [Phase 3] VectorBT integration and strategy wrappers
├── portfolio/                  [Phase 4] Signal → target weights (cvxpy / PyPortfolioOpt)
├── risk/                       [Phase 4] Pre-trade checks, VaR, exposure monitoring
│
├── execution/                  Broker adapters and order management
│   ├── base.py                 BrokerAdapter protocol, Order, Fill, Position types
│   └── paper_broker.py         In-memory paper broker for local testing
│
├── orchestration/              Job scheduling and pipeline coordination
│   ├── ingest_daily.py         Nightly incremental ingestion (run by cron)
│   └── backfill_history.py     One-shot historical backfill (run once to seed data)
│
├── reports/                    Streamlit dashboards
│   └── health_dashboard.py     Ops dashboard: ingestion status, row counts, log tail
│
├── research/                   Jupyter notebooks (exploration only, never production)
│
├── tests/                      Pytest test suite
│   ├── test_adapters.py        Adapter schema and behaviour tests
│   ├── test_storage.py         Parquet store roundtrip and idempotency tests
│   └── test_features.py        Feature snapshot and correctness tests
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
| `EQUITY_UNIVERSE` | Union of all above, sorted |
| `CRYPTO_UNIVERSE` | BTC, ETH, SOL, AVAX, LINK, ADA, DOT, MATIC, BNB, XRP (USDT pairs) |

**`settings.py`** — loads all configuration from `.env`. Copy `.env.example` to `.env` and fill in your keys. Never commit `.env`.

---

### `data_adapters`

All adapters implement the `DataAdapter` protocol defined in `base.py`.

```python
from data_adapters import YFinanceAdapter, CCXTAdapter
from datetime import date

adapter = YFinanceAdapter()
df = adapter.get_bars("SPY", date(2020, 1, 1), date(2024, 12, 31))
# Returns Polars DataFrame with: date, symbol, open, high, low, close, volume, adj_close
```

Every adapter returns the same canonical OHLCV schema (`OHLCV_SCHEMA` in `base.py`). This is the contract all downstream code depends on.

| Adapter | Asset class | Cost | Notes |
|---|---|---|---|
| `YFinanceAdapter` | Equities, ETFs | Free | Auto-adjusted + raw close stored. Phase 1. |
| `CCXTAdapter` | Crypto | Free | Any CCXT exchange; defaults to Kraken. Pagination handled. |
| `IBKRAdapter` | All | Requires account | Phase 6 — not yet implemented. |

---

### `storage`

Parquet files on disk, queryable via DuckDB. No server required.

**Layout:**
```
data/
└── bronze/
    ├── equity/daily/symbol=SPY/year=2024/data.parquet
    ├── equity/daily/symbol=QQQ/year=2024/data.parquet
    └── crypto/daily/symbol=BTC_USDT/year=2024/data.parquet
```

**Key functions:**

```python
from storage import write_bars, load_bars, list_symbols

# Write (idempotent — re-writing the same rows produces no duplicates)
write_bars(df, asset_class="equity", symbol="SPY")

# Read via DuckDB hive scan — fast multi-symbol time-range queries
df = load_bars(["SPY", "QQQ"], start=date(2020,1,1), end=date(2024,12,31))

# List all symbols present in storage
symbols = list_symbols("equity")  # ["AGG", "BTC_USDT", "GLD", ...]
```

**Bronze → Silver → Gold convention:**
- `bronze` — raw vendor data, write-once, never overwritten
- `silver` — normalized, adjustments applied *(Phase 2)*
- `gold` — analysis-ready features *(Phase 2)*

**Validators** (`storage/validators.py`) run after every ingestion:

| Check | What it catches |
|---|---|
| `check_row_counts` | Missing trading days |
| `check_null_rate` | Null rates above 1% in price columns |
| `check_price_jumps` | Daily moves > 8σ — flags for manual review |
| `check_staleness` | Last bar older than 3 trading days |

---

### `features`

Five canonical factors implemented as pure functions in `features/canonical.py`. All are backward-looking only — no lookahead bias possible by construction.

| Function | Description |
|---|---|
| `log_return(prices, periods=1)` | Log return over N periods |
| `realized_vol(prices, window=21)` | Rolling annualized volatility (sqrt-252 scaled) |
| `momentum_12m_1m(prices)` | 12-month return excluding most recent month (Jegadeesh & Titman) |
| `dollar_volume(close, volume, window=63)` | Rolling average dollar volume — used as liquidity filter |
| `reversal_5d(prices)` | Negative 5-day return (positive value = recent loser expected to revert) |

**Batch compute:**

```python
from features import compute_features

# Compute all 5 features for a multi-symbol DataFrame
result = compute_features(bars_df)

# Compute a subset
result = compute_features(bars_df, feature_list=["log_return_1d", "momentum_12m_1m"])
```

Snapshot tests in `tests/test_features.py` pin output to a fixed input so any accidental lookahead contamination fails the test suite immediately.

---

### `signals`

*Phase 3 — stub.* Will contain cross-sectional ranking signals that combine features into a `[date, symbol, signal_value]` DataFrame.

Planned signals:
1. 12-1 cross-sectional momentum
2. Risk-adjusted momentum (momentum / realized vol)
3. Trend + mean-reversion overlay
4. Sector carry (relative yield approximation)
5. Regime overlay (market trend → net exposure)

---

### `backtest`

*Phase 3 — stub.* Will integrate VectorBT with the strategy interface:

```python
run_backtest(features, params, cost_model) -> BacktestResult
```

**Canary validation target:** equal-weight top-decile 12-1 momentum on the equity universe, monthly rebalance, 5bps round-trip cost model — should reproduce ~0.7–1.0 Sharpe over 2010–2024. If it doesn't, there's a bug in the pipeline.

---

### `portfolio`

*Phase 4 — stub.* Pure function interface:

```python
construct_portfolio(signals, cov_matrix, constraints) -> weights_df
```

Will implement vol-scaled equal weighting (baseline), then Ledoit-Wolf shrinkage + mean-variance optimization via `cvxpy` / `PyPortfolioOpt`.

---

### `risk`

*Phase 4 — stub.* Pre-trade checks are **hard blocks**, not advisory warnings.

Planned:
- Historical VaR (1-day, 95%, last 252 days)
- Gross/net/sector exposure limits
- Top-10 concentration cap
- Stress scenarios: 2008 Sep, 2020 Mar, 2022 rate shock

---

### `execution`

**`base.py`** — `BrokerAdapter` protocol that all broker implementations must satisfy:

```python
class BrokerAdapter(Protocol):
    def get_positions(self) -> list[Position]: ...
    def get_cash(self) -> float: ...
    def place_order(self, order: Order) -> str: ...       # returns order_id
    def cancel_order(self, order_id: str) -> bool: ...
    def get_fills(self, since: datetime | None) -> list[Fill]: ...
```

**`paper_broker.py`** — in-memory paper broker for smoke-testing order flow locally without any broker connection. Fills instantly at provided mark prices.

```python
from execution.paper_broker import PaperBroker
from execution.base import Order

broker = PaperBroker(initial_cash=100_000)
broker.set_prices({"SPY": 520.0, "QQQ": 445.0})
broker.place_order(Order(symbol="SPY", qty=10))
print(broker.get_positions())
```

*Phase 6:* `IBKRAdapter` (via `ib-async`) and live `CCXTAdapter` will be added. Paper and live share the same code path — only the adapter changes.

---

### `orchestration`

**`ingest_daily.py`** — nightly incremental ingestion script.

```bash
# Run manually
uv run python orchestration/ingest_daily.py

# Cron entry (10pm ET, weekdays) — add via: crontab -e
0 22 * * 1-5 cd ~/quantpipe && uv run python orchestration/ingest_daily.py >> logs/ingest.log 2>&1
```

Fetches a 30-day overlap window per symbol (to catch vendor revisions), writes to Parquet, validates every batch, and sends a Pushover/ntfy alert on any failure. Exits with code 0 (all ok), 1 (partial failures), or 2 (all failed).

**`backfill_history.py`** — one-shot historical seed. Run once after cloning.

```bash
# Full 7-year backfill for all asset classes
uv run python orchestration/backfill_history.py

# Equities only, custom date range
uv run python orchestration/backfill_history.py --asset-class equity --start 2018-01-01
```

---

### `reports`

**`health_dashboard.py`** — Streamlit ops dashboard.

```bash
streamlit run reports/health_dashboard.py
# Opens at http://localhost:8501
```

Shows:
- Last successful ingestion timestamp per asset class
- Symbol counts in storage
- Row counts per symbol for the last 30 days
- Recent validation failures highlighted in red
- Full ingestion log tail

*Phase 5:* A second dashboard (`performance_dashboard.py`) will show daily P&L, drawdown, rolling Sharpe, and exposures.

---

### `research`

Jupyter notebooks for exploratory work. Naming convention: `YYYY-MM-DD_short-description.ipynb`.

Rules:
- Notebooks are for exploration only — logic that matters migrates to a module and gets imported.
- Never use `data.shift(-1)` shortcuts or any future-looking operation.
- Track every hypothesis in the notebook header: what you expected, what you found.

First notebook to create: `research/01_universe_exploration.ipynb` — load 5 years of bars, plot cumulative returns, compute rolling correlations, confirm the pipeline works end-to-end.

---

### `tests`

```bash
# Run all tests
uv run pytest

# Run a specific file
uv run pytest tests/test_features.py -v

# Skip network tests (adapter tests hit Yahoo Finance)
uv run pytest tests/test_storage.py tests/test_features.py -v
```

| File | What it covers |
|---|---|
| `test_adapters.py` | YFinance schema, date range, null checks, batch fetch (hits network) |
| `test_storage.py` | Write/read roundtrip, idempotency, empty writes, list_symbols |
| `test_features.py` | Per-feature correctness, no-lookahead snapshots, compute_features orchestrator |

---

## Setup

### Prerequisites

- Windows with WSL2 Ubuntu (recommended) or macOS/Linux
- Python 3.12 via `uv`

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Clone and install

```bash
git clone git@github.com:micahabanschick/QuantPipe.git
cd QuantPipe
uv sync                          # installs core dependencies
uv sync --extra dev              # adds pytest, ruff, black, pre-commit
uv sync --extra backtest         # adds VectorBT (Phase 3)
uv sync --extra portfolio        # adds cvxpy, PyPortfolioOpt (Phase 4)
uv sync --extra execution        # adds ib-async (Phase 6)
```

### Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys (see .env.example for all variables)
```

### Seed historical data

```bash
uv run python orchestration/backfill_history.py
# Takes ~5–15 minutes depending on rate limits
```

---

## Running the Pipeline

### One-off ingestion

```bash
uv run python orchestration/ingest_daily.py
```

### Historical backfill

```bash
uv run python orchestration/backfill_history.py --asset-class equity --start 2018-01-01
```

### Health dashboard

```bash
streamlit run reports/health_dashboard.py
```

### Tests

```bash
uv run pytest
```

### Cron setup (WSL2)

```bash
crontab -e
# Add:
0 22 * * 1-5 cd ~/QuantPipe && /home/$USER/.local/bin/uv run python orchestration/ingest_daily.py >> logs/ingest.log 2>&1
```

---

## Build Phases

The pipeline is built incrementally — each phase produces something usable before moving to the next.

| Phase | Weeks | Apps | Deliverable |
|---|---|---|---|
| **0 — Setup** | 0 | — | Repo, dependencies, accounts open |
| **1 — Data layer** | 1–3 | Ingestion + Storage | 90 days of daily bars for full universe, queryable in <1s |
| **2 — Data quality + features** | 4–5 | Data Quality + Feature Library | Adjustments, universe-as-of-date, 5 factors with snapshot tests |
| **3 — Backtest + first signal** | 6–8 | Research + Backtest | Canary momentum strategy reproducing known ~0.7–1.0 Sharpe |
| **4 — Portfolio + risk** | 9–10 | Portfolio + Risk | Every backtest run produces a risk report; pre-trade checks block violations |
| **5 — Reporting + orchestration** | 11 | Reporting + Orchestration | Green dashboard every morning, Pushover alert on failures |
| **6 — Paper trading** | 12–15 | Execution | 4-week paper trading gate: reconciler green, implementation gap < 25% of Sharpe |
| **7 — Go live** | 16+ | — | $1000 live across 3–5 equity positions + 2 crypto |
| **8 — Expand** | 6 months+ | — | Paid data (EODHD / Norgate), additional strategies, scale up capital |

**Gate criteria before going live** (all must be true):
- Paper trading green for 4+ consecutive weeks
- Implementation gap understood and within bounds
- Risk pre-trade checks tested and blocking violations
- Written kill-switch procedure exists
- Dashboard checked daily

---

## Design Principles

1. **Separation of concerns.** Each module does one thing and exposes a clean interface.
2. **Point-in-time correctness.** No lookahead bias. Features use only data available on or before date T.
3. **Research-to-production parity.** The code that generates a signal in research is the same code that runs live — no rewrite gap.
4. **Honest performance evaluation.** Cost models (5bps round-trip minimum), slippage, and walk-forward validation are non-negotiable.
5. **Incremental deployability.** The data layer alone is useful. Each phase ships something real.
6. **Reproducibility.** Every backtest run is re-runnable from a git SHA + config. No notebooks as production.

---

## Reference Documents

The three design documents in this repo capture the full reasoning behind every architecture decision:

| Document | Contents |
|---|---|
| [`quantfinance_pipeline_outline.md`](quantfinance_pipeline_outline.md) | Full description of all 11 apps, feasibility notes, difficulty ratings, and recommended build order |
| [`quantfinance_build_plan.md`](quantfinance_build_plan.md) | Constrained build plan: tooling choices, budget allocation, per-phase milestones, per-app difficulty adjusted for solo/multi-asset/daily constraints |
| [`quantfinance_kickoff_week1.md`](quantfinance_kickoff_week1.md) | Day-by-day Week 1 plan, Windows/WSL2 setup, account opening checklist, strategy decisions (ETF 9-box rotation, crypto sidecar, options as live overlay only) |
