# QuantFinance Pipeline — Kickoff & Week 1 Plan

**Stack context:** Windows · Solo · $50/mo · Daily frequency · $1000 live / $100k paper · All asset classes in scope · Paid backtesters OK if justified · Strategy core = Sector × Size × Style.

---

## 1. Strategic refinements based on your answers

### Asset class prioritization (revised)

| Asset class | Research priority | Paper trading | Live trading ($1000) |
|---|---|---|---|
| **US Equities (via ETFs)** | **HIGH** — primary research vehicle | Yes, full book | Yes, 3–5 positions with fractionals |
| **US Equities (single stocks)** | MED — later phase, needs data upgrade | Yes | Wait until >$5k capital |
| **Crypto** | HIGH — uncorrelated, cheap data | Yes | Yes, 2–3 positions |
| **Options** | LOW — live overlay only, no backtest | Live dry-run only | Defer |
| **Futures** | LOW — paper only | Yes, educational | **No — margin too high for $1000** |

### The single most important strategy decision

**Build the entire research platform around a 9-box × 11-sector ETF rotation model first.** This is a real institutional framework and it's tractable at your data budget.

**Why this fits your constraints:**
- **Maps directly to your Sector × Size × Style core.** The 9 style-box ETFs (e.g., IWF, IWD, IWB, IWP, IWS, IWR, IWO, IWN, IWM — large/mid/small × growth/blend/value) plus 11 GICS sector ETFs give you a 20-instrument universe that covers the entire equity style space.
- **No historical fundamentals required.** The ETF wrapper *is* the factor exposure. You're ranking ETFs by price-based signals (momentum, vol, trend), not individual stock fundamentals.
- **Free data is genuinely sufficient.** yfinance gives perfectly clean daily OHLCV on liquid ETFs.
- **Tradable at $1000 live.** IBKR Lite + fractional shares = zero commission, ~2 bps spread. A 5-ETF portfolio at $200 each is realistic.
- **Graduates cleanly.** The same cross-sectional ranking / portfolio construction / execution code works identically when you later add single stocks; you just expand the universe.

### What "building strategies" will actually look like

You'll be generating **cross-sectional signals** that rank ETFs within the Sector × Size × Style universe, combining them into a target weight vector, and rebalancing monthly or biweekly. Canonical starting signals:

1. **12-1 cross-sectional momentum** (rank by 12-month return ex the last month)
2. **Risk-adjusted momentum** (momentum / realized vol)
3. **Trend + mean reversion overlay** (trend filter at the index level, reversion at the ETF level)
4. **Sector carry** (relative dividend yield / earnings yield — approximation via ETF distributions)
5. **Regime overlay** (broad market trend determines net exposure)

Your first deliverable strategy — the "canary" — will be **9-box × sector cross-sectional 12-1 momentum, monthly rebalance, vol-targeted equal weight, top-5 positions long only**. Aim to reproduce a ~0.7–1.0 Sharpe net of 5bps costs over 2010–2024 as the sanity check that your pipeline is correct.

### Crypto sidecar

Run a parallel, simpler strategy in crypto: **5–10 asset momentum rotation** (BTC, ETH, SOL, plus top liquid alts). Weekly rebalance. Completely independent of the equity pipeline — different universe, different signals, lower correlation to equities, no PDT rules, no commission issues. Live-tradeable at $100–$500 allocation.

### Options: how to actually use them at your budget

Since historical options backtesting is out of budget, the right posture is: **options as a live overlay driven by equity signals**. Examples:
- If your equity model is net long and volatility is elevated, sell covered calls on positions for extra yield.
- If a position hits a stop-loss signal, buy short-dated puts instead of selling (tax / PDT efficient).
- Skip directional options speculation — you have no backtest evidence for it.

Defer this entirely until the equity pipeline is live and profitable. No work on options in Months 1–5.

---

## 2. Revised tooling stack (Windows-specific)

| Layer | Choice | Install notes |
|---|---|---|
| Dev environment | **WSL2 Ubuntu inside Windows** | Strongly recommended — gives you native Linux tooling (cron, bash, pip ecosystem edge cases) while keeping Windows for Norgate + IBKR TWS if needed. Install via `wsl --install`. |
| Editor | VS Code with Remote-WSL extension | Lives in Windows, executes inside WSL seamlessly. |
| Python | 3.12 via `uv` | Inside WSL. |
| Scheduler | `cron` inside WSL2 | Windows Task Scheduler works but is clunkier; cron inside WSL is cleaner once WSL is set to auto-start. |
| Broker (equities, options, futures) | **IBKR Lite** + `ib_async` (the maintained fork of `ib_insync`) | Lite gives free commissions on US ETFs. API works on both Lite and Pro. |
| Broker (crypto) | CCXT → Kraken or Coinbase Advanced | Start with one exchange. |
| Backtester | **VectorBT (free)** for research; **NautilusTrader (free)** only if execution fidelity becomes an issue | Don't pay for VectorBT PRO yet. QuantConnect Cloud is worth bookmarking for later if you want a hosted live environment but creates platform lock-in. |
| Data (free, Phase 1) | yfinance, CCXT, Alpaca paper API | |
| Data (paid, Phase 2+) | **Norgate Gold US Stocks ($30/mo)** — native Windows client, 20yr history, Python API, current fundamentals | Or EODHD All World (€22/mo) if you want global breadth over US depth. Pick one. |
| Fundamentals | Defer | Either Norgate Platinum or EODHD Fundamentals — both push you over budget. Not needed for the ETF strategy. |
| Storage | Parquet + DuckDB + Polars | All free, Windows-compatible via WSL. |
| Dashboard | Streamlit | |
| Alerts | Pushover ($5 one-time for iOS/Android app, unlimited alerts) or ntfy.sh (free) | |

**Why WSL2:** cron, shell scripts, Linux-style package management, tolerance for edge-case Python libs that struggle on Windows-native Python. Norgate runs on Windows proper and writes to a local DB that WSL can read via the `/mnt/c/...` path. Best of both worlds.

---

## 3. Account opening checklist — START TODAY

These run in parallel with Week 1 coding. Applications take real calendar time and are your critical path to live trading.

### Interactive Brokers (CRITICAL — start immediately)

- Go to interactivebrokers.com, apply for **IBKR Lite** individual account.
- Expect 3–14 calendar days for approval (varies by country of residence).
- While waiting, you can use the **paper trading account** which is usable within hours of application submission.
- Documents needed: government ID, proof of address (utility bill), bank account for funding. Tax forms for non-US residents.
- After approval: enable API access in Account Management → Settings → API → Settings.
- Download **IBKR Gateway** (lighter than TWS, API-only). You'll connect to Gateway from your code.

### Crypto exchange (start immediately)

Pick one to start. Recommend **Kraken** (best regulatory standing in US) or **Coinbase Advanced** (cleanest US tax reporting).

- Apply, do KYC. Expect 1–7 days.
- Deposit small amount ($50–100) to verify the flow.
- **Create API keys with trade + read permissions, NO withdrawal permission.** This is a hard rule.
- Store keys in environment variables, never in code.

### Pushover or ntfy.sh (5 minutes)

- Sign up. Note your API token + user key. You'll need these for alerting.

---

## 4. Windows setup — Day 0 (one evening, ~2 hours)

Run these in order. Each block is a checkpoint.

### 4.1 WSL2 Ubuntu

In PowerShell (Admin):
```
wsl --install -d Ubuntu-24.04
```
Restart when prompted. Set up Linux username/password on first launch.

### 4.2 Inside WSL (Ubuntu terminal)

```bash
# Update
sudo apt update && sudo apt upgrade -y

# Essentials
sudo apt install -y build-essential curl git python3-pip python3-venv cron

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Confirm
uv --version
```

### 4.3 VS Code + Remote-WSL

- Install VS Code on Windows (code.visualstudio.com).
- Install the **"WSL"** extension.
- Open VS Code → Command Palette (Ctrl+Shift+P) → `WSL: Connect to WSL`.
- You're now editing in Windows, executing in Linux. All subsequent terminals in VS Code are WSL bash.

### 4.4 Start the project

Still in WSL bash inside VS Code:
```bash
cd ~
mkdir quantpipe && cd quantpipe
git init
uv init --python 3.12
```

Accept the generated `pyproject.toml`. You now have a working project.

---

## 5. Week 1 — day-by-day

**Goal:** by end of Week 1, you have a working end-to-end data pipeline pulling daily bars for your 20-ETF universe, storing in Parquet, queryable in a notebook, with basic validation. This is the foundation every other app sits on.

### Day 1 — Project skeleton (~2 hours)

Create the directory structure:
```bash
mkdir -p {data_adapters,storage,features,signals,backtest,portfolio,risk,execution,reports,orchestration,research,tests,config,logs}
touch data_adapters/__init__.py storage/__init__.py features/__init__.py signals/__init__.py \
      backtest/__init__.py portfolio/__init__.py risk/__init__.py execution/__init__.py \
      reports/__init__.py orchestration/__init__.py
```

Add core dependencies:
```bash
uv add polars pyarrow duckdb pandas numpy scipy yfinance ccxt jupyter ipykernel
uv add --dev pytest ruff black pre-commit
```

Create `.gitignore` (exclude `data/`, `logs/`, `*.env`, `.venv/`, `.ipynb_checkpoints/`, `__pycache__/`).

Commit: `git add . && git commit -m "Initial project skeleton"`.

### Day 2 — Universe definition + config (~2 hours)

Create `config/universes.py`:
```python
# Morningstar 9-box style-size ETFs (iShares Russell variants)
STYLE_SIZE_9BOX = [
    "IWB", "IWF", "IWD",   # Large: Blend, Growth, Value
    "IWR", "IWP", "IWS",   # Mid:   Blend, Growth, Value
    "IWM", "IWO", "IWN",   # Small: Blend, Growth, Value
]

# GICS sector SPDRs
SECTOR_SPDRS = [
    "XLK", "XLV", "XLF", "XLY", "XLP",
    "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC",
]

# Broad benchmarks / risk-on-risk-off anchors
BENCHMARKS = ["SPY", "QQQ", "IWM", "AGG", "TLT", "GLD"]

EQUITY_UNIVERSE = sorted(set(STYLE_SIZE_9BOX + SECTOR_SPDRS + BENCHMARKS))

# Crypto universe (CCXT symbol format)
CRYPTO_UNIVERSE = [
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD",
    "LINK/USD", "ADA/USD", "DOT/USD", "MATIC/USD",
]
```

Commit.

### Day 3 — Data adapter interface + yfinance adapter (~3 hours)

Create `data_adapters/base.py`:
```python
from typing import Protocol
from datetime import date
import polars as pl

class DataAdapter(Protocol):
    name: str
    asset_classes: set[str]

    def get_bars(
        self, symbol: str, start: date, end: date, freq: str = "1d"
    ) -> pl.DataFrame:
        """Return DataFrame with columns: date, open, high, low, close, volume, adj_close."""
        ...
```

Create `data_adapters/yfinance_adapter.py` implementing this for equities. Key detail: always request auto-adjusted prices and store both raw and adjusted.

Create `data_adapters/ccxt_adapter.py` implementing this for crypto (use CCXT's `fetch_ohlcv`).

Write 2 unit tests in `tests/` confirming each adapter returns the expected schema.

Commit.

### Day 4 — Storage layer (~3 hours)

Create `storage/parquet_store.py` with two functions:
- `write_bars(df, asset_class, symbol)` → writes to `data/{asset_class}/daily/symbol={symbol}/` partitioned by year.
- `load_bars(symbols, start, end, asset_class)` → returns a single DataFrame, powered by DuckDB reading the Parquet paths.

Key detail: DuckDB's `read_parquet` with hive partitioning is effectively zero-config and very fast.

Write a unit test that writes then reads back and asserts equality.

Commit.

### Day 5 — Ingestion orchestrator (~2 hours)

Create `orchestration/ingest_daily.py`:
```python
# Pseudocode structure
def main():
    today = date.today()
    lookback_start = today - timedelta(days=30)  # overlap window for revisions

    for symbol in EQUITY_UNIVERSE:
        df = YFinanceAdapter().get_bars(symbol, lookback_start, today)
        write_bars(df, "equity", symbol)

    for symbol in CRYPTO_UNIVERSE:
        df = CCXTAdapter().get_bars(symbol, lookback_start, today)
        write_bars(df, "crypto", symbol.replace("/", "_"))

    log_ingestion_summary(...)

if __name__ == "__main__":
    main()
```

Then do a full historical backfill (5+ years) once, manually, by calling adapters with `start=date(2018, 1, 1)`.

Commit.

### Day 6 — Validation + smoke dashboard (~3 hours)

Create `storage/validators.py`:
- `check_row_counts(df)` — flag trading-day count deviations
- `check_null_rate(df, threshold=0.01)`
- `check_price_jumps(df, sigma_threshold=8)` — flag >8σ daily moves for manual review
- `check_staleness(df, max_age_days=3)`

Run after every ingestion. Log failures to `logs/validation.log` AND send Pushover alert on any failure.

Create `reports/health_dashboard.py` — a 50-line Streamlit app showing:
- Last successful ingestion timestamp per asset class
- Row counts per symbol for the last 30 days
- Recent validation failures

Run with `streamlit run reports/health_dashboard.py`. Leave it running on `localhost:8501`.

Commit.

### Day 7 — Schedule + first research notebook (~2 hours)

Set up cron (inside WSL):
```bash
crontab -e
# Add:
0 22 * * 1-5 cd /home/YOURUSER/quantpipe && /home/YOURUSER/.local/bin/uv run python orchestration/ingest_daily.py >> logs/ingest.log 2>&1
```

Make sure WSL is configured to auto-start (Windows Settings → System → Linux, or via `wsl --setdefault`). On Windows 11 22H2+, WSL services auto-start when cron is enabled.

Open a notebook in `research/01_universe_exploration.ipynb`. Load 5 years of bars for the full equity universe using `load_bars`. Plot cumulative returns. Compute rolling 63-day correlations. Eyeball the data. This is your "the pipeline works" moment.

Commit. Tag: `git tag v0.1-data-layer`.

---

## 6. After Week 1 — short checklist to stay on track

| Week | Deliverable |
|---|---|
| 2 | Adjustments table + universe-as-of-date function (App 3) |
| 3 | 5 pure-function features with snapshot tests (App 4) |
| 4 | VectorBT integration + canary strategy (9-box 12-1 momentum) backtest runs (App 5 + App 6) |
| 5 | Portfolio construction module (vol-scaled equal weight, top-5) (App 7) |
| 6 | Risk module (VaR, exposures, pre-trade checks) (App 8) |
| 7 | Performance dashboard (App 10) |
| 8–10 | IBKR + CCXT execution adapters, reconciler, paper trading loop (App 9) |
| 11–14 | 4-week paper trading gate with reconciler green |
| 15+ | Go live with $1000 across 3 equity positions + 2 crypto |

---

## 7. Specific risks at your scale

1. **$1000 is fragile.** A single 10% drawdown = $100, which you'll absorb psychologically and maybe not rationally. **Pre-commit in writing** to the position sizes and risk limits before you go live, and don't override them to "make back losses."

2. **Rebalance frequency matters more than signal quality at small capital.** Weekly rebalance with $200 positions = tiny turnover costs only on spread. Daily rebalance at this size is pointless — slippage dominates.

3. **Paper-live divergence is asymmetric.** Paper over-fills at close prices; live fills are worse. Budget ~5–10 bps additional slippage vs backtest on entries/exits, and don't be surprised if month-1 live underperforms by that margin.

4. **Don't scale up capital too fast.** Rule of thumb: don't double your live size until you've had 4 consecutive profitable weeks at the current size. Even that's aggressive.

5. **Windows-specific: IBKR Gateway vs TWS.** Gateway is lighter and survives daily restarts better. Schedule a daily Gateway restart (TWS-level requirement to re-auth every ~24h) via a Windows Task Scheduler task.

---

## 8. What to ask me when you hit a wall

Bring specific questions when you're stuck. Especially useful moments to ping back:

- **End of Week 1:** Show me your `research/01_universe_exploration.ipynb`. I'll flag anything off about the data.
- **End of Week 4:** Share the canary backtest tearsheet. If Sharpe is <0.3 or >2.0 you have a bug; I'll help locate it.
- **Before going live:** Walk me through the reconciler + risk checks. I'll try to break them.
- **Any time you're about to spend money on data or infrastructure:** sanity-check the buy.

You now have three docs: the original outline (reference), the constrained build plan (strategy), and this (action). The next move is yours — open the IBKR application today and start Day 1 of Week 1 tonight.
