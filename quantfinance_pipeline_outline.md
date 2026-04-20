# QuantFinance Pipeline — Project Outline

## 0. Framing assumptions

You haven't pinned down the specific apps in this doc yet, so I'm working from the standard architecture used by both small systematic shops and independent quant researchers in 2026. Adjust / prune based on your actual scope.

**Assumed overall goal:** build an end-to-end system that ingests market data, researches and validates signals, constructs portfolios, executes trades (paper first, live optional), and monitors risk/performance — modular enough that you can swap components.

**Assumed constraints:** solo or small-team build, Python-primary (C++ only where latency forces it), cloud-optional (local dev first), equities + optionally futures/options/crypto.

**Difficulty scale** used below: **L** = Low (weekend/week), **M** = Medium (a few weeks), **H** = High (1–3 months done well), **VH** = Very High (ongoing, never really "done").

---

## 1. Objectives for the project (overall)

1. **Separation of concerns.** Each app does one thing and exposes a clean interface. Data layer never talks to execution; research never hard-codes a broker.
2. **Reproducibility.** Every backtest, signal, and portfolio construction run should be re-runnable from a git SHA + config. No notebooks-as-production.
3. **Point-in-time correctness.** No lookahead bias. Universe, fundamentals, and features are as-of-date, not as-of-today.
4. **Research-to-production parity.** The code that generates a signal in research is the same code that runs it live — no rewrite gap.
5. **Honest performance evaluation.** Costs, slippage, borrow, capacity, and overfitting checks baked in. A strategy that only works net of zero costs is a toy.
6. **Incremental deployability.** You should be able to ship the data layer alone, then research alone, and so on. Don't build everything before anything works.

---

## 2. The apps, one by one

### App 1 — Data Ingestion

**Objective:** Pull market data (OHLCV, tick, corporate actions), fundamentals, and optionally alt data into a local or cloud landing zone on a schedule.

**Difficulty:** M for daily/minute equities; H–VH if you want tick/MBO or multi-asset.

**Feasibility notes:**
- Free tiers (yfinance, Alpha Vantage, IEX Cloud) are fine for prototyping but have holes and revisions that will bite you later.
- Paid options scale with seriousness: Polygon.io for broad US equities/options at flat rates, Databento for tick/MBO with nanosecond timestamps, Intrinio for institutional breadth, Interactive Brokers' feed if you're also executing there.
- Biggest hidden costs: survivorship-bias-free historical universes and point-in-time fundamentals. These are genuinely hard to get cheaply.

**How to start:**
1. Pick one asset class and one frequency to begin (daily US equities is the safest first target).
2. Write a thin adapter interface (`get_bars(symbol, start, end, freq)`) so you can swap providers later without touching downstream code.
3. Start with yfinance + one paid provider's free tier. Get a nightly cron/scheduler pulling and writing to Parquet files partitioned by date.
4. Only add corporate actions (splits, dividends) and delisted symbols once the basic pipeline runs daily for a week without manual fixes.

---

### App 2 — Data Storage / Warehouse

**Objective:** Persist raw and cleaned data in a format that supports fast time-range queries, append-only writes, and reproducible point-in-time reads.

**Difficulty:** L for Parquet-on-disk daily data; H if you need intraday tick at scale.

**Feasibility notes:**
- **Daily/minute bars:** Parquet + DuckDB or Parquet + Polars is extremely good in 2026 — no server to run, fast queries, trivially cheap.
- **Intraday tick / order book:** you're pushed toward ClickHouse, QuestDB, TimescaleDB, or (at institutional level) kdb+. kdb+ is ubiquitous at pro shops but expensive and requires learning q.
- Partition by date first, symbol second — this is the single most important decision for query speed.

**How to start:**
1. Skip the database for week one. Write Parquet files in a `data/{asset_class}/{freq}/date=YYYY-MM-DD/` layout and query with DuckDB.
2. Add a "bronze → silver → gold" convention: bronze = raw vendor data, silver = normalized, gold = analysis-ready. Never overwrite bronze.
3. Only graduate to a running database (Timescale / QuestDB) when you actually have a query pattern that Parquet+DuckDB can't serve fast enough. Most solo projects never need this.

---

### App 3 — Data Quality & Normalization

**Objective:** Turn messy vendor data into clean, aligned, point-in-time-correct time series that downstream apps can trust.

**Difficulty:** H. Underestimated by nearly everyone. This is where bugs hide for months.

**Feasibility notes:**
- Problems you will hit: timezone drift, corporate-action-adjusted vs unadjusted prices, symbol changes (ticker reuse!), mid-day revisions, missing bars, bad ticks, halt gaps.
- Must have: symbology mapping (PERMNO / FIGI / ISIN / internal ID), an adjustments table applied at query time rather than baked into storage.
- Survivorship bias is the silent killer — if your universe on date T includes only companies still listed today, every backtest is wrong.

**How to start:**
1. Build a validator app that runs after every ingestion: checks row counts, null rates, price jumps > N sigma, zero-volume sessions, stale symbols.
2. Store adjustments (splits, dividends, ticker changes) as a separate table keyed by date. Apply them at read time in a `get_adjusted_prices()` helper.
3. Write a "universe-as-of-date" function early. Everything downstream should call it rather than globing all CSVs.
4. Log every data issue to a dashboard. Don't silent-fix — you want to see your vendor's quality degrading.

---

### App 4 — Feature / Factor Library

**Objective:** A versioned library of reusable features (returns, volatility, momentum, value, quality, microstructure features, etc.) that are point-in-time safe and composable.

**Difficulty:** M–H. The math is easy; the correctness is hard.

**Feasibility notes:**
- Every feature must be a pure function of data available on or before date T. The moment one feature uses `data.shift(-1)` in a shortcut, your backtest is contaminated.
- Rolling-window features need to handle missing bars, listings, and delistings explicitly.
- Keep features small and testable — a 3-line `momentum_12m_1m` function with unit tests beats a 200-line "FactorEngine" class.

**How to start:**
1. Pick 5 canonical factors: 12-1 momentum, realized vol, log returns, dollar volume, short-term reversal. Implement each as a pure function with a clear signature.
2. Write a snapshot test per feature: given a fixed input CSV, feature output should be byte-identical over time. This catches accidental lookahead.
3. Add a `compute_features(symbols, dates, feature_list)` orchestrator that writes results to your gold layer.
4. Once 5 features are bulletproof, expand.

---

### App 5 — Research / Signal Generation

**Objective:** Exploratory environment (notebooks + scripts) where you combine features into signals/alphas, fit models, and decide what to promote to backtesting.

**Difficulty:** M mechanically, VH to actually find edge.

**Feasibility notes:**
- Notebooks are great for exploration, terrible for production. Keep the research-vs-library discipline from day one: logic that matters migrates to the feature library or a `signals/` module and gets imported into notebooks, not copy-pasted.
- ML is overused here. Linear models + careful feature construction dominate most ML approaches in low-SNR finance problems until you have massive data.
- Cross-validation for time-series is its own discipline — purged, embargoed K-fold à la López de Prado, or walk-forward. Random K-fold is wrong for time series.

**How to start:**
1. Set up a `research/` folder with one notebook per hypothesis. Prefix notebooks with dates.
2. Write a baseline: equal-weight top-decile momentum. If you can't reproduce a known anomaly with your data + features, your pipeline is broken — fix that before trying clever stuff.
3. Adopt walk-forward validation immediately. Don't use scikit-learn's default KFold on prices.
4. Track every experiment in a simple CSV or MLflow — in a year you'll have hundreds and won't remember what you tried.

---

### App 6 — Backtesting Engine

**Objective:** Simulate how a strategy would have performed historically, accounting for costs, slippage, borrow, and realistic fills.

**Difficulty:** H if you build it; M if you adopt an existing one well.

**Feasibility notes (2026 landscape):**
- **Vectorized, research-speed:** VectorBT (and VectorBT PRO) or Backtesting.py — fast, great for parameter sweeps, weaker on realistic execution.
- **Event-driven, live-parity focus:** NautilusTrader — actively developed, designed so the same strategy code runs backtest and live. Best choice if you care about execution fidelity.
- **Factor / daily-universe:** Zipline-Reloaded or the `bt` library for portfolio-weighting strategies.
- **All-in-one platform:** QuantConnect / LEAN — hosted, broad asset coverage, locks you into their ecosystem.
- **Building your own:** educational but rarely a good use of time unless you have exotic needs.

**How to start:**
1. Do not build your own. Pick one: VectorBT for research speed, NautilusTrader for live-parity, Zipline-Reloaded for equity factor work.
2. Reproduce a known result (e.g., S&P 500 buy-and-hold, then 12-1 momentum long-short) to validate the tool's outputs against published numbers.
3. Add explicit cost models: per-share commission, bid-ask spread, market impact as a function of % ADV. A strategy whose Sharpe survives 10bps round-trip is worth looking at; one that needs zero costs is not.
4. Wire the backtester to your data layer via the adapter from App 1 — never hard-code a data source inside a strategy file.

---

### App 7 — Portfolio Construction / Optimization

**Objective:** Turn signals into target portfolio weights given risk constraints, turnover limits, and capital.

**Difficulty:** M for basic approaches, H for production-grade with constraints.

**Feasibility notes:**
- Mean-variance optimization is famously unstable with estimated inputs — small changes in expected returns cause huge swings in weights. Shrinkage (Ledoit-Wolf), Black-Litterman, or robust/constrained optimization all help.
- Libraries: `PyPortfolioOpt` for textbook methods, `cvxpy` for custom convex formulations, `Riskfolio-Lib` for more advanced risk measures, `skfolio` for scikit-learn-style workflows.
- Equal-weight or vol-scaled weighting often beats optimizer-driven weights in practice once costs are realistic.

**How to start:**
1. Begin with vol-targeted equal weighting across your signal's long and short legs. This is a strong baseline and reveals whether the signal has edge before optimizer hides it.
2. Add Ledoit-Wolf shrinkage on the covariance matrix before any mean-variance step.
3. Build the app as a pure function: `signals + covariance + constraints → weights`. No I/O, no state. Makes testing trivial.
4. Add turnover and position-size constraints before leverage targeting.

---

### App 8 — Risk Management

**Objective:** Monitor and bound exposure — per-name, per-sector, per-factor, and at the book level. Compute VaR / drawdown / stress scenarios.

**Difficulty:** M for a retail / small-book version, VH for genuine institutional-grade.

**Feasibility notes:**
- Most solo pipelines conflate risk with ex-post performance. Real risk management is ex-ante: what *could* the portfolio lose under defined scenarios before we place the trade?
- For factor risk, a simple 5–10 factor model (market, size, value, momentum, quality, sector dummies) covers most of what a solo shop needs.
- Historical simulation VaR is fine for most purposes and easier to explain than parametric.

**How to start:**
1. Compute per-day exposures: gross, net, by sector, by top-10 concentration. Log them every run.
2. Implement historical VaR using the last N trading days of returns applied to current weights.
3. Add hard pre-trade checks: reject orders that would violate position/sector caps. Nothing fancy — just `if new_exposure > limit: block`.
4. Add a small library of stress scenarios (2008 Sept, 2020 March, 2022 rate shock) replayed against current positions.

---

### App 9 — Execution / Order Management

**Objective:** Take target weights, compute orders, route them to a broker (paper or live), track fills, update positions.

**Difficulty:** M for paper / daily rebalance, H for live, VH for anything latency-sensitive.

**Feasibility notes:**
- Broker choices for retail/prosumer in 2026: Interactive Brokers (IBKR API / ib_insync — still the standard), Alpaca (simpler, US equities + crypto), Tradier, Tradovate for futures. Crypto: Binance, Coinbase, Kraken.
- "Paper" and "live" should share 99% of the code path. Feature-flag the broker, not the strategy.
- Order management = idempotency is everything. Network retries, duplicate fills, partial fills, exchange halts — assume every bad thing will happen eventually.

**How to start:**
1. Start with paper trading via Alpaca or IBKR paper account. Get the full loop working: compute weights → generate orders → submit → record fills → reconcile to target.
2. Build a reconciler that compares your internal position state to the broker's every run. Flag drift loudly.
3. Only move to live capital after: 2+ weeks of paper trading matches backtest within expected variance, reconciler has been green for that whole period, risk app has hard pre-trade checks.
4. Use small size for the first live month. Whatever size feels embarrassingly small, go smaller.

---

### App 10 — Monitoring, Reporting, Attribution

**Objective:** Dashboards and reports showing P&L, drawdown, exposures, Sharpe/Sortino, attribution by signal / sector / factor, and pipeline health (data freshness, job success).

**Difficulty:** M.

**Feasibility notes:**
- Two audiences: you (the PM) looking at performance, and you (the sysadmin) looking at whether the pipeline is healthy. Keep them on separate dashboards — mixing them means you ignore both.
- Tools: Grafana for ops/data freshness (plays well with QuestDB/Timescale/Prometheus), Streamlit or a Jupyter-based dashboard for research-oriented P&L/attribution views.
- Attribution is where you learn what's actually working. Break returns down by signal, by holding period, by sector, by size decile — regularly.

**How to start:**
1. Build the pipeline-health dashboard first. Last successful ingestion run, row counts per day, data-validation pass rate. If this is missing, you're flying blind.
2. Add a daily P&L report: positions at open, at close, realized + unrealized, turnover, fees.
3. Add attribution last, once you have at least a few months of real or paper returns to decompose.

---

### App 11 — Orchestration (the glue)

**Objective:** Schedule, chain, retry, and monitor all the jobs above. Ingestion → validation → feature compute → signal generation → portfolio construction → (optional) execution → reporting.

**Difficulty:** M.

**Feasibility notes:**
- **Airflow** — industry standard, heavy, overkill for solo.
- **Dagster** — asset-oriented model fits this domain really well (features and datasets are assets). Strong choice in 2026.
- **Prefect** — lighter, Pythonic, good for small teams.
- **Cron + Python scripts** — totally valid for a solo project for the first 6 months. Don't over-engineer.

**How to start:**
1. Use `cron` + a shell script that runs each stage and exits non-zero on failure, with an alerting hook (Pushover, Telegram, email). Do this for the first few months.
2. Migrate to Dagster or Prefect once you have more than ~10 scheduled jobs or you want data-lineage visualization.
3. Never run orchestration as the compute layer — it coordinates jobs, it doesn't *do* them.

---

## 3. Suggested build order

Don't build top-to-bottom. Build in the order that lets you *use* the pipeline as early as possible:

1. **App 1 (Ingestion) + App 2 (Storage)** — minimal, one asset class, daily. Week 1–2.
2. **App 3 (Data Quality)** — just enough to not embarrass yourself: validation + adjustments + universe-as-of. Week 2–3.
3. **App 4 (Feature Library)** — 5 canonical factors, tested. Week 3–4.
4. **App 6 (Backtest)** — adopt an existing framework, reproduce a known anomaly. Week 4–5.
5. **App 5 (Research)** — now you can actually iterate on ideas. Ongoing forever.
6. **App 7 (Portfolio Construction)** — once you have at least one signal that looks real.
7. **App 10 (Reporting)** — stand up the monitoring dashboards around the time you start backtesting seriously.
8. **App 8 (Risk)** — before any paper trading, not after.
9. **App 9 (Execution)** — paper first, live only after paper matches backtest.
10. **App 11 (Orchestration)** — upgrade from cron when the cron script gets ugly, not before.

---

## 4. Biggest traps, ranked

1. **Lookahead bias.** Will invalidate months of work silently. Defend against it at the feature-library layer.
2. **Survivorship bias.** Your "universe" must include delisted names.
3. **Overfitting.** Walk-forward, purged CV, out-of-sample holdout always. Track every hyperparameter you've touched.
4. **Ignoring costs.** 5–10bps round-trip is a realistic floor for retail US equities. Options and small caps are much worse.
5. **Building your own backtester.** Fun, almost always a mistake.
6. **Skipping paper trading.** Unhedged ego is the most expensive factor.

---

## 5. Where I'd push back on the scope

Unless you have a specific reason:
- **Skip HFT / microstructure** as a first project. It demands C++, colocation, and data budgets that invalidate most of the above.
- **Skip building your own portfolio optimizer from scratch.** Use `cvxpy` + `PyPortfolioOpt`; save energy for research.
- **Skip custom time-series DB** until you have proof you need it.
- **Skip dashboards that look impressive but measure nothing decision-relevant.** A single daily email with 6 numbers > a 20-panel Grafana board you don't read.

---

## 6. What I need from you to sharpen this

Tell me:
1. Which asset classes (equities / futures / options / crypto)?
2. What frequency (daily / intraday / tick)?
3. Solo or team? Budget for data & cloud?
4. Is live trading a goal, or is this research/paper-only?
5. Are any of the 11 apps above out of scope, or are there apps I've missed that you had in mind?

Once I have those, I can turn this into a concrete milestone-by-milestone build plan with rough time estimates.
