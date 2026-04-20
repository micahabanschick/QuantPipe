# QuantFinance Pipeline — Tailored Build Plan

**Constraints recap:** solo developer, ≤$50/month total, daily frequency only, all asset classes (equities / futures / options / crypto), live trading is the goal, all 11 apps in scope.

---

## 1. What the budget forces

| Asset class | Realistic data source at budget | Live trading path |
|---|---|---|
| Equities (global) | EODHD EOD All World (~$22/mo) or Tiingo Power (~$10/mo, US only) | IBKR |
| Futures | Stooq / Yahoo continuous contracts free (research quality limited); Norgate Futures ~$25/mo (budget killer combined with stocks) | IBKR |
| Options | yfinance / IBKR live chains — **forward-only, no historical backtest** | IBKR |
| Crypto | CCXT + public exchange APIs — **$0, high quality** | CCXT → Binance / Coinbase / Kraken |

**The hard call:** rigorous historical options backtesting (vol surfaces, Greeks, bid-ask) costs $30–100/mo standalone, which breaks the budget. Recommended posture: treat options as a **live-only overlay** (e.g., covered calls / protective puts driven by signals from equities research) rather than a standalone quantitatively backtested strategy class.

**Cloud posture:** run locally. A $5/mo Hetzner/Vultr VPS for scheduled ingestion is fine as an upgrade later. Don't touch AWS — even "cheap" AWS will eat your budget in inscrutable ways.

---

## 2. Opinionated tooling stack

Pick these defaults and don't re-deliberate. You can revisit any of them after 6 months of real usage.

| Layer | Pick | Why |
|---|---|---|
| Language | Python 3.12+ | Everything integrates. C++ only if you hit a real bottleneck (you won't at daily). |
| Dep mgmt | `uv` | Fast, modern, handles everything `pip` + `venv` + `pip-tools` did. |
| Storage | Parquet + DuckDB | Free, zero ops, blazing fast on daily-scale data. |
| DataFrames | Polars (with pandas interop) | Faster than pandas, handles multi-GB data on a laptop. |
| Research | Jupyter via VS Code | No separate Jupyter server to manage. |
| Backtest | **VectorBT** for research; **custom thin live layer** for production | VectorBT is fast and flexible; NautilusTrader is better architecturally but much heavier — skip at your stage. |
| Portfolio opt | `cvxpy` + `PyPortfolioOpt` | Standard, maintained. |
| Broker (stocks/futures/options) | Interactive Brokers via `ib_insync` | Only realistic multi-asset retail broker with a solid Python API. |
| Broker (crypto) | CCXT | De facto standard, unified API across exchanges. |
| Dashboards | Streamlit | Local, free, fast to build. |
| Alerting | Pushover or Telegram bot | Cheap/free, works. Skip PagerDuty. |
| Scheduler | `cron` + shell scripts → later Dagster | Don't adopt Dagster before you need it. |
| Source control | GitHub (free) | Obvious. Use private repos. |
| CI (later) | GitHub Actions | Free tier is plenty for a solo project. |

---

## 3. Budget allocation (month-by-month)

**Months 1–2: $0.** Build everything on free data (yfinance, CCXT, Alpaca paper). Prove the pipeline works end-to-end before spending anything.

**Months 3–6: ~$22/mo.** Subscribe to EODHD EOD All World. This unlocks global equities with real fundamentals and cleaner data than yfinance.

**Months 6+: up to $50/mo.** Options:
- Add Norgate Futures (~$25/mo) if your research shows futures strategies are promising.
- Or add EODHD Options add-on ($30/mo) if you want options historicals — **you'll need to drop EODHD to a cheaper tier to stay in budget.**
- Or keep the slack for a VPS + a small premium data boost (e.g., Polygon Starter if US-focused).

Students get 50% off EODHD — mention that if applicable.

---

## 4. Milestone-by-milestone schedule

Times assume ~10–15 hours/week of solo time. Double them if you're working full-time elsewhere.

### Phase 0 — Setup (Week 0, ~5 hours)

- Create repo: `quantpipe/`. Suggested top-level dirs: `data_adapters/`, `storage/`, `features/`, `signals/`, `backtest/`, `portfolio/`, `risk/`, `execution/`, `reports/`, `orchestration/`, `research/`, `tests/`.
- Set up `uv` + pyproject.toml, pre-commit (ruff + black), pytest.
- Sign up (all free): IBKR paper account, Alpaca paper, Binance / Coinbase / Kraken (read-only API keys to start).
- Install `norgatedata` if you plan to add it later (works without subscription for setup).
- **Deliverable:** `hello world` script that imports from every top-level module without error.

### Phase 1 — Data layer (Weeks 1–3, App 1 + App 2 + start App 3)

- Write `DataAdapter` protocol: `get_bars(symbol, start, end, asset_class) → DataFrame`.
- Implement adapters: `YFinanceAdapter`, `CCXTAdapter`, `AlpacaAdapter` (free).
- Storage: Parquet under `data/{asset_class}/{freq}/date=YYYY-MM-DD/`. One writer, one reader (`load_bars(...)` powered by DuckDB).
- Ingestion script pulls daily bars for a seed universe: S&P 500 + top 20 crypto + a handful of ETFs. Runs nightly via cron.
- Basic validators: row counts, null rate, price jump >N sigma, stale data.
- **Deliverable:** 90 days of daily bars for ~500 symbols stored as partitioned Parquet, queryable in <1 second from a notebook.
- **Risk:** spend no more than 3 weeks here. It's tempting to perfect it; don't.

### Phase 2 — Features + data quality hardening (Weeks 4–5, App 3 + App 4)

- Adjustments table (splits, dividends) applied at read time via `get_adjusted_prices()`.
- Universe-as-of-date function. Hard-code a survivorship-bias-free S&P 500 list from a free source (e.g., Wikipedia historical constituents) for starters.
- Feature library: 5 features — `log_return`, `realized_vol_21d`, `momentum_12m_1m`, `dollar_volume_63d`, `reversal_5d`. Each a pure function, each with a snapshot test.
- Feature compute script: `compute_features(universe, dates, features) → Parquet`.
- **Deliverable:** a notebook that computes all 5 features over 5 years of SPX history in <60 seconds and serves as ground truth for future work.

### Phase 3 — Backtest + first signal (Weeks 6–8, App 5 + App 6)

- Integrate VectorBT. Strategy API: `(features, params) → target_weights_over_time`.
- **Canary backtest:** equal-weight top-decile 12-1 momentum, S&P 500 universe, monthly rebalance. Must reproduce the published ~8–12% annual excess return ballpark. If it doesn't, there's a bug in the pipeline — fix it here, not later.
- Add cost model: 5bps round-trip (conservative for US equities). Sharpe should still be positive.
- Walk-forward cross-validation helper.
- **Deliverable:** one strategy notebook with full pipeline: load data → compute features → generate signal → backtest → tearsheet.

### Phase 4 — Portfolio + risk (Weeks 9–10, App 7 + App 8)

- Portfolio app: pure function `(raw_signals, cov_matrix, constraints) → target_weights`. Start with vol-scaled top-decile long/short.
- Add Ledoit-Wolf shrinkage via scikit-learn.
- Risk app: historical VaR (1-day, 95%, last 252 days), gross/net/sector exposures, top-10 concentration.
- Hard pre-trade check function that rejects orders violating caps.
- **Deliverable:** every backtest run now produces a risk report alongside the tearsheet.

### Phase 5 — Reporting + orchestration (Week 11, App 10 + App 11)

- Streamlit dashboard #1 (ops health): last ingestion time per asset class, row counts, validation pass/fail, cron job status.
- Streamlit dashboard #2 (performance): daily P&L, drawdown, rolling Sharpe, exposures. Reads directly from Parquet.
- Cron chain: `ingest → validate → compute_features → generate_signals → rebalance → report`. Each step exits non-zero on failure; a wrapper script sends a Pushover alert if any step fails.
- **Deliverable:** wake up in the morning to either a green dashboard or a phone notification that something broke.

### Phase 6 — Paper trading loop (Weeks 12–15, App 9)

- `BrokerAdapter` protocol: `get_positions`, `get_cash`, `place_order`, `cancel_order`, `get_fills`.
- Implement `IBKRAdapter` (via `ib_insync`) and `CCXTAdapter`. Both implement the same protocol.
- `Trader` module takes target weights, current positions, and produces orders. Idempotent: rerunning with the same inputs should produce zero new orders.
- Reconciler: after each rebalance, compare internal positions to broker reported positions. Log drift.
- **Paper-trade the canary strategy for 4+ weeks** across equities (IBKR) + crypto (CCXT paper or small real). Track realized vs backtest: the difference is your "implementation gap."
- **Deliverable:** green reconciler for 4 consecutive weeks + a documented implementation gap ≤25% of backtested Sharpe.

### Phase 7 — Go live, tiny (Month 5+)

Gate criteria before risking real money (all must be true):
- Paper trading has been green for 4+ weeks.
- Implementation gap is understood and within acceptable bounds.
- Risk pre-trade checks are in place and tested (try to violate them; they should block).
- You have a written kill-switch procedure.
- You have a dashboard you actually check daily.

Then:
- Start with ~5–10% of the capital you're willing to eventually deploy. Whatever feels "too small to matter" — that's the right size.
- Double the size only after 4+ weeks of live trading matches paper-trading P&L within variance.
- **Never add a second strategy live until the first has been profitable for a full quarter.**

### Phase 8 — Expand (Month 6+, ongoing)

This is when you upgrade data sources (EODHD paid, maybe Norgate futures), add more asset classes to live trading, and start researching additional signals. The pipeline is now a tool, not a project.

---

## 5. Per-app difficulty with your constraints

Updated from the generic outline given multi-asset daily solo context:

| App | Baseline difficulty | Your constraint shifts it to | Note |
|---|---|---|---|
| 1. Ingestion | M | **M+** | Multi-asset means 3–4 adapters (yfinance, CCXT, IBKR, eventually EODHD). More code but each is small. |
| 2. Storage | L | L | Daily bars across all asset classes fit easily in Parquet. |
| 3. Data quality | H | **H** | Survivorship-bias-free universes across equities/futures is genuinely hard on your budget. Plan to cut corners in research and be honest about it. |
| 4. Features | M | **M+** | Features per asset class diverge (term structure for futures, greeks for options). Keep them in separate modules. |
| 5. Research | M / VH for edge | VH | Edge is always hard. Daily multi-asset is actually a more forgiving regime than HFT — momentum, carry, and value anomalies are well-documented. |
| 6. Backtest | H (build) / M (adopt) | **M** | Adopt VectorBT. Don't build. |
| 7. Portfolio | M | **M+** | Multi-asset needs cross-asset covariance and notional-vs-margin thinking for futures. |
| 8. Risk | M | **M+** | Same reason as portfolio. Futures margin and options delta/gamma exposures add complexity. |
| 9. Execution | M paper / H live | **H** | IBKR API has a learning curve. CCXT is easier. Idempotency and reconciliation are the hard parts. |
| 10. Reporting | M | M | Keep it simple. Don't build a "platform." |
| 11. Orchestration | M | **L** | cron is fine at your scale. Don't adopt Dagster/Airflow yet. |

---

## 6. The single most important warning

> **Do not go live with real capital until you have been running paper trading for at least 4 weeks with the full pipeline green.** At daily frequency with small account sizes, the expected value of rushing to live is negative. The kind of bugs that destroy accounts — duplicate orders, wrong sign, stale signals, reconciler drift — almost all surface within the first 4 weeks of paper trading if you're watching.

A $5,000 account trading aggressively can lose 20% to a single sign-flip bug before you see the dashboard. A 4-week paper trading period costs you nothing but time.

---

## 7. Realistic timeline summary

| Milestone | Cumulative time | Cumulative spend |
|---|---|---|
| End-to-end pipeline on free data | ~11 weeks | $0 |
| First paper trading live | ~15 weeks | $0 (or $22/mo if you've subscribed to EODHD for real data) |
| First real money, small size | ~20 weeks | ~$50–100 cumulative |
| Stable multi-asset live trading | ~30 weeks (~7 months) | steady-state $22–45/mo |

If any phase takes twice as long as estimated, that's normal. If any phase takes four times as long, something is architecturally wrong — stop and refactor before pushing forward.

---

## 8. Open questions I'd want to pin down next

When you're ready to actually start building, answering these will sharpen the first week of work:

1. **OS:** Windows, macOS, or Linux? (Affects Norgate feasibility — Norgate is Windows-only or requires a Windows VM.)
2. **Starting capital intended for live trading?** (Under $25k in the US triggers PDT rules, which pushes you toward swing trading + futures/crypto rather than intraday equities.)
3. **Do you have existing IBKR or crypto accounts?** (Several-week onboarding if not.)
4. **Are you comfortable using a paid backtesting platform if free options fall short?** (I'd say no, but worth checking.)
5. **Any strategy ideas you already want to build around?** (If yes, the research and feature work should be biased toward those; if no, start with the canonical momentum canary.)
