# Regime-Adaptive Dynamic Allocation (RADA)

> Regime-driven multi-asset ETF allocation: a 4-component market regime score
> sets the equity/cash split; skip-month momentum with a trend filter selects
> the top-N positions within that allocation.

---

## Strategy Overview

RADA dynamically adjusts how much of the portfolio is invested in equities versus
held as cash, based on a real-time estimate of market regime. When the regime score
is high (risk-on), up to 100 % is deployed; when it is low (risk-off), as little
as 0 % is deployed and capital sits in cash.

Within the equity allocation, RADA selects the top-N ETFs from a diversified
investable universe using **skip-month momentum** (Jegadeesh-Titman 1993), filtered
by a 50-day SMA trend requirement.

---

## Regime Score

The regime score is a weighted average of four components, all computed from
independent ETF price data:

| Component | Weight | Signal | Source |
|---|---|---|---|
| **Trend** | 40 % | SPY price vs SMA-50 and SMA-200 (3 binary sub-signals) | Faber (2007) |
| **Breadth** | 25 % | Fraction of breadth-only ETFs above their 50-day SMA | — |
| **Volatility** | 20 % | Inverse of SPY 63-day annualized vol vs low/high thresholds | Moreira & Muir (2017) |
| **Macro Momentum** | 15 % | SPY 6-month return mapped to [0, 1] | — |

The raw score is smoothed with an EWMA (λ = 0.70) to reduce whipsawing. The
smoother is **seeded from the first valid score** (not from 0.5) to avoid a
warm-up artifact in the first few months of the backtest.

### Regime → Equity Allocation

| Regime score | Equity % |
|---|---|
| ≥ 0.65 (Risk-On) | 85 % – 100 % (linear) |
| 0.35 – 0.65 | 30 % – 85 % (linear) |
| ≤ 0.35 (Risk-Off) | 0 % – 30 % (linear) |

---

## Investable Universe

18 ETFs, all liquid and tradeable since ≤ 2010:

| Bucket | Symbols |
|---|---|
| Sectors | XLK · XLV · XLF · XLE · XLI · XLU · XLP · XLY · XLC · XLB · XLRE |
| Broad Market | SPY · QQQ · IWM · EFA · VWO |
| Alternatives | TLT · GLD |

### Breadth-Only Universe (never held)

KRE · XBI · SMH · ITB · XRT · IBB — sub-industry ETFs used **only** to measure
breadth. Keeping them separate from the investable universe prevents circularity
(the breadth signal would otherwise inflate both regime and selection signals for
the same ETFs).

---

## Signal Construction

1. **Skip-month momentum** (`momentum_12m_1m`): return from 252 trading days ago
   to 21 trading days ago. Excluding the most recent month avoids short-term
   reversal contamination (Jegadeesh & Titman 1993).

2. **Trend filter**: a candidate must have its current price > 50-day SMA. Symbols
   failing this filter are excluded from selection. If too few pass, the filter
   is relaxed to ensure at least `top_n` candidates are considered.

3. **Ranking**: surviving candidates are ranked by skip-month momentum. The top-N
   are selected.

4. **Weighting** (`weight_scheme = "momentum"`): positions are sized proportional
   to their momentum score, subject to a minimum-weight floor of `MIN_EQ_W / top_n`.
   Final weights are rescaled to sum to `equity_pct`. The remaining
   `1 − equity_pct` is left as uninvested cash (implicit in the backtest engine).

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical data period for the backtest |
| `top_n` | 5 | Number of ETF positions to hold |
| `cost_bps` | 5.0 | Round-trip transaction cost in basis points |
| `weight_scheme` | `momentum` | `"equal"` or `"momentum"` (proportional to score) |

### Frozen Regime Parameters

These are set to academic prior values and should **not** be re-tuned on in-sample
data. Doing so would introduce look-ahead bias.

| Parameter | Value | Reference |
|---|---|---|
| `SMA_FAST` | 50 | Faber (2007) |
| `SMA_SLOW` | 200 | Faber (2007) |
| `VOL_LOOKBACK` | 63 days | Moreira & Muir (2017) |
| `VOL_LOW` | 12 % annualized | Moreira & Muir (2017) |
| `VOL_HIGH` | 25 % annualized | Moreira & Muir (2017) |
| `MOM_REGIME` | 126 days (6 months) | — |
| `REGIME_SMOOTH` | 0.70 (EWMA λ) | — |
| `RISK_ON` | 0.65 | — |
| `RISK_OFF` | 0.35 | — |

---

## Bias Fixes vs Naive Momentum

| # | Fix | Problem addressed |
|---|---|---|
| 1 | ETF-only universe | No individual stock survivorship bias; no look-ahead from picking ex-post winners (e.g., NVDA, LLY) |
| 2 | Parameters frozen to academic priors | Eliminates in-sample curve-fitting of SMA windows, vol thresholds, etc. |
| 3 | Independent breadth universe (sub-industry ETFs) | Prevents circularity where breadth and selection signals feed back into each other |
| 4 | Skip-month momentum (12-1, not 12-0) | Avoids short-term reversal contamination |
| 5 | EWMA smoother seeded from first valid score | Eliminates warm-up artifact (0.5 pseudo-signal) in early backtest months |
| 6 | Regime reference is SPY (breadth is SECTORS + SUB_IND) | Prevents regime signal from being tautologically correlated with investable universe |

---

## Known Limitations

- **Long-only**: cannot short underperformers or hedge with inverse ETFs.
- **Monthly rebalance**: the signal is checked monthly; intra-month regime
  deterioration is not acted upon.
- **Implicit cash**: the cash buffer earns no return in the backtest. In live
  trading, this would typically be deployed in BIL or a money market fund.
- **ETF factor concentration**: each sector ETF aggregates many individual
  stocks — concentration within an ETF is not captured.
- **Regime lag**: the EWMA smoother adds useful noise reduction but also
  introduces a 1–3 month lag in detecting regime transitions.
- **No short-term alpha**: RADA is a macro allocation strategy; it does not
  exploit earnings, flows, or sentiment signals.

---

## References

- Faber, M. T. (2007) — *A Quantitative Approach to Tactical Asset Allocation*
- Jegadeesh, N. & Titman, S. (1993) — *Returns to Buying Winners and Selling Losers*
- Moreira, A. & Muir, T. (2017) — *Volatility-Managed Portfolios*
- Asness, C., Moskowitz, T. & Pedersen, L. (2013) — *Value and Momentum Everywhere*
