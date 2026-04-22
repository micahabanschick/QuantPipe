# Tactical Defense Momentum

> Hybrid offense/defense: composite dual momentum with 40% equity floor.

## Origin

Adapted from **Strategy C — Momentum with Tactical Defense** (QuantConnect).  
Academic basis:  
- Faber (2007): SMA(50/200) regime filter  
- Antonacci (2014): Dual momentum composite scoring  
- Moskowitz (2012): Time-series momentum  
- Same 4-component regime score as Regime-Adaptive Dynamic Allocation (RADA)

## QuantPipe Adaptation

Direct adaptation. The trailing stop (exit below SMA-50 intraday) is replaced  
by the monthly rebalance SMA-50 check in `get_signal()`.

| Feature | Original | This Adaptation |
|---|---|---|
| Equity floor | 40% (never fully defensive) | 40% |
| Equity ceiling | 100% | 100% |
| Composite score | 50%×12m + 30%×3m + 20% golden cross | Same (3m from prices_df) |
| Trailing exit | Daily SMA-50 check | Monthly SMA-50 filter in signal |
| Rebalance | Every 10 trading days | Monthly |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical window |
| `top_n` | 6 | Max equity positions |
| `cost_bps` | 8.0 | Higher cost for more frequent composition changes |
| `weight_scheme` | `equal` | `equal` or `vol_scaled` (score-proportional) |

## Signal Logic

1. **Regime score** (4-component, same as RADA):  
   Trend 40% · Breadth 25% · Volatility 20% · Momentum 15%.
2. **Equity allocation**:  
   `≥ 0.65` → 100% · `0.40–0.65` → linear 60%–100% · `≤ 0.40` → floor at 40%.
3. **Composite score** per candidate:  
   `0.50 × mom_12m + 0.30 × mom_3m + 0.20 × golden_cross_bonus`.
4. Qualification: positive 12-month momentum AND price > SMA-50.
   In bear regime (score < 0.40): SMA-200 filter is relaxed.

## Tradability Assessment

**Tradability: GOOD.**  
- The 40% equity floor prevents over-defensiveness — suitable for growth-oriented accounts.  
- Composite scoring with 3-month momentum and golden cross adds signal quality.  
- Requires `prices_df` to be injected (done automatically by backtest_runner).  
- Expected: Sharpe > 0.7, CAGR > 12%, Max DD < 25%.  
- Moderate-aggressive profile; more aggressive than RADA, less than Aggressive Concentrated.

## Known Limitations

- Monthly rebalance misses the original's daily trailing stop capability — drawdowns  
  during fast market crashes will be larger than in the original bi-weekly version.
- The 3-month momentum computation relies on `prices_df`; if not available, falls back  
  to 12-month momentum only (composite is then 100% × 12-month).
