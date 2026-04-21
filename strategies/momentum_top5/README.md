# Momentum Top-5

> Cross-sectional 12-1 momentum on the ETF universe, equal-weight top-5 long-only.

## Strategy Overview

Ranks all symbols in the equity universe by their 12-month minus 1-month return (Jegadeesh & Titman momentum). Selects the top-5 ranked ETFs and holds them with equal weight, rebalancing monthly.

This is the **canary strategy** — the baseline against which all other strategies in this lab should be compared. Expected Sharpe: 0.8–1.2 over a 6-year backtest.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical period used for the backtest |
| `top_n` | 5 | Number of top-ranked ETFs to hold |
| `cost_bps` | 5.0 | Round-trip transaction cost in basis points |
| `weight_scheme` | equal | Position sizing — `equal` or `vol_scaled` |

## Signal

**12-1 Momentum** (`momentum_12m_1m`): return from 252 trading days ago to 21 trading days ago. Excluding the most recent month avoids short-term reversal contamination.

## Known Limitations

- Long-only: cannot short underperformers.
- Monthly rebalance at month-open: does not account for intra-month signal decay.
- Universe is ETFs, not individual stocks — factor concentration risk within each ETF.

## References

- Jegadeesh & Titman (1993) — *Returns to Buying Winners and Selling Losers*
- Asness, Moskowitz & Pedersen (2013) — *Value and Momentum Everywhere*
