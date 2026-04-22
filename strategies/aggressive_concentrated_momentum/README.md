# Aggressive Concentrated Momentum

> Fully-invested top-N by composite dual-momentum. No cash allocation. High drawdown risk.

## Origin

Adapted from the **Aggressive Concentrated Momentum** strategy (QuantConnect).  
Academic basis:  
- Jegadeesh & Titman (1993): 12-1 month skip-month momentum  
- Faber (2007): SMA-50 / SMA-200 trend filters as qualification gate  
- Three-factor composite: 12-month momentum + 3-month confirmation + golden-cross bonus

## QuantPipe Adaptation

Near-direct port. The bi-weekly rebalance becomes monthly.  
The daily trailing stop (exit below SMA-50) is replaced by the SMA-50 qualification  
filter applied at each monthly signal generation.

| Feature | Original | This Adaptation |
|---|---|---|
| Cash allocation | None (100% equity always) | None (equity_pct = 1.0) |
| Rebalance | Bi-weekly | Monthly |
| Trailing exit | Daily SMA-50 liquidation | Monthly SMA-50 filter |
| Universe | Sector ETFs + mega-cap stocks + thematic | QuantPipe equity universe |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical window |
| `top_n` | 6 | Concentrated: 6 positions |
| `cost_bps` | 8.0 | Slightly higher for concentrated turnover |
| `weight_scheme` | `equal` | `equal` or `vol_scaled` (score-proportional with 10% floor) |

## Signal Logic

1. For each ETF in the universe with `momentum_12m_1m > 0`:
   - Compute `mom_3m` from `prices_df` (3-month return, 63 trading days).
   - Check `price > SMA-50` (from `prices_df`). Fail → discard.
   - `golden_cross = 0.20` if SMA-50 > SMA-200, else 0.
   - `composite = 0.50 × mom_12m + 0.30 × mom_3m + golden_cross`.
2. Select top-`top_n` by composite score.
3. Fallback to positive-momentum assets if SMA filter leaves < 2 candidates.

## Tradability Assessment

**Tradability: GOOD — with high-risk acknowledgement.**  
- Simplest strategy in this set; very close to a momentum_top5 with larger positions.  
- 6-position concentration means single-stock risk if individual equities are in the universe.  
- No defensive allocation: drawdowns during market crashes will be severe (25–35%).  
- Expected: Sharpe > 0.8, Total Return > SPY, Max DD 25–35%.  
- Suitable only for accounts with high drawdown tolerance and long time horizons.

## Known Limitations

- The original includes individual mega-cap stocks (AAPL, NVDA, MSFT, etc.). QuantPipe  
  defaults to an ETF-only universe. Adding stocks significantly increases concentration risk.
- No cash allocation means no safe harbour during extended bear markets.
- Monthly rebalance misses the original's daily trailing stop; drawdowns will be larger.
