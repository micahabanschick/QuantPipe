# ETF Pairs Relative Momentum

> Within 8 related ETF pairs, go long the stronger asset by 12-1 month momentum.

## Origin

Adapted from the **OU Mean Reversion / Pairs Trading** strategy (QuantConnect).  
Academic basis: Chan (2008) pairs trading, Avellaneda & Lee (2010) OU processes,  
Kalman (1960) filter for dynamic hedge ratios.

## QuantPipe Adaptation

| Feature | Original | This Adaptation |
|---|---|---|
| Direction | Long one, short the other (dollar-neutral) | **Long-only** winner from each pair |
| Signal | OU Z-score deviation from equilibrium | Relative 12-1 month momentum within pair |
| Entry | Z-score > 2.0 from OU mean | Positive absolute momentum of winner |
| Exit | Z-score reverts to 0.25 | Monthly rebalance (implicit) |
| Hedge ratio | Kalman filter (dynamic) | No hedge ratio needed (long-only) |

**Important:** The OU mean-reversion framework requires short selling and is fundamentally  
incompatible with long-only constraints. This adaptation replaces mean reversion with  
**pair-wise relative momentum rotation** — going long the stronger asset in each pair.  
This is a trend-following strategy, not mean-reverting.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical window for the backtest |
| `top_n` | 5 | Max pairs held simultaneously |
| `cost_bps` | 7.0 | Round-trip cost (slightly higher for less-liquid pairs) |
| `weight_scheme` | `equal` | `equal` or `vol_scaled` (momentum-proportional) |

## Defined Pairs

| Pair | Relationship |
|---|---|
| EWA / EWC | Australia / Canada (commodity-linked economies) |
| GLD / GDX | Gold bullion / Gold miners |
| XLF / KBE | Broad financials / Regional banks |
| EWG / EWQ | Germany / France (Euro-area equity) |
| XLU / XLP | Utilities / Consumer staples (defensives) |
| TLT / IEF | 20-year / 7-year Treasuries (duration) |
| USO / XLE | Oil ETF / Energy sector |
| EEM / EFA | Emerging markets / Developed international |

## Tradability Assessment

**Tradability: MODERATE.**  
- Pairs cover diverse asset classes providing good diversification.  
- Several symbols (GDX, KBE, EWG, EWQ, USO) may not be in the default QuantPipe universe.  
- **Action required:** Add missing symbols to `config/settings.py` universe and run backfill.  
- Beta is low (~0.3–0.5) due to cross-asset pair selection.  
- Expected: Sharpe 0.5–0.9, Max DD 10–20%.

## Known Limitations

- True OU mean reversion (the original's edge) is lost — this is momentum, not mean reversion.
- Several symbols may require separate data ingestion if not in the default universe.
- Pairs that are cointegrated in the original may have different momentum dynamics.
