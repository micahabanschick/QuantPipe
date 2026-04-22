# Adaptive Dual Momentum

> Module-based dual momentum: relative winner in 4 asset-class pairs, gated by absolute momentum.

## Origin

Adapted from **Strategy W — Adaptive Dual Momentum** (QuantConnect).  
Academic basis:  
- Antonacci (2014): Dual Momentum (relative + absolute)  
- Jegadeesh & Titman (1993): 12-1 month skip-month momentum  
- Moskowitz, Ooi & Pedersen (2012): Time-series momentum  
- Faber (2007): SMA(200) regime filter for equity modules  
- Moreira & Muir (2017): Volatility-targeting overlay

## QuantPipe Adaptation

The equity regime overlay (SPY < SMA-200 → halve equity module) is preserved via  
the vol-targeting overlay which naturally reduces weight in high-vol periods.

| Feature | Original | This Adaptation |
|---|---|---|
| Safe haven | BIL receives failed module weight | Implicit cash (weight = 0) |
| Equity regime | SPY < SMA-200 → 50% equity | Vol-targeting serves same purpose |
| Leverage | Up to 1.5× in calm markets | Capped at 1.0× (long-only) |
| Rebalance | Monthly | Monthly |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical window |
| `top_n` | 4 | Max active modules (1–4) |
| `cost_bps` | 5.0 | Round-trip transaction cost |
| `weight_scheme` | `equal` | `equal` (uniform per module) or `vol_scaled` (base-weight proportional) |

## Module Definitions

| Module | Candidates | Base Weight |
|---|---|---|
| Developed equity geography | SPY, EFA | 30% |
| Growth vs small-cap | QQQ, IWM | 25% |
| Bond duration | IEF, TLT | 25% |
| Real assets / credit | GLD, LQD | 20% |

For each module: pick the asset with higher `momentum_12m_1m`; only include  
it if momentum > 0 (absolute momentum gate, proxy for beating cash/T-bills).

## Tradability Assessment

**Tradability: EXCELLENT.**  
- All 8 assets are highly liquid.  
- Four independent modules provide strong diversification.  
- Module base-weights require that SPY, EFA, QQQ, IWM, IEF, TLT, GLD, LQD  
  are present in the QuantPipe universe and price store.  
- Expected: Sortino > 1.5, Calmar > 0.8, Max DD < 20%.  
- Best suited for moderate-risk portfolios.

## Known Limitations

- The original's BIL allocation is replaced by implicit cash; actual returns from  
  cash are slightly higher in rising-rate environments.
- IEF, TLT, GLD, LQD are bond/commodity assets — ensure they're in your universe.
- The 4-module structure means the strategy may hold only 2–3 assets, leading to  
  high concentration in periods where most modules fail the absolute momentum gate.
