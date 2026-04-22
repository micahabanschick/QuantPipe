# Volatility-Scaled Sector Momentum (VAMS)

> VAMS = momentum / vol ranks sectors; regime filter shifts between concentrated bets and defensive.

## Origin

Adapted from **Strategy X — Volatility-Scaled Sector Momentum** (QuantConnect).  
Academic basis:  
- Barroso & Santa-Clara (2015): Volatility-managed momentum (VAMS)  
- Daniel & Moskowitz (2016): Momentum crash prediction via VAMS  
- Moskowitz, Ooi & Pedersen (2012): Time-series momentum  
- Jegadeesh & Titman (1993): 12-1 month signal  
- Moreira & Muir (2017): Volatility-targeting with leverage cap  
- Faber (2007): SMA(200) regime filter  
- Grossman & Zhou (1993): Proportional drawdown control

## QuantPipe Adaptation

| Feature | Original | This Adaptation |
|---|---|---|
| MACD gate | EMA(12) − EMA(26) > signal | 3-month momentum > 0 (MACD proxy) |
| Safe haven | IEF (55–70%) + BIL (30–45%) | Implicit cash (weights < 1.0) |
| Leverage | Up to 1.5× | Capped at 1.0× (long-only) |
| Rebalance | Weekly (Monday) | Monthly |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical window |
| `top_n` | 3 | Concentrated: top-3 sectors (bull regime) |
| `cost_bps` | 7.0 | Round-trip cost |
| `weight_scheme` | `equal` | `equal` or `vol_scaled` (VAMS-proportional) |

## Signal Logic

1. **VAMS score**: `score_i = momentum_12m_1m / max(realized_vol_21d, 5%)`.
2. **Three-gate eligibility** (all required):  
   - VAMS > 0 (positive momentum).  
   - Price > SMA-200 (from `prices_df`).  
   - 3-month return > 0 (MACD proxy).
3. **Regime classification** from SPY drawdown from 252-day high:  
   - DD ≥ −10%: **Bull** — top-3 sectors, vol-targeting, equity_pct scales up  
   - DD < −10%: **Warning** — 1 sector, equity_pct = 25%  
   - DD < −15%: **Critical** — no equity (implicit 100% cash)
4. **Vol-targeting**: scale weights to 15% annualised portfolio vol (max 1.0×).

## Tradability Assessment

**Tradability: GOOD.**  
- VAMS directly uses `momentum_12m_1m` and `realized_vol_21d` — both available in QuantPipe features.  
- Requires `prices_df` for SMA-200 filter and 3-month momentum.  
- Drawdown-based regime filter is more responsive than SMA alone.  
- Expected: Sortino > 1.2, CAGR > 15%, Max DD < 25%.  
- Aggressive-growth profile; most aggressive of the regime-filtered strategies.

## Known Limitations

- Concentration in top-3 sectors means severe drawdowns if SPY drawdown threshold is missed  
  by a few days (e.g., regime signal arrives 1 week after crash begins).
- Weekly rebalance in the original is important for the drawdown threshold to respond quickly.  
  Monthly rebalance increases lag significantly — the critical-regime filter may engage late.
- MACD replacement (3-month momentum) is a proxy; true MACD state cannot be maintained  
  across monthly `get_signal()` calls without persistent state storage.
