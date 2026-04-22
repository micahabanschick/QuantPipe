# Protective Breadth Momentum

> PAA breadth count scales equity exposure; top-K assets by inverse-vol weighting with vol-targeting overlay.

## Origin

Adapted from **Strategy V — Protective Breadth Momentum** (QuantConnect).  
Academic basis:  
- Keller & Keuning (2016): Protective Asset Allocation (PAA) / breadth momentum  
- Maillard, Roncalli & Teiletche (2010): Inverse-vol (risk parity) weighting  
- Moreira & Muir (2017): Volatility-targeting overlay  
- Faber (2007): SMA-relative momentum as trend measure

## QuantPipe Adaptation

This strategy translates cleanly to QuantPipe. Key changes:

| Feature | Original | This Adaptation |
|---|---|---|
| Safe haven | Allocates to BIL/IEF/TIP by SMA momentum | Implicit cash (weights sum < 1.0) |
| Rebalance | Monthly | Monthly |
| Universe | 12 risky + 3 safe-haven assets | QuantPipe equity universe |
| Anti-churn | Yes (2% drift threshold) | Not implemented (monthly rebalance handles it) |

The "bond fraction" from the PAA formula becomes implicit cash — the backtest engine  
leaves uninvested capital earning 0% return (conservative but correct for backtesting).

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical window |
| `top_n` | 6 | TOP_K: max risky assets to hold |
| `cost_bps` | 5.0 | Round-trip transaction cost |
| `weight_scheme` | `equal` | `equal` or `vol_scaled` (inverse-vol weighting) |

## Signal Logic

1. **Breadth count**: count symbols with `momentum_12m_1m > 0`.
2. **Bond fraction** (PAA, PF=2): `BF = clip((N − N_pos) / (N − 0.5·N), 0, 1)`.
   PF=2 means aggressive protection: 50% positive assets required for full equity.
3. **Equity fraction**: `equity_pct = 1 − bond_frac`.
4. **Selection**: top-K assets by positive momentum (momentum-ranked).
5. **Vol-targeting**: scale equity_pct so portfolio vol ≈ 8% annualised.

## Tradability Assessment

**Tradability: EXCELLENT.**  
- Adapts directly to QuantPipe features without price-level data.  
- Breadth-based protection is robust and academically validated.  
- Inverse-vol weighting available when `weight_scheme = vol_scaled`.  
- Expected: Sharpe 0.8–1.3, Max DD 5–12%, Sortino > 1.2.  
- Most conservative of the 7 strategies — suitable for capital preservation.

## Known Limitations

- The "safe haven rotation" into bonds (BIL/IEF/TIP) is replaced by implicit cash.
  In real trading, you'd want to explicitly hold BIL during high-protection periods.
- PF=2 (aggressive protection) may keep the strategy too defensive in prolonged bull markets.
  Reduce PF to 1 via the `DEFAULT_PARAMS` if you want more equity participation.
