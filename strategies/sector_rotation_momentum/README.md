# Sector Rotation Momentum

> Long-only rotation across 11 SPDR sector ETFs by 12-1 month momentum with regime filter.

## Origin

Adapted from the **Dollar-Neutral Sector Rotation** strategy (QuantConnect).  
Academic basis: Moskowitz & Grinblatt (1999) sector momentum, Faber (2010) SMA regime filter.

## QuantPipe Adaptation

| Feature | Original | This Adaptation |
|---|---|---|
| Direction | Long top-3, short bottom-3 (dollar-neutral) | **Long-only** top-N sectors |
| Short leg | Yes — bottom-N sectors | Dropped; remaining capital = implicit cash |
| Rebalance | Weekly | Monthly |
| Regime | SPY < SMA-200 → reduce allocation | SPY < SMA-200 → halve top_n, 50% equity |

The short leg is not implementable in QuantPipe's long-only framework.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical window for the backtest |
| `top_n` | 3 | Sectors held in normal (bull) regime |
| `cost_bps` | 5.0 | Round-trip transaction cost (basis points) |
| `weight_scheme` | `equal` | `equal` or `vol_scaled` (momentum-proportional) |

## Universe

11 SPDR sector ETFs: XLK, XLV, XLF, XLE, XLI, XLU, XLP, XLY, XLC, XLB, XLRE.  
Symbols must be present in the QuantPipe universe and price store.

## Signal Logic

1. Rank all sector ETFs by `momentum_12m_1m` (descending).
2. Check SPY vs 200-day SMA for regime.
3. Select top `effective_top_n` (= `top_n` in bull; `top_n // 2` in bear).
4. Weights sum to 1.0 (bull) or 0.5 (bear).

## Tradability Assessment

**Tradability: GOOD.**  
- All 11 sectors are highly liquid, low-cost ETFs.  
- Monthly rebalance keeps turnover reasonable.  
- Regime filter meaningfully reduces drawdown.  
- Without the short leg, beta ≈ 0.5–0.8 (vs ~0 in the original).  
- Expected: Sharpe 0.6–1.0, Max DD 15–25%, Beta 0.5–0.8.

## Known Limitations

- Losing the short leg removes market-neutrality; returns will correlate with SPY.
- 11 sectors covers only US equities — no international or alternative exposure.
- Monthly rebalance may miss fast-moving sector rotations (original was weekly).
