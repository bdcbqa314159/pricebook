# PriceBook

Quantitative finance library for derivatives pricing and risk analytics. Independent verification of vendor pricing systems — built from first principles, validated against published research.

## Install

```bash
pip install pricebook
```

With visualisation support:
```bash
pip install pricebook[viz]
```

## Quick Start

```python
import pricebook

# Price a European option via Black-Scholes
from pricebook.models.engine_registry import price
result = price(spot=100, strike=100, vol=0.20, rate=0.04, T=1.0)
print(f"Price: {result.price:.4f}, Delta: {result.greeks.delta:.4f}")

# Build a discount curve
from pricebook.core.discount_curve import DiscountCurve
from datetime import date
curve = DiscountCurve.from_zero_rates(
    date(2024, 1, 1),
    [(1, 0.04), (2, 0.042), (5, 0.045), (10, 0.048)],
)

# Price a CDS
from pricebook.credit.cds import CDS, Frequency
cds = CDS(date(2024, 1, 1), date(2029, 1, 1), spread=0.01, notional=10_000_000)
```

## Features

**Pricing Engines**
- Monte Carlo: 15+ processes (GBM, Heston, SABR, Bates, Rough Bergomi, SLV), 6 variance reduction techniques, pathwise/LR/bump Greeks
- Trees: 5 binomial methods (CRR, JR, LR, Tian) + trinomial, BDT, implied tree, adaptive mesh
- PDE: Crank-Nicolson, ADI (Heston, SABR, 2-asset), PIDE (Merton, Kou), FEM, spectral (Chebyshev), adaptive refinement
- Unified protocol: switch engine with one parameter

**Fixed Income**
- Bonds, swaps, FRAs, caps/floors, swaptions, callable/puttable bonds
- Hull-White calibration, SABR smile, swaption vol cube
- 33 currency markets with dedicated conventions
- Bond futures (CTD, basis, delivery options), IR futures (SOFR, Euribor)

**Credit**
- CDS (single-name, index, basket, nth-to-default), CDS swaptions (European, Bermudan)
- CLN (vanilla, leveraged, floating, basket, stochastic recovery, XVA)
- CDO/CLO tranches, bespoke CDO, recovery waterfall
- Credit VaR (historical, parametric, copula), distressed debt analytics

**FX**
- Vanilla (Garman-Kohlhagen, 3 delta conventions), barriers (PDE + Vanna-Volga)
- Exotics: touch, lookback, Asian, TARF, accumulator, PRDC, autocallable
- SABR/SVI vol cube, SLV calibration, Merton/Bates jumps, regime switching
- FX forwards, swaps, NDFs, cross-currency basis

**Equity & Commodity**
- European/American (BSM, binomial, PDE), dividend modeling
- Autocallables, cliquets, mountain range (Napoleon, Everest, Atlas)
- Commodity: futures options, spread options (Kirk), seasonal vol, swing, tolling
- Variance/vol swaps, VIX futures

**Structured Products**
- MBS (PSA, CPR/SMM, OAS, IO/PO), ABS (auto, credit card, student loan), CMBS
- Range accruals, steepener/flattener, CMS, snowball, TARN
- Convertible bonds, capital-protected notes, reverse convertibles

**Portfolio & Risk**
- Mean-variance, Black-Litterman, HRP, CVaR optimisation, Kelly criterion
- Brinson attribution, Sharpe/Sortino/Calmar, tracking error
- ISDA SIMM, SA-CCR, Basel III capital
- Shapley value allocation, cooperative games

**Game Theory & Microstructure**
- Nash equilibrium, Stackelberg, bargaining (Nash, Rubinstein)
- Kyle lambda, Glosten-Milgrom, optimal execution (Almgren-Chriss)

## Testing

```bash
cd python
pip install -e ".[dev]"
pytest tests/ -n auto
```

11,000+ tests covering all modules.

## License

MIT — Bernardo Cohen / deLaPatada Software
