"""Market Model II's calibration (L4). BERHAN-2012 §5.4 — the tranche that closes the book.

Two stages, and only the second is a fit:

```
§5.4.1  σn,i = the quoted Black ATM cap volatility        an IDENTITY, not a fit
§5.4.2  {σ_I,i} fitted to the ATM floor strip             σ_I,1 bootstrapped from the 1Y floor
```

**§5.4.1 is an identity and that is the section's whole content.** *"By the log-normal dynamics of
`Fn(·,Ti)` we get an automatic calibration to the quoted (Black) cap volatility `σ_ATM_i`."* MM-II
models the forward index, and its nominal input is the forward *rate*'s volatility — which is
exactly what a Black cap quote states. So `nominal_vol_from_black_atm` is the identity function with
a validity check, and treating it as a residual to minimise would be a mis-reading. Contrast
§5.3.1, where JY needs a Hull-White cap engine and a genuine two-parameter least squares (tranche
10) — that difference is one of the thesis's stated advantages for MM-II.

**§5.4.2's bootstrap is structural, and this library can show why.** *"σ_I,1 is directly obtained
from the 1-Year Floor, since it depends on no other unknown parameters."* For the first period the
lead `T₀ − t` is zero, so `Di` vanishes (it carries the lead as a factor) and `V²` collapses to
`σ_I,1²·(T₁ − T₀)`. One unknown against one price — so it inverts by a plain Black inversion, reusing
tranche 9's `black_implied_vol` rather than entering a solver at all.

**Why the expiry dimension is free, which is the thesis's own argument for MM-II.** *"Since each
expiry has it's own set of parameters, there is no need to restrict the number of contracts in the
expiry dimension."* Compare tranche 11: JY's §5.3.2 frees six *global* parameters against the whole
surface and three of them are structurally unidentifiable. MM-II's are per-expiry, so an added
expiry adds a parameter *and* a quote. The oracle tests that claim directly — a short strip's fitted
vols must survive unchanged when later expiries are added.

**What it does NOT fix: the strike dimension.** `V²` contains no strike, so MM-II's implied smile is
flat and *"the skew in the strike dimension results in a poor fit if we want to include non ATM
contracts … OTM contracts are underpriced and ITM contracts are overpriced."* That signed direction
is oracled; it is the same failure JY has, and the thesis says so.

Provenance:
  stock:  — the quarry had no forward-CPI market model and so no calibrator for one
  source: BERHAN-2012 (docs/references.md) §5.4.1 (the σn identity), §5.4.2 (the per-expiry set, the
          iterative procedure, the 1-Year-Floor bootstrap, and the OTM-underpriced / ITM-overpriced
          direction); ch 6's conclusions for the model comparison this tranche measures
  oracle: tests/L4/test_mmii_calibration.py — σn as an exact identity round-trip; the zero-lead
          structure that makes the 1Y bootstrap valid, asserted before being relied on, plus the
          bootstrap proved blind to every later parameter; per-expiry identifiability by the tranche
          11 behavioural method (truth among the starts, distance-from-truth beside spread, no
          condition number); the expiry dimension proved free; the skew failure with the source's
          signed direction and the companion that the ATM fit itself is good; the tranche 13
          recursion refusal shown reachable; failure as a value
  crosscheck: none — QuantLib has no inflation forward-CPI market-model calibrator; the independent
          side is the synthetic-truth round trip
  slice:  tranche 14 — Market Model II's calibration, closing BERHAN ch 1–5
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from pricebook.engine.capfloor import price_inflation_capfloor
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.solvers import least_squares_fit
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black import black_implied_vol
from pricebook.models.market_model_ii import MarketModelII, MMIICorrelations, MMIIParams
from pricebook.products.inflation import InflationCapFloor

Quote = tuple[InflationCapFloor, float]

_FLOOR = 1e-10   # a volatility must be strictly positive to be fitted
_BP = 1.0e4      # residuals in basis points of notional, as in tranches 10(b) and 11


@dataclass(frozen=True)
class MMIICalibration:
    """A completed §5.4.2 fit. `params` is a complete `MMIIParams`, so it is directly a model's
    input — no unpacking step in which a caller could pair a fitted vol with the wrong correlation
    set."""

    params: MMIIParams
    rms_residual: float
    n_quotes: int
    iterations: int


def nominal_vol_from_black_atm(black_atm_vol: float) -> float:
    """§5.4.1: `σn,i` **is** the quoted Black ATM cap volatility.

    The identity function, and deliberately written as one rather than folded into the caller. It
    is a *claim about the model* — that MM-II's nominal input is the forward rate's own lognormal
    volatility, which a Black cap quote states directly — and claims deserve a name and an oracle.
    Compare §5.3.1, where JY must fit `(κn, σn)` through a Hull-White cap engine (tranche 10)."""
    if black_atm_vol < 0.0:
        raise ValueError(f"a Black ATM cap volatility cannot be negative, got {black_atm_vol}")
    return black_atm_vol


def index_vol_from_first_floor(
    floor: InflationCapFloor, quote: float, market: MarketSnapshot
) -> float:
    """`σ_I,1` from the 1-Year floor alone — §5.4.2's bootstrap, by Black inversion.

    Valid because the first period's lead is zero: `Di` vanishes and `V²` is `σ_I,1²·(T₁−T₀)`, so
    one unknown faces one price. No solver and no other parameter is involved, which is the oracle's
    assertion as much as the source's."""
    periods = list(floor.schedule.periods())
    if len(periods) != 1:
        raise ValueError(
            f"the bootstrap reads a ONE-period floor (the 1-Year), got {len(periods)} periods"
        )
    start, pay, accrual = periods[0]
    currency = floor.notional.currency
    nominal = market.curves.discount(currency)
    real = market.curves.real(currency)
    measure = nominal.time_measure
    if measure.year_fraction(start) != 0.0:
        raise ValueError(
            f"the bootstrap needs a floor starting at the curve anchor, got {start}"
        )

    span = measure.year_fraction(pay)
    forward = (nominal.df(start) / nominal.df(pay)) * (real.df(pay) / real.df(start))
    black_value = quote / (floor.notional.amount * accrual * nominal.df(pay))
    # `black_implied_vol` returns `sqrt(V^2 / variance_years)`, and here `V^2 = sigma_I^2 * span`,
    # so passing `span` as the horizon returns `sigma_I` directly.
    return black_implied_vol(black_value, forward, 1.0 + floor.strike, span,
                             CapFloorKind.FLOOR)


def _params_from(
    quotes: Sequence[Quote], nominal_vol: float, correlations: MMIICorrelations,
    vols: Sequence[float],
) -> MMIIParams:
    """Assemble a term structure from the per-expiry vector, keyed by each quote's own pay date."""
    maturities = [q[0].schedule.dates[-1] for q in quotes]
    return MMIIParams(dict(zip(maturities, vols)), nominal_vol, correlations)


def _strip_pvs(
    quotes: Sequence[Quote], market: MarketSnapshot, params: MMIIParams
) -> list[float] | PricingFailure:
    model = MarketModelII(market, params)
    pvs: list[float] = []
    for product, _ in quotes:
        out = price_inflation_capfloor(product, model)
        if not isinstance(out, PricingResult):
            return PricingFailure(f"a quoted floor cannot be priced: {out.reason}")
        pvs.append(out.pv.amount)
    return pvs


def calibrate_mmii_to_atm_floors(
    quotes: Sequence[Quote],
    market: MarketSnapshot,
    black_atm_vol: float,
    correlations: MMIICorrelations,
    start: tuple[float, ...] | None = None,
) -> MMIICalibration | PricingFailure:
    """Fit MM-II's index-vol term structure to a strip of ATM floor quotes — §5.4.2.

    `black_atm_vol` is **read**, not fitted (§5.4.1), and the correlations are inputs here: the
    free set is the index-vol term structure, one point per quoted expiry, which is what makes the
    expiry dimension free. Quotes are **nested** floors, so quote `i` adds one period and one
    unknown."""
    if not quotes:
        return PricingFailure("cannot calibrate without at least one ATM floor quote")
    try:
        nominal_vol = nominal_vol_from_black_atm(black_atm_vol)
    except ValueError as exc:
        return PricingFailure(f"cannot calibrate Market Model II: {exc}")

    opening = tuple(0.012 for _ in quotes) if start is None else tuple(start)
    if len(opening) != len(quotes):
        return PricingFailure(
            f"the start vector has {len(opening)} points for {len(quotes)} quotes"
        )
    if any(v <= 0.0 for v in opening):
        return PricingFailure(f"the starting index volatilities must be positive, got {opening}")

    # Price once at the start point, so a missing curve is a value before the solver is entered
    # rather than a KeyError escaping mid-iteration (tranche 6's precedent).
    first = _strip_pvs(quotes, market, _params_from(quotes, nominal_vol, correlations, opening))
    if isinstance(first, PricingFailure):
        return first

    targets = [q for _, q in quotes]

    def residuals(x: Sequence[float]) -> list[float]:
        pvs = _strip_pvs(quotes, market, _params_from(quotes, nominal_vol, correlations, x))
        if isinstance(pvs, PricingFailure):
            return [math.inf] * len(targets)
        return [(pv - target) * _BP for pv, target in zip(pvs, targets)]

    lower = tuple(_FLOOR for _ in quotes)
    upper = tuple(math.inf for _ in quotes)
    outcome = least_squares_fit(residuals, opening, bounds=(lower, upper))
    if not outcome.converged:
        return PricingFailure(
            f"the Market Model II index-vol fit did not converge from {opening} "
            f"on {len(quotes)} quotes"
        )

    rms = math.sqrt(2.0 * outcome.cost / len(quotes)) / _BP
    return MMIICalibration(
        params=_params_from(quotes, nominal_vol, correlations, outcome.x),
        rms_residual=rms,
        n_quotes=len(quotes),
        iterations=outcome.iterations,
    )
