"""Monte-Carlo expected-exposure engine (L5 risk & capital).

Produces the `ExposurePair` that CVA and DVA consume: expected positive exposure
`EPE(t_j) = E[(V(t_j))^+]` and expected negative exposure `ENE(t_j) = E[(-V(t_j))^+]`,
both from one simulation. For each grid date it draws the Hull-White short rate
under that date's t_j-forward measure — one exact Gaussian (`model.forward_short_rate`,
the same simulation the MC swaption uses) — and reprices the *remaining* swap
analytically from the model's `zero_bond` (the swap value is `notional - couponbond`
for a payer, its negative for a receiver, exactly as in the Jamshidian decomposition).

Two consequences make the oracle exact-able:
- `sigma = 0` removes all randomness, so `EE(t_j)` is the deterministic forward
  swap value's positive part.
- Under the t_j-forward measure, `P(0,t_j) * EE(t_j)` is the co-terminal swaption
  expiring at t_j — so the discounted exposure IS a swaption strip, and feeding
  `EE(t_j)` to `cva` (which multiplies by `DF(t_j)`) yields the correct discounted
  expected exposure `Sum_j swaption(t_j) * dQ_j`.

Scope (like the MC swaption): a vanilla swap under a flat-curve HW with `a > 0`;
the exposure grid is the fixed-coupon dates. Path-dependent exposure, netting sets,
collateral (PFE), and non-swap products are later slices.

Provenance:
  quarry: python/pricebook/risk/ (exposure / xva)
  source: Brigo & Mercurio s.3.3 (T-forward simulation); Gregory, The xVA Challenge
  oracle: sigma=0 deterministic exposure (exact) + discounted EE == co-terminal swaptions;
          PFE quantile == V at the r-quantile (monotone transform)
  slice:  mc-exposure; bcva (EPE+ENE from one pass); pfe-quantile (PFE / dynamic-IM)
"""

from __future__ import annotations

import random

from pricebook_ng.engine.swaption import coupon_bond_cashflows
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import VanillaSwap
from pricebook_ng.risk.xva import ExposurePair, ExposureProfile

_CURVE_DC = DayCountConvention.ACT_365_FIXED


def _simulate_swap_values(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig
) -> tuple[list[date], list[list[float]]]:
    """Per exposure date, the list of simulated remaining-swap values `V(t_j)` across
    paths (`r` drawn under the t_j-forward measure, repriced from the model's `zero_bond`).
    Shared by the EE means and the PFE quantiles. Exposure dates: valuation (mark today)
    plus each coupon date strictly before maturity (past the last flow exposure is zero)."""
    valuation = model.market.valuation_date
    dates, amounts, notional = coupon_bond_cashflows(swap)
    is_payer = swap.pay_fixed
    maturity = dates[-1]
    grid = [valuation, *(d for d in dates if valuation < d < maturity)]

    rng = random.Random(numerics.mc_seed)
    values_by_date: list[list[float]] = []
    for t_j in grid:
        remaining = [(d, amt) for d, amt in zip(dates, amounts) if d > t_j]
        t = year_fraction(valuation, t_j, _CURVE_DC)
        values = []
        for _ in range(numerics.mc_paths):
            r = model.forward_short_rate(t, rng.gauss(0.0, 1.0))
            coupon_bond = sum(amt * model.zero_bond(t_j, d, r) for d, amt in remaining)
            values.append(notional - coupon_bond if is_payer else coupon_bond - notional)
        values_by_date.append(values)
    return grid, values_by_date


def _quantile(sorted_values: list[float], q: float) -> float:
    """The `q`-quantile of an ascending-sorted sample (linear interpolation, type 7)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(pos)
    if lo + 1 >= n:
        return sorted_values[-1]
    return sorted_values[lo] + (pos - lo) * (sorted_values[lo + 1] - sorted_values[lo])


def exposure_profiles(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig
) -> ExposurePair:
    """MC expected exposure of `swap` under `model`, on a grid of the fixed coupon
    dates — both sides from one simulation. Per path the remaining swap value is
    `V = +-(notional - couponbond(t_j; r))` with `r` drawn under the t_j-forward
    measure; `EPE = mean max(V, 0)` (feeds CVA), `ENE = mean max(-V, 0)` (feeds DVA)."""
    grid, values_by_date = _simulate_swap_values(swap, model, numerics)
    epe = [sum(max(v, 0.0) for v in values) / len(values) for values in values_by_date]
    ene = [sum(max(-v, 0.0) for v in values) / len(values) for values in values_by_date]

    g = tuple(grid)
    return ExposurePair(ExposureProfile(g, tuple(epe)), ExposureProfile(g, tuple(ene)))


def pfe_profile(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig, quantile: float
) -> ExposureProfile:
    """Potential future exposure at confidence `quantile`: `PFE_q(t_j)` is the q-quantile
    of positive exposure `max(V(t_j), 0)` across paths — the tail of the exposure the EPE
    averages over. A high-quantile PFE (e.g. 99%) also serves as a dynamic initial-margin
    proxy that feeds `mva`. (A margin-period-of-risk IM on the change in V is a refinement.)"""
    grid, values_by_date = _simulate_swap_values(swap, model, numerics)
    pfe = [
        _quantile(sorted(max(v, 0.0) for v in values), quantile) for values in values_by_date
    ]
    return ExposureProfile(tuple(grid), tuple(pfe))
