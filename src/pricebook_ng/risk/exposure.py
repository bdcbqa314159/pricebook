"""Monte-Carlo expected-exposure engine (L5 risk & capital).

Produces the `ExposureProfile` that CVA consumes: `EE(t_j) = E[(V(t_j))^+]`, the
expected positive mark of a swap at future dates. For each grid date it draws the
Hull-White short rate under that date's t_j-forward measure — one exact Gaussian
(`model.forward_short_rate`), the same simulation the MC swaption uses — and
reprices the *remaining* swap analytically from the model's `zero_bond` (the swap
value is `notional - couponbond` for a payer, its negative for a receiver, exactly
as in the Jamshidian decomposition).

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
  oracle: sigma=0 deterministic exposure (exact) + discounted EE == co-terminal swaptions
  slice:  mc-exposure
"""

from __future__ import annotations

import random

from pricebook_ng.engine.swaption import coupon_bond_cashflows
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import VanillaSwap
from pricebook_ng.risk.xva import ExposureProfile

_CURVE_DC = DayCountConvention.ACT_365_FIXED


def expected_exposure(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig
) -> ExposureProfile:
    """MC expected positive exposure of `swap` under `model`, on a grid of the fixed
    coupon dates. `EE(t_j) = mean_paths max(+-(notional - couponbond(t_j; r)), 0)`
    with `r` drawn under the t_j-forward measure."""
    valuation = model.market.valuation_date
    dates, amounts, notional = coupon_bond_cashflows(swap)
    is_payer = swap.pay_fixed
    maturity = dates[-1]
    # exposure dates: valuation (mark today) + each coupon date strictly before
    # maturity (a non-empty remaining swap; exposure past the last flow is zero).
    grid = [valuation, *(d for d in dates if valuation < d < maturity)]

    rng = random.Random(numerics.mc_seed)
    ee: list[float] = []
    for t_j in grid:
        remaining = [(d, amt) for d, amt in zip(dates, amounts) if d > t_j]
        t = year_fraction(valuation, t_j, _CURVE_DC)
        total = 0.0
        for _ in range(numerics.mc_paths):
            r = model.forward_short_rate(t, rng.gauss(0.0, 1.0))
            coupon_bond = sum(amt * model.zero_bond(t_j, d, r) for d, amt in remaining)
            value = notional - coupon_bond if is_payer else coupon_bond - notional
            total += max(value, 0.0)
        ee.append(total / numerics.mc_paths)

    return ExposureProfile(tuple(grid), tuple(ee))
