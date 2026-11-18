"""The collateralised swap's simulated mark, LGM-driven, in either discounting era (L5).

The wiring the collateralised-discounting arc was built toward: the **LGM (tranche 57)** simulates
the rate state, this module marks the **collateralised interest-rate swap (tranche 55)** on every
path, and the result is an ordinary `SimulatedExposure` - so **`collateral.py` (29)** takes the
margin-period gap and **`mc_exposure_cva` (26)** prices the CVA, both unchanged. A CSA changes
*what* is exposed, not *how* the adjustment is computed, and that tranche-29 design is reused whole.

**Why a new surface at all.** `risk/mc_exposure.swap_values_on_paths` already values *a* swap on
simulated paths - but it is LMM-specific: it takes forward-curve paths, an `LmmTenor` and a
`SwapSpan`. It cannot value a tranche-55 `InterestRateSwap` on LGM states, so this is the
"extract a buried step into named surface" shape the tranche flagged, not a duplicate.

**THE MODELLING ASSUMPTION, stated rather than buried: the tenor basis is frozen.** The LGM evolves
one curve. In the multi-curve era a floating leg also needs a *projection* curve at time `t`, and
what this module assumes is the standard one - that today's forward basis is deterministic, so

    P_proj(t,T) = P_lgm(t,T) * [P_proj(0,T)/P_proj(0,t)] / [P_ois(0,T)/P_ois(0,t)]

the LGM-evolved discount bond times today's basis factor. It is a *choice*, not a transcription, and
the alternative (a second stochastic factor for the basis) is a different model that nothing here
pulls. It has one property worth having: **in single-curve mode the basis factor is identically 1**,
so one code path serves both eras and the arc's era invariant holds **by construction** rather than
by a branch - the same structural guarantee tranche 55's pricer gets.

**The era is read off the snapshot, never required.** A snapshot carrying a projection curve for the
swap's tenor is the post-crisis era; one carrying only a discount curve is the pre-crisis era, and
the swap projects off that. Collateralisation and the discounting era are **separable**: the CSA
mechanics (the margin-period gap, the threshold) are what make an exposure collateralised, and the
discount source stays a parameter. A single-curve collateralised exposure is meaningful - it is the
pre-crisis analog - so this engine must never make a `CurveSet` mandatory, which is the design
constraint ruled for this tranche and the thing its invariant oracle guards.

Provenance:
  stock:  none - the archived predecessor had no collateralised exposure
  source: LICHTERS-2015 (docs/references.md) Part II/III (collateralised exposure under a CSA; the
          LGM as the exposure engine's rate model, ch 11.1) with CREPEY-2026's margin-period gap
          frame, already held and implemented at tranche 29
  oracle: tests/L5/test_collateralised_capstone.py - the threshold's two reductions (to tranche
          29's gap at H=0, exactly; to the uncollateralised exposure as H grows) with a live-H
          companion asserting monotonicity, the ERA INVARIANT (the same swap's collateralised CVA
          in both modes, agreeing where the eras coincide and differing on a real basis), the mark
          tying back to tranche 55's own price, and collateral never increasing the charge
  crosscheck: quantlib - deferred to the optional QuantLib tier
  slice:  tranche 58 - the collateralised-discounting arc's capstone
"""

from __future__ import annotations

import math

from pricebook.foundation.schedule import CouponSchedule
from pricebook.market.curve import CurveHandle
from pricebook.market.curve_set import ProjectionKey
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.hull_white import HullWhiteParams
from pricebook.models.lgm import lgm_zero_bond, simulate_lgm_states
from pricebook.products.interest_rate_swap import InterestRateSwap
from pricebook.risk.mc_exposure import SimulatedExposure


def _basis_factor(
    projection: CurveHandle, discount: CurveHandle, start: float, pay: float
) -> float:
    """Today's forward tenor basis over `[start, pay]`, as the factor that turns the LGM-evolved
    discount ratio into the projection ratio:

        P_proj(t,start)/P_proj(t,pay) = [P_lgm(t,start)/P_lgm(t,pay)] * basis

    so the factor is `(P_proj(0,start)/P_proj(0,pay)) / (P_disc(0,start)/P_disc(0,pay))`. The
    **orientation matters and the reciprocal is a real bug**: built the other way up it under-states
    the floating leg and flipped this fixture's mark from +122,861 to -204,826, which is how it was
    caught.

    Identically `1.0` when the two curves are the same object - the single-curve era, where the
    basis does not exist - so one code path serves both.
    """
    if projection is discount:
        return 1.0
    return (projection.df_at(start) / projection.df_at(pay)) / (
        discount.df_at(start) / discount.df_at(pay)
    )


def swap_exposure_on_lgm(
    swap: InterestRateSwap,
    market: MarketSnapshot,
    params: HullWhiteParams,
    sampling: MonteCarloConfig,
    margin_period: float | None = None,
) -> SimulatedExposure:
    """`V(t)` per path on the margin-call grid, marking `swap` on LGM states.

    The grid is the swap's own payment dates (or `margin_period`-spaced, which must agree with
    them): an exposure date strictly inside a period would need that period's **fixing**, which the
    LGM state does not carry, so it is refused rather than approximated - the same choice
    `collateralized_exposure` makes about an interpolated collateral balance.
    """
    currency = swap.notional.currency
    discount = market.curves.discount(currency)
    key = ProjectionKey(currency, swap.tenor)
    projection = (
        market.curves.projection(currency, swap.tenor)
        if key in market.curves.projections
        else discount
    )

    measure = discount.time_measure
    periods = [
        (measure.year_fraction(start), measure.year_fraction(pay), accrual)
        for start, pay, accrual in swap.schedule.periods()
    ]
    boundaries = [periods[0][0]] + [pay for _, pay, _ in periods]
    # Exposure dates: every schedule boundary strictly inside the trade's life.
    times = tuple(t for t in boundaries[1:-1] if t > 0.0)
    if not times:
        raise ValueError("the swap is too short to carry an exposure profile")
    if margin_period is not None:
        for earlier, later in zip((0.0,) + times, times):
            if not math.isclose(later - earlier, margin_period, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"an exposure date would straddle a coupon period: the schedule's dates are "
                    f"{times} but the margin period is {margin_period}; a straddling date needs a "
                    "fixing the LGM state does not carry"
                )

    states = simulate_lgm_states(params, times[-1], sampling)
    # One draw per path at the LAST date would mis-state the earlier ones, so each date is drawn
    # on its own: the state is Gaussian with a deterministic variance, so each is exact.
    per_date = [simulate_lgm_states(params, t, sampling) for t in times]

    values: list[tuple[float, ...]] = []
    for path in range(sampling.paths):
        row: list[float] = []
        for index, observed in enumerate(times):
            state = per_date[index][path]
            floating = 0.0
            fixed = 0.0
            for start, pay, accrual in periods:
                if pay <= observed:
                    continue
                bond = lgm_zero_bond(params, observed, pay, state, discount)
                start_bond = lgm_zero_bond(params, observed, start, state, discount)
                forward = (
                    start_bond
                    * _basis_factor(projection, discount, start, pay)
                    / bond
                    - 1.0
                ) / accrual
                floating += forward * accrual * bond
                fixed += swap.fixed_rate * accrual * bond
            row.append(
                swap.notional.amount * swap.direction.omega * (floating - fixed)
            )
        values.append(tuple(row))
    return SimulatedExposure(times, tuple(values))
