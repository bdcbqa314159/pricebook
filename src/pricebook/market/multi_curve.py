"""The forward-projection curve, bootstrapped against a known discount curve (L1).

Post-crisis, a collateralised trade discounts at the **collateral (OIS) rate** while each floating
leg projects its forwards off a **tenor-specific projection curve**. So a currency's curves become a
set: an OIS discount curve, and one projection curve per tenor calibrated *against* it.

**Half of that set needs no new code, and saying so is the point.** An OIS par swap discounts its
own cashflows, so its floating leg telescopes and the bootstrap equation is BERHAN 5.1.2 — which is
`nominal_bootstrap.nominal_curve_from_par_swaps` (tranche 8). The OIS curve is therefore that
function applied to OIS quotes, **reused verbatim**; this module holds only the genuinely new half,
where the discounting curve is a *different, already-known* curve and nothing telescopes.

**The relation, from LICHTERS-2015 §3.5** (read in `-raw` and `-layout` and reconciled, §4):

    c * A^ON_{0,n} = sum_{i=1..n} f^delta(t_{i-1}, t_i) * delta_i * df^ON_i

*"where `A^ON` is the fixed side's annuity based on OIS discount factors which are all known"* -
with the projection curve's own recursion and the forward definition (1.1):

    df^delta_{s+delta} = df^delta_s / (1 + delta * f^delta(s, s+delta))
    f(t, t+delta)      = (df_t / df_{t+delta} - 1) / delta

Solving pillar by pillar: the earlier forwards are known, so the last period's forward is whatever
makes the floating leg match the fixed leg, and the recursion turns it into the pillar's discount
factor. **Closed form per pillar, no root-find** - the same reason tranche 8 gives: given the earlier
pillars this is an exact rearrangement, and solving it numerically would be solving a closed form.

**What distinguishes the two eras is a telescoping identity, not a tolerance.** Eq (1.2) says
`f(t,t+delta) * df_{t+delta} * delta = df_t - df_{t+delta}`, so a floating leg sums to `1 - df_n`
**only when one curve both projects and discounts**. That is why the single-curve case is an exact
reduction (the sum collapses and §3.5 becomes BERHAN 5.1.1, so this bootstrap lands on tranche 8's
curve) and why it is also the **degenerate** fixture: a projection bootstrap that ignored the
discount curve and discounted on itself would pass the reduction and be wrong everywhere else.
Measured on the tranche's fixture: single-curve telescopes exactly, multi-curve differs by 1.2e-03.

**Single-curve remains a first-class mode, not a limit.** Nothing here replaces
`nominal_curve_from_par_swaps`; a snapshot holding one curve per currency is the pre-crisis era and
still prices, and a snapshot holding an OIS curve plus projection curves is the post-crisis one. The
arc's design invariant (`docs/modern_discounting_arc.md`): multi-curve is a sibling, not a
replacement.

Provenance:
  stock:  none - the archived predecessor discounted on one curve per currency
  source: LICHTERS-2015 (docs/references.md) §3.5 "Building the forward projection curves" - the
          swap-stripping relation, the projection recursion, and eq (1.1)/(1.2) of §1 for the
          forward and its PV identity; transcribed in both renderings and reconciled
  oracle: tests/L1/test_multi_curve.py - every calibrating swap reprices to par off (projection,
          OIS); the single-curve reduction reproduces tranche 8's curve (measured worst 1 ULP, so
          the committed claim is a relative tolerance, not bit-identity); a genuine-basis companion
          where the curves differ by ~100bp; the telescoping identity as the era discriminator; and
          the sequential dependency, that shifting an OIS quote moves the projection curve
  crosscheck: quantlib PiecewiseYieldCurve with an external discounting curve (multi-curve
          bootstrapping) - deferred to the optional QuantLib tier
  slice:  tranche 54 - the OIS / multi-curve bootstrap
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from pricebook.market.curve import CurveHandle, DiscountCurve
from pricebook.market.nominal_bootstrap import ParSwapQuote, swap_annuity
from pricebook.foundation.time_measure import TimeMeasure


def projection_curve_from_par_swaps(
    measure: TimeMeasure, quotes: Sequence[ParSwapQuote], discount: CurveHandle
) -> DiscountCurve:
    """The tenor's projection curve implied by `quotes`, discounted on `discount`.

    `discount` is the **OIS** curve and is read, never rebuilt - which is the whole content of the
    multi-curve construction and the thing the sequential-dependency oracle pins: perturb the
    discount curve and this curve's forwards move.

    Returns an ordinary `DiscountCurve`, so every consumer reads forwards out of it through `df`
    exactly as it reads the discount curve - there is no projection-specific curve type and no
    second forward path. (Lichters makes the same point: the projection curve *is* a discount curve,
    "the risky discount curve attached to tenor delta default risk", whose purpose is forwards.)
    """
    if not quotes:
        raise ValueError("a projection curve needs at least one par swap quote")

    factors: dict[date, float] = {measure.anchor: 1.0}
    times: list[float] = [0.0]
    rates: list[float] = []

    for quote in quotes:
        if quote.schedule.dates[0] != measure.anchor:
            raise ValueError(
                f"a par swap schedule must start at the curve anchor {measure.anchor}, got "
                f"{quote.schedule.dates[0]}"
            )
        # **One equation, one unknown.** A quote prices at par, so it determines exactly one new
        # pillar; a quote that skips a pillar leaves two unknowns in one equation, and solving them
        # sequentially would invent a value rather than strip one. Stated here because tranche 8's
        # own `_validate` admits such a set (its prefix check passes) and fails later with a
        # `KeyError` on the missing pillar - a precondition surfacing as a crash instead of a
        # refusal. Raised as a finding rather than patched here.
        unknown = [pay for _, pay, _ in quote.schedule.periods() if pay not in factors]
        if len(unknown) != 1:
            raise ValueError(
                f"the quote maturing {quote.schedule.dates[-1]} introduces {len(unknown)} unknown "
                f"pillars ({', '.join(str(d) for d in unknown) or 'none'}); a par quote determines "
                "exactly one, so the quote set must step one pillar at a time"
            )
        target = quote.par_rate * swap_annuity(discount, quote.schedule)
        accumulated = 0.0
        previous = measure.anchor
        # No contiguity check: `CouponSchedule.periods()` is built from `zip(dates, dates[1:])`, so
        # each period starts where the previous one paid **by construction**, and the first period's
        # start is the anchor check above. A guard here was written and the mutation sweep found it
        # unreachable - §4's third survivor kind, where the remedy is to delete the branch rather
        # than keep a check no test can exercise.
        for start, pay, accrual in quote.schedule.periods():
            if pay in factors:
                # An already-solved pillar: its forward is known, so it only contributes.
                forward = (factors[previous] / factors[pay] - 1.0) / accrual
            else:
                # The last unknown: whatever makes this swap price at par (§3.5), turned into a
                # discount factor by the projection recursion.
                forward = (target - accumulated) / (accrual * discount.df(pay))
                implied = factors[previous] / (1.0 + accrual * forward)
                if implied <= 0.0:
                    raise ValueError(
                        f"the quote maturing {pay} at {quote.par_rate} implies a non-positive "
                        f"projection discount factor ({implied}); the curve is unsolvable there"
                    )
                factors[pay] = implied
                years = measure.year_fraction(pay)
                times.append(years)
                rates.append(-math.log(implied) / years)
            accumulated += forward * accrual * discount.df(pay)
            previous = pay

    # The shortest pillar's zero rate extends back to the anchor - flat at the short end, as
    # `nominal_bootstrap` and `real_curve_from_breakeven` both do, and as a one-quote curve must.
    return DiscountCurve(measure, tuple(times), (rates[0], *rates))
