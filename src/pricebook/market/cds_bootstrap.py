"""The survival curve bootstrapped from a strip of par CDS quotes (L1). `OKANE-2008` ch 7.

At each maturity `Tₘ`, shortest first, solve for the forward hazard `λₘ` on `(Tₘ₋₁, Tₘ]` such that

    V(CDSₘ) = Protection(Tₘ) − S_quote(Tₘ)·RPV01(Tₘ) = 0                      (ch 7.5)

with `λ₁…λₘ₋₁` already fixed — piecewise-flat forward hazard (ch 7.4), the same interpolation
family as tranche 8's piecewise-flat forward rate, and the same sequential shape.

**This is L1 because of what it produces.** It yields *market data* — a `SurvivalCurve` — from
quotes, using only L0 and L1 (a schedule, two curves, a scalar). It never calls the L4 `price()`
engine; doing so would make it an engine-inverting L4 calibrator (the Jarrow-Yildirim-to-floor-strip
case). It does curve arithmetic, as `nominal_bootstrap` does, through the leg functional both it and
the engine now share.

**A root-find rather than a closed form, unlike tranche 8.** T8 deliberately refused a solver
because eq 5.1.2 is an exact rearrangement for the next pillar. Here there is no rearrangement:
`λₘ` enters the protection integral inside `exp(−∫λ)` *and* the premium leg's survivals, so `V = 0`
is genuinely implicit. `V` is strictly increasing in `λₘ` (more default risk raises the protection
leg and lowers the risky annuity, both pushing `V` up), which is what makes a bracketed solve the
right instrument and gives the arbitrage test below its exact form.

**Arbitrage shows up as a bracket that cannot straddle, and it is refused in domain vocabulary.**
Since `V` increases in `λₘ`, a root with `λₘ ≥ 0` exists if and only if `V(λₘ=0) ≤ 0`. So a quote
too low to be consistent with the shorter maturities — which would require a *negative* forward
hazard, a survival curve that rises — is detected exactly, by a sign test and not a tolerance, and
raised here rather than left for `bracketed_root` to report as a caller bug (ch 7.7).

Provenance:
  stock:  none — the archived predecessor had no CDS curve
  source: OKANE-2008 ch 7 (7.4 flat-forward interpolation, 7.5 the sequential bootstrap, 7.7 the
          arbitrage condition); the leg formulas are market/credit_legs.py, not re-transcribed
  oracle: tests/L1/test_cds_bootstrap.py — the known-curve round trip (an independently
          constructed term-structured curve, par spreads computed in the test with a DIFFERENT
          quadrature) plus the credit triangle inverted as a premium-frequency convergence
  crosscheck: quantlib SpreadCdsHelper / PiecewiseDefaultCurve — deferred, dev-only
  slice:  tranche 39 — the CDS survival bootstrap
"""

from __future__ import annotations

from collections.abc import Sequence

from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.solvers import bracketed_root
from pricebook.market.credit_legs import protection_leg, risky_pv01
from pricebook.market.curve import CurveHandle
from pricebook.market.survival_curve import SurvivalCurve

#: Points per grid segment. 50 is where the round trip reaches 3.6e-07 and the cost is ~150 points
#: for a 5y strip; the committed value is measured, and the suite asserts the convergence rate
#: rather than trusting it. (Second order, so 25/50/100 read 1.4e-06 / 3.6e-07 / 8.9e-08.)
_DEFAULT_PER_SEGMENT = 50

#: The hazard the bracket searches up to — 1000% a year, far beyond any quotable spread. Reached
#: only by expanding from a smaller bound, so a normal strip never evaluates the objective here.
_MAX_HAZARD = 10.0


def bootstrap_survival_curve(
    quotes: Sequence[tuple[CouponSchedule, float]],
    discount: CurveHandle,
    recovery: float,
    per_segment: int = _DEFAULT_PER_SEGMENT,
) -> SurvivalCurve:
    """Par CDS quotes, shortest maturity first, to a piecewise-flat forward hazard curve.

    Each quote is its own premium schedule and its quoted par spread, because the schedule is what
    carries the maturity *and* the premium frequency — a strip is not a mapping from maturity to
    spread, and treating it as one would silently assume one frequency across the curve.
    """
    if not quotes:
        raise ValueError("a survival bootstrap needs at least one CDS quote")
    if not 0.0 <= recovery <= 1.0:
        raise ValueError(f"the recovery must sit in [0, 1], got {recovery}")

    maturities = [discount.time_measure.year_fraction(schedule.dates[-1])
                  for schedule, _ in quotes]
    if any(b <= a for a, b in zip(maturities, maturities[1:])):
        raise ValueError(f"CDS quotes must be in strictly increasing maturity order, got "
                         f"{maturities}")
    if maturities[0] <= 0.0:
        raise ValueError(f"a CDS quote needs a maturity after the valuation date, got "
                         f"{maturities[0]}")

    times: list[float] = [0.0]
    hazards: list[float] = []

    for index, (schedule, spread) in enumerate(quotes):
        horizon = maturities[index]
        periods = schedule.periods()

        def value(hazard: float, horizon=horizon, periods=periods, spread=spread) -> float:
            """`V(λ)` for the trial curve — strictly increasing in `λ`."""
            trial = SurvivalCurve(discount.time_measure, tuple(times), (*hazards, hazard))
            return (protection_leg(discount, trial, recovery, horizon, per_segment)
                    - spread * risky_pv01(periods, discount, trial))

        at_zero = value(0.0)
        if at_zero > 0.0:
            raise ValueError(
                f"the quote {spread:.6%} at {horizon:.4f}y implies a NEGATIVE forward hazard: it "
                f"is too low to be consistent with the shorter maturities, which already price "
                f"{at_zero:.6g} of protection beyond it. A negative forward hazard is a survival "
                f"curve that rises, so the strip is arbitrageable (OKANE-2008 ch 7.7).")

        upper = 0.01
        while value(upper) < 0.0:
            upper *= 4.0
            if upper > _MAX_HAZARD:
                raise ValueError(
                    f"the quote {spread:.6%} at {horizon:.4f}y needs a forward hazard above "
                    f"{_MAX_HAZARD:.0%} a year, which is past anything quotable — the strip or "
                    f"the assumed loss given default is wrong.")

        hazards.append(bracketed_root(value, 0.0, upper))
        if index + 1 < len(quotes):
            times.append(horizon)

    return SurvivalCurve(discount.time_measure, tuple(times), tuple(hazards))
