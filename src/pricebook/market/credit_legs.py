"""The CDS leg arithmetic, shared by the engine and the bootstrap (L1). `OKANE-2008` ch 6.

    RPV01      = ½ Σₙ Δₙ · Z(tₙ) · (Q(tₙ₋₁) + Q(tₙ))                          (6.4)
    Protection = (1−R) ∫₀ᵀ Z(t) λ(t) Q(t) dt                                   (6.2)

Both **per unit notional** — the notional scaling belongs to whoever has a notional, which the
bootstrap does not: a par quote is a rate, and the notional cancels out of `V = 0`.

**Why this is L1 and not L4.** The arithmetic reads a schedule's periods (L0 `CouponSchedule`), a
discount curve and a survival curve (L1), and a scalar recovery. It never touches a
`CreditDefaultSwap`, so nothing here needs L2 — the currency→curve selection that *does* need the
product stays in the engine. It sits beside `nominal_bootstrap.py` for the same reason that does:
a piece the bootstrap needs, expressible in curves alone.

**`risky_pv01` is shared; `protection_leg` is NOT yet.** Pulling `risky_pv01` out of tranche 38's
engine was a pure refactor, proved bit-identical over 324 fixtures. The protection leg is a
different story: the integration here is **not** the engine's, because the engine's is less
accurate, and correcting it would move a shipped number by 1.8e-05. That is a ruling, not a
refactor, so the engine keeps its own loop until Cowork answers `OPEN.md` OI-4, and this is the
one knowingly duplicated formula in the tree — ledgered, with that answer as its re-open trigger.

Provenance:
  stock:  none — the archived predecessor had no CDS
  source: OKANE-2008 ch 6 (eqs 6.2, 6.4); `risky_pv01` extracted unchanged from tranche 38
  oracle: tests/L1/test_credit_legs.py — `risky_pv01` is proved bit-identical to the tranche-38
          engine's arithmetic over 324 fixtures; `protection_leg` is anchored by the bootstrap's
          known-curve round trip and by agreeing with the engine's closed form where the two
          integrations provably coincide (a flat hazard has no interior pillars)
  crosscheck: quantlib CreditDefaultSwap couponLegNPV / defaultLegNPV — deferred, dev-only
  slice:  tranche 39 — the CDS survival bootstrap pulls the leg arithmetic out of the engine
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from pricebook.market.curve import CurveHandle
from pricebook.market.survival_curve import SurvivalCurve


def risky_pv01(
    periods: Sequence[tuple[date, date, float]],
    discount: CurveHandle,
    survival: SurvivalCurve,
) -> float:
    """`RPV01` per unit notional — eq 6.4, premium payments plus the coupon accrued at default.

    The averaged survival is the combined form: the surviving premium `Q(tₙ)` plus the
    accrual-on-default `½(Q(tₙ₋₁) − Q(tₙ))`. Dropping the `Q(tₙ₋₁)` half is worth 0.59% on the
    tranche-38 fixture and is invisible to every property oracle, which is why eq 6.4 is
    transcribed pointwise in the tests rather than merely bounded.
    """
    total = 0.0
    for start, pay, accrual in periods:
        start_years = discount.time_measure.year_fraction(start)
        pay_years = discount.time_measure.year_fraction(pay)
        total += (
            0.5
            * accrual
            * discount.df_at(pay_years)
            * (survival.survival(start_years) + survival.survival(pay_years))
        )
    return total


def protection_leg(
    discount: CurveHandle,
    survival: SurvivalCurve,
    recovery: float,
    horizon: float,
    per_segment: int,
) -> float:
    """`(1−R) ∫₀ᵀ Z(t)λ(t)Q(t)dt` per unit notional — eq 6.2, integrated segment by segment.

    **The hazard is read once per segment, not pointwise, and that is the whole point.** Between
    two pillars `λ` is constant, so evaluating it at the segment's midpoint is *exact*, while
    `Q(t)` is continuous and is evaluated at `t`. Reading `λ(t)` pointwise instead goes wrong at a
    breakpoint: `default_density(hi)` returns the hazard of the segment *starting* at `hi` — the
    wrong one for the piece ending there — and a trapezoid fed that endpoint loses an order. The
    cost is visible in the tranche-39 round trip: the recovered hazards converge at `O(1/M)` with
    pointwise density against `O(1/M²)` here, 2.9e-03 against 3.6e-07 at the committed grid.

    Breaking at the pillars is therefore not a tuning choice but a correctness one, and it is also
    cheaper — 50 points per segment beats 500 uniform points by ~800×.

    Returns 0 for a non-positive horizon: a matured or same-day contract has no protection left,
    which is invariant 6's business and not a failure.
    """
    if per_segment < 1:
        raise ValueError(
            f"the protection leg needs at least one point per segment, got {per_segment}")
    if horizon <= 0.0:
        return 0.0

    breaks = sorted({0.0, horizon} | {t for t in survival.times if 0.0 < t < horizon})
    area = 0.0
    for lo, hi in zip(breaks, breaks[1:]):
        hazard = survival.hazard(0.5 * (lo + hi))
        step = (hi - lo) / per_segment
        previous = discount.df_at(lo) * survival.survival(lo)
        for index in range(1, per_segment + 1):
            t = lo + step * index
            current = discount.df_at(t) * survival.survival(t)
            area += 0.5 * (previous + current) * step * hazard
            previous = current
    return (1.0 - recovery) * area
