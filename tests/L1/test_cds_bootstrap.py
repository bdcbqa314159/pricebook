"""The CDS survival-curve bootstrap (L1) — `OKANE-2008` ch 7. Pre-registered RED, tranche 39.

**The anchors, in the charter's preference order.** The correctness anchor is the **known-curve
round trip** (T8's, the one that caught two `BERHAN-2012` misprints): build a known term-structured
hazard curve, price par spreads off it *with this file's own arithmetic*, bootstrap those spreads
back, and recover the known hazards. It is non-telescoping because the reference is an independently
constructed curve, not the quotes — and it is the §4 requirement that *a formula and its stated
inverse must actually invert*. Beside it sits the **inverted credit triangle**, a closed form the
round trip never uses, so it catches a round trip that is self-consistently wrong.

**The reference here is independent in its QUADRATURE, not only in its code.** The protection leg
is referenced as `Σ ½(Z(a)+Z(b))·(Q(a)−Q(b))` — exact in the survival differences, no density — while
the implementation integrates `Z(t)·λ(t)S(t)dt`. Two different discretisations of one integral, so
their agreement is evidence about the composition rather than a restatement of it (§4: a second
rendering of one formula is not independence).

**Every band here is measured, never guessed** (§4). The numbers in the assertions were obtained by
sweeping each error source to its own convergence, and the tests assert the *rates* as well as the
values, because a rate is what a flat band cannot fake.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.cds_bootstrap import bootstrap_survival_curve
from pricebook.market.curve import DiscountCurve
from pricebook.market.survival_curve import SurvivalCurve

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
RECOVERY = 0.40

#: NON-FLAT, and read beyond its first segment — a flat curve hides the discounting entirely.
DISCOUNT = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0, 5.0), (0.020, 0.026, 0.030, 0.033, 0.036))

#: The curve the round trip must recover: pillars at the quote maturities, hazards TERM-STRUCTURED
#: and rising. A flat hazard would make every segment's solve identical, so the per-segment
#: sequencing — which segment the root-find is actually solving — would be invisible (§4).
PILLARS = (0.0, 1.0, 3.0, 5.0)
KNOWN_HAZARDS = (0.0100, 0.0220, 0.0400)
MATURITIES = (1.0, 3.0, 5.0)


def _years(d: date) -> float:
    return (d - VAL).days / 365.0


def _schedule(maturity_years: float, per_year: int = 2) -> CouponSchedule:
    """A premium schedule on its own day count — SEMI-ANNUAL by default, so `Δ ≠ 1`."""
    days = 365.0 / per_year
    count = round(maturity_years * per_year)
    return CouponSchedule(
        tuple(VAL + timedelta(days=round(days * i)) for i in range(count + 1)), DC)


def _known_survival(t: float) -> float:
    """`exp(−∫₀ᵗλ)` off `KNOWN_HAZARDS`, summed here segment by segment."""
    if t <= 0.0:
        return 1.0
    total = 0.0
    for index, hazard in enumerate(KNOWN_HAZARDS):
        start = PILLARS[index]
        if t <= start:
            break
        end = PILLARS[index + 1] if index + 1 < len(PILLARS) else math.inf
        total += hazard * (min(t, end) - start)
    return math.exp(-total)


def _reference_rpv01(schedule: CouponSchedule, survival) -> float:
    """`½ Σ Δ·Z(tₙ)·(Q(tₙ₋₁)+Q(tₙ))` — O'Kane 6.4, per unit notional."""
    return sum(
        0.5 * accrual * DISCOUNT.df_at(_years(pay))
        * (survival(_years(start)) + survival(_years(pay)))
        for start, pay, accrual in schedule.periods())


def _reference_protection(schedule: CouponSchedule, survival, steps: int = 4000) -> float:
    """`(1−R)·Σ ½(Z(a)+Z(b))·(Q(a)−Q(b))` — exact in the survival DIFFERENCES.

    Deliberately not the density form the implementation uses: differencing `Q` needs no `λ(t)S(t)`
    and so cannot share the implementation's error at a hazard discontinuity."""
    horizon = _years(schedule.dates[-1])
    step = horizon / steps
    area = 0.0
    for index in range(steps):
        lo, hi = step * index, step * (index + 1)
        area += 0.5 * (DISCOUNT.df_at(lo) + DISCOUNT.df_at(hi)) * (survival(lo) - survival(hi))
    return (1.0 - RECOVERY) * area


def _par_spread(schedule: CouponSchedule, survival) -> float:
    return _reference_protection(schedule, survival) / _reference_rpv01(schedule, survival)


def _known_quotes(per_year: int = 2) -> tuple[tuple[CouponSchedule, float], ...]:
    """Par spreads off the KNOWN curve, computed entirely in this file."""
    return tuple((_schedule(m, per_year), _par_spread(_schedule(m, per_year), _known_survival))
                 for m in MATURITIES)


class TestTheFixtureIsNotDegenerate:
    """The premises each oracle's power rests on (§4 — nine shapes and counting)."""

    def test_the_premium_accruals_are_NON_UNIT_and_UNEQUAL(self) -> None:
        accruals = [accrual for _, _, accrual in _schedule(5.0).periods()]
        assert all(abs(a - 1.0) > 0.4 for a in accruals), accruals
        assert len(set(round(a, 12) for a in accruals)) > 1, "a constant Δ hides a dropped offset"

    def test_the_known_hazards_are_TERM_STRUCTURED(self) -> None:
        """Else every segment's solve is the same number and the sequencing is invisible."""
        assert len(set(KNOWN_HAZARDS)) == len(KNOWN_HAZARDS)
        assert min(h2 / h1 for h1, h2 in zip(KNOWN_HAZARDS, KNOWN_HAZARDS[1:])) > 1.5

    def test_the_discount_curve_is_NON_FLAT_and_READ_BEYOND_its_first_segment(self) -> None:
        assert len(set(DISCOUNT.zero_rates)) > 1
        assert _years(_schedule(5.0).dates[-1]) > DISCOUNT.times[1]

    def test_the_quotes_are_DISTINCT_and_RISING(self) -> None:
        spreads = [spread for _, spread in _known_quotes()]
        assert spreads == sorted(spreads) and len(set(spreads)) == len(spreads), spreads


class TestTheKnownCurveRoundTrip:
    """The correctness anchor: the bootstrap must invert the par-spread calculation."""

    def test_the_known_hazards_are_RECOVERED(self) -> None:
        curve = bootstrap_survival_curve(_known_quotes(), DISCOUNT, RECOVERY)
        assert curve.hazards == pytest.approx(KNOWN_HAZARDS, rel=2e-06), curve.hazards

    def test_the_pillars_land_on_the_QUOTE_MATURITIES(self) -> None:
        curve = bootstrap_survival_curve(_known_quotes(), DISCOUNT, RECOVERY)
        assert curve.times[0] == 0.0
        assert curve.times[1:] == pytest.approx(
            tuple(_years(_schedule(m).dates[-1]) for m in MATURITIES[:-1]), rel=1e-12)

    def test_the_recovery_CONVERGES_as_the_grid_refines(self) -> None:
        """A rate, not a tolerance: a wrong segment or a flipped objective does not converge."""
        errors = []
        for per_segment in (25, 50, 100):
            curve = bootstrap_survival_curve(
                _known_quotes(), DISCOUNT, RECOVERY, per_segment=per_segment)
            errors.append(max(abs(g / k - 1.0)
                              for g, k in zip(curve.hazards, KNOWN_HAZARDS)))
        for coarse, fine in zip(errors, errors[1:]):
            assert coarse > fine, f"refining must improve the recovery: {errors}"
        assert errors[-1] < 2e-07, errors


class TestTheInvertedCreditTriangle:
    """Rung 1, and a closed form the round trip never touches.

    A FLAT strip at `S` must bootstrap to a flat hazard `λ = S/(1−R)` — but only in the
    **continuous-premium** limit. The gap at semi-annual is 1.0%, and measurement shows it is *not*
    the integration grid (identical at 50 and 400 points per segment): it is the premium
    discretisation, first-order in the premium period. So this is asserted as a convergence in the
    premium FREQUENCY, which is sharper than a band at any one frequency.
    """

    SPREAD = 0.0100

    def _lambda_error(self, per_year: int) -> float:
        quotes = tuple((_schedule(m, per_year), self.SPREAD) for m in MATURITIES)
        curve = bootstrap_survival_curve(quotes, DISCOUNT, RECOVERY)
        exact = self.SPREAD / (1.0 - RECOVERY)
        return max(abs(h / exact - 1.0) for h in curve.hazards)

    def test_the_hazard_CONVERGES_to_the_triangle_as_the_premium_period_shrinks(self) -> None:
        errors = {n: self._lambda_error(n) for n in (2, 4, 12)}
        # measured: 1.015e-02, 5.065e-03, 1.687e-03 — first order in the period
        assert errors[2] == pytest.approx(1.015e-02, rel=0.05), errors
        assert errors[4] == pytest.approx(5.065e-03, rel=0.05), errors
        assert errors[12] == pytest.approx(1.687e-03, rel=0.05), errors
        assert 1.8 < errors[2] / errors[4] < 2.2, f"first order in the period: {errors}"

    def test_a_DAILY_premium_strip_reaches_the_closed_form(self) -> None:
        """The continuous limit itself: 5.5e-05 measured at 365 periods per year."""
        assert self._lambda_error(365) < 1e-04

    def test_a_FLAT_strip_gives_a_FLAT_hazard_to_the_same_order(self) -> None:
        quotes = tuple((_schedule(m, 12), self.SPREAD) for m in MATURITIES)
        curve = bootstrap_survival_curve(quotes, DISCOUNT, RECOVERY)
        assert max(curve.hazards) / min(curve.hazards) - 1.0 < 1e-03, curve.hazards


class TestTheRequirementOracles:
    def test_the_survival_curve_is_MONOTONE_NON_INCREASING(self) -> None:
        curve = bootstrap_survival_curve(_known_quotes(), DISCOUNT, RECOVERY)
        values = [curve.survival(t / 4.0) for t in range(0, 21)]
        for earlier, later in zip(values, values[1:]):
            assert later <= earlier, values
        assert values[0] == 1.0

    def test_every_hazard_is_NON_NEGATIVE(self) -> None:
        curve = bootstrap_survival_curve(_known_quotes(), DISCOUNT, RECOVERY)
        assert all(h >= 0.0 for h in curve.hazards), curve.hazards

    def test_each_quote_REPRICES_to_par(self) -> None:
        """Nearly-telescoping, and the gap is the point.

        The spec called this the telescoping guard, and it would be if the repricer were the
        bootstrap's own. It is not: this file integrates the protection leg as
        `½(Z(a)+Z(b))·(Q(a)−Q(b))` while the bootstrap integrates `Z·λ·Q`, so what is asserted is
        that **two different quadratures of one integral agree at the solved hazards** — a
        cross-quadrature check, not a restatement. The band is measured (max 1.4e-08 observed,
        asserted at 1e-07 for cross-platform ULP room); before the per-segment hazard fix the same
        check read 6.9e-05, so it resolves that defect ~700x over."""
        curve = bootstrap_survival_curve(_known_quotes(), DISCOUNT, RECOVERY)
        for schedule, spread in _known_quotes():
            value = (_reference_protection(schedule, curve.survival)
                     - spread * _reference_rpv01(schedule, curve.survival))
            assert abs(value) < 1e-07, (spread, value)


class TestArbitrageIsREFUSED:
    """ch 7.7 — a strip implying a negative forward hazard must fail, not return a rising curve."""

    def test_a_strip_implying_a_NEGATIVE_FORWARD_HAZARD_is_refused(self) -> None:
        quotes = (
            (_schedule(1.0), 0.0400),   # 1y at 400bp
            (_schedule(3.0), 0.0100),   # 3y at 100bp — far too low to be consistent
        )
        with pytest.raises(ValueError, match="forward hazard"):
            bootstrap_survival_curve(quotes, DISCOUNT, RECOVERY)

    def test_THE_PREMISE_the_same_strip_is_fine_when_the_long_quote_is_RAISED(self) -> None:
        """Proves the refusal is about the arbitrage and not about the fixture's shape."""
        quotes = ((_schedule(1.0), 0.0400), (_schedule(3.0), 0.0450))
        curve = bootstrap_survival_curve(quotes, DISCOUNT, RECOVERY)
        assert all(h > 0.0 for h in curve.hazards)

    def test_a_ZERO_spread_strip_is_ACCEPTED_at_zero_hazard_not_refused(self) -> None:
        """The exact boundary of the arbitrage test, and the reason it is `>` and not `>=`.

        A 0bp par quote says the name cannot default, so `V(λ=0) = 0` **exactly** — the only
        input for which the sign test's boundary is reached. `>=` would reject it as
        arbitrageable, which is wrong: zero hazard is the right answer, not an inconsistency.
        Nothing else in the suite reaches `at_zero == 0.0`, so a `>`/`>=` mutation survived until
        this fixture existed."""
        quotes = tuple((_schedule(m), 0.0) for m in MATURITIES)
        curve = bootstrap_survival_curve(quotes, DISCOUNT, RECOVERY)
        assert curve.hazards == pytest.approx((0.0,) * len(MATURITIES), abs=1e-15), curve.hazards
        assert curve.survival(5.0) == 1.0

    def test_quotes_OUT_OF_MATURITY_ORDER_are_refused(self) -> None:
        quotes = ((_schedule(3.0), 0.0110), (_schedule(1.0), 0.0060))
        with pytest.raises(ValueError, match="maturity"):
            bootstrap_survival_curve(quotes, DISCOUNT, RECOVERY)

    def test_an_EMPTY_strip_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            bootstrap_survival_curve((), DISCOUNT, RECOVERY)

    def test_a_RECOVERY_outside_the_unit_interval_is_refused(self) -> None:
        """Matched on the guard's OWN words, which is not fussiness.

        This test first read `match="recovery"` and **passed with the guard deleted**: with
        `R = 1.4` the protection leg goes negative, no hazard can price the quote, and the
        bracket-expansion failure raised a message that happened to contain the word "recovery"
        too. A mutation sweep caught it. A refusal oracle has to name the condition it refuses, or
        it is only asserting that *something* went wrong."""
        with pytest.raises(ValueError, match=r"recovery must sit in \[0, 1\]"):
            bootstrap_survival_curve(_known_quotes(), DISCOUNT, 1.4)

    def test_an_UNREACHABLE_quote_is_refused_by_its_OWN_message(self) -> None:
        """The other path the test above used to collide with — pinned separately so it cannot
        stand in for anything else. A 60% spread at 40% recovery needs a hazard beyond the cap."""
        with pytest.raises(ValueError, match="past anything quotable"):
            bootstrap_survival_curve(((_schedule(1.0), 0.60),), DISCOUNT, 0.999999)
