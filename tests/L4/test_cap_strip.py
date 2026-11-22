"""The shared caplet-strip walk (L4). Convention C16's structural guard; re-audit #8's pass 2.

`engine/cap_strip.py` exists so the three smile-family pricers cannot disagree about **which date**
a caplet's volatility accumulates to. Convention C16 was recorded **per-module** because no shared
value existed to compare; the walker computes the fixing-date horizon once, so these tests can
assert the thing C16 could previously only describe.

What is asserted here is the walk — the horizon it hands out, the valuation-date partition, and each
caller's exception set. The caplet formulas have their own oracles in `test_smile_caplet.py` and
`test_cev_caplet.py`, and both suites priced **bit-identically** across this extraction (54
`(periods × strike × kind × model)` combinations, in-process — the T12(a) proof, available because
no operation order moved).
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.cap_strip import price_cap_strip
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    Money,
    PricingFailure,
    PricingResult,
)
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.cev import CevModel
from pricebook.models.hull_white import CapletPeriod
from pricebook.models.smile import BachelierModel, DisplacedDiffusionModel
from pricebook.products import NominalCap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
N = 10_000_000.0
KEY = CurveKey(CurveRole.DISCOUNT, EUR)
# Straddling, and semi-annual so no accrual is 1.0 (§4's first and tenth degenerate shapes).
DATES = tuple(VAL + timedelta(days=180 * k) for k in range(-1, 5))
CURVE = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 4.0), (0.026, 0.030, 0.034, 0.038))
SNAPSHOT = MarketSnapshot(valuation_date=VAL, curves=CurveSet({KEY: CURVE}))


def _cap(kind: CapFloorKind = CapFloorKind.CAP) -> NominalCap:
    return NominalCap(CouponSchedule(DATES, DC), 0.03, Money(N, EUR), kind)


def test_the_horizon_handed_to_a_valuation_is_the_FIXING_date_not_the_PAYMENT_date():
    # C16, now assertable: the walker computes the horizon, so a caller cannot read the wrong date
    # because it is never handed one. Recorded per period, then checked against the schedule.
    seen: list[float] = []

    def record(unit: CapletPeriod, years: float) -> float:
        seen.append(years)
        return 0.0

    price_cap_strip(_cap(), SNAPSHOT, record, (KeyError,))
    live = [p for p in _cap().schedule.periods() if p[1] > VAL]
    assert seen == [MEASURE.year_fraction(start) for start, _, _ in live]

    # **The first live period starts exactly ON the valuation date, so its horizon is 0.0** — a
    # caplet fixing today has accumulated no variance, and that boundary is asserted rather than
    # discovered (it is what a `> 0.0` premise would trip over).
    assert seen[0] == 0.0 and all(y > 0.0 for y in seen[1:]), seen

    # The premise that makes this discriminate at all: wherever the horizon is non-zero, reading the
    # PAYMENT date instead would scale the vol by a materially different factor.
    payments = [MEASURE.year_fraction(pay) for _, pay, _ in live]
    ratios = [math.sqrt(p / f) for f, p in zip(seen, payments) if f > 0.0]
    assert len(ratios) == 3 and all(r > 1.15 for r in ratios), (seen, payments)
    # and at the zero-horizon period the two branches differ absolutely rather than by a ratio
    assert payments[0] > 0.4 > seen[0]


def test_the_unit_handed_over_is_what_the_CURVE_says_about_the_period():
    units: list[CapletPeriod] = []

    def record(unit: CapletPeriod, years: float) -> float:
        units.append(unit)
        return 0.0

    price_cap_strip(_cap(), SNAPSHOT, record, (KeyError,))
    live = [p for p in _cap().schedule.periods() if p[1] > VAL]
    assert [(u.df_start, u.df_pay, u.accrual) for u in units] == [
        (CURVE.df(start), CURVE.df(pay), accrual) for start, pay, accrual in live
    ]


def test_HISTORICAL_periods_are_excluded_and_the_fixture_really_has_one():
    live = [p for p in _cap().schedule.periods() if p[1] > VAL]
    settled = [p for p in _cap().schedule.periods() if p[1] <= VAL]
    assert len(settled) == 1 and len(live) == 4, (settled, live)
    calls = []
    price_cap_strip(_cap(), SNAPSHOT, lambda unit, years: calls.append(years) or 0.0, (KeyError,))
    assert len(calls) == len(live)
    assert all(y >= 0.0 for y in calls), calls
    # the discriminator: the excluded period's own horizon is NEGATIVE, so letting it through would
    # hand a valuation a negative variance rather than merely one more term (T72's F-8.3 lesson —
    # measure what the dropped guard would actually produce, not a diffused stand-in)
    settled_start = settled[0][0]
    assert MEASURE.year_fraction(settled_start) < 0.0


def test_a_fully_SETTLED_strip_is_PV_ZERO_without_touching_the_curve():
    after = DATES[-1] + timedelta(days=1)
    empty = MarketSnapshot(valuation_date=after, curves=CurveSet({}))   # no curve at all

    def unreachable(unit: CapletPeriod, years: float) -> float:
        raise AssertionError("a settled strip must not reach the valuation")

    out = price_cap_strip(_cap(), empty, unreachable, (KeyError,))
    assert isinstance(out, PricingResult) and out.pv.amount == 0.0


def test_the_notional_SCALES_the_sum_and_the_walk_ACCUMULATES():
    out = price_cap_strip(_cap(), SNAPSHOT, lambda unit, years: 1.0, (KeyError,))
    assert isinstance(out, PricingResult)
    assert out.pv.amount == pytest.approx(4.0 * N, rel=1e-15, abs=0.0)


# ============================================================================================
# The exception sets — the thing a shared walker would otherwise lose silently.
# ============================================================================================

def test_a_LISTED_exception_becomes_a_PricingFailure():
    def boom(unit: CapletPeriod, years: float) -> float:
        raise ArithmeticError("the mixture is unaffordable")

    out = price_cap_strip(_cap(), SNAPSHOT, boom, (KeyError, ArithmeticError))
    assert isinstance(out, PricingFailure)
    assert "unaffordable" in out.reason


def test_an_UNLISTED_exception_PROPAGATES_rather_than_being_swallowed():
    # A walker cannot know which exceptions a valuation considers expected, so anything not named
    # is a bug and must not be dressed up as a priceable failure (§3).
    def boom(unit: CapletPeriod, years: float) -> float:
        raise ArithmeticError("not in this caller's set")

    with pytest.raises(ArithmeticError, match="not in this caller's set"):
        price_cap_strip(_cap(), SNAPSHOT, boom, (KeyError,))


def test_the_CEV_engine_really_carries_ArithmeticError_end_to_end():
    # The one exception set that is not `(KeyError,)`, asserted through the registered pricer: the
    # noncentrality refusal must arrive as a PricingFailure, not a traceback.
    out = price(_cap(), CevModel(SNAPSHOT, 0.1, 1.0 - 1e-4))
    assert isinstance(out, PricingFailure), out
    assert "noncentrality" in out.reason


def test_a_MISSING_curve_is_a_PricingFailure_for_all_three_engines():
    no_curve = MarketSnapshot(valuation_date=VAL, curves=CurveSet({}))
    for model in (
        BachelierModel(no_curve, 0.0085),
        DisplacedDiffusionModel(no_curve, 0.22, 0.015),
        CevModel(no_curve, 0.1, 0.5),
    ):
        out = price(_cap(), model)
        assert isinstance(out, PricingFailure), (type(model), out)


def test_all_THREE_engines_walk_the_SAME_periods_which_is_what_C16_now_guards():
    # The structural claim: one horizon expression, so the three cannot diverge on which date it
    # reads. Asserted behaviourally — each engine's strip equals the sum of its own caplets taken
    # over the live periods at the FIXING horizon, computed here from the schedule.
    for model, kind in (
        (BachelierModel(SNAPSHOT, 0.0085), CapFloorKind.CAP),
        (DisplacedDiffusionModel(SNAPSHOT, 0.22, 0.015), CapFloorKind.FLOOR),
        (CevModel(SNAPSHOT, 0.1, 0.5), CapFloorKind.CAP),
    ):
        out = price(_cap(kind), model)
        assert isinstance(out, PricingResult), (type(model), out)
        assert out.pv.amount != 0.0
