"""Tranche 3 — the inflation-linked bond (linker), model-free. BERHAN-2012 §1.2.1.

`ILB(t,TM) = I(t)/I0 · N[ C·Σᵢ Pr(t,Tᵢ) + Pr(t,TM) ]` (1.2.5), unit real value
`P_IL(t,T) = I(t)·Pr(t,T)` (1.2.4). Note 1.2.5 **never touches the nominal curve**: a linker is a
real bond scaled by the index ratio.

**The paper's 1.2.5 is the annual, unit-accrual case.** A real linker pays `C·ψᵢ` per period, so
the library prices the general form and 1.2.5 is its **reduction** at `ψᵢ = 1` — asserted below.
That is why `CouponSchedule.day_count` is genuinely read rather than carried unread.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.inflation import inflation_leg_value
from pricebook.foundation import (
    CouponSchedule,
    Currency,
    DayCountConvention,
    IndexInterpolation,
    InflationIndex,
    Money,
    PricingFailure,
    PricingResult,
    RealLeg,
    Term,
)
from pricebook.market import FixingSeries, CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models import DiscountingModel
from pricebook.products import InflationLinkedBond, ZeroCouponInflationSwap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
HICP = InflationIndex("HICPxt", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
N, I0, C = 1_000_000.0, 100.0, 0.015

#: start + five annual payment dates, each period exactly 365 days ⇒ every ψ is exactly 1.0, which
#: is what makes the reduction to 1.2.5 an exact identity rather than an approximation.
DATES = tuple(VAL + timedelta(days=365 * k) for k in range(6))
SCHEDULE = CouponSchedule(DATES, DC)
NOMINAL = CurveKey(CurveRole.DISCOUNT, EUR)
REAL = CurveKey(CurveRole.REAL, EUR)



def _level_series(level: float) -> FixingSeries:
    """A series that resolves to `level` at ANY date, because every print is equal — so neither the
    publication lag nor the interpolation weight can change the answer.

    Deliberately flat *here*: these are the tranche 2–6 oracles, where the index LEVEL is what is
    under test and resolution is not. The conventions get a sloped, mid-month fixture in
    `tests/L1/test_fixing_series.py`, which is where a flat series would have no power."""
    return FixingSeries(
        {date(y, m, 1): level for y in range(2024, 2032) for m in range(1, 13)}
    )

def _market(nominal: float = 0.03, real: float = 0.01, index_level: float = I0) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {NOMINAL: DiscountCurve.flat(VAL, nominal, DC), REAL: DiscountCurve.flat(VAL, real, DC)}
        ),
        fixings={HICP: _level_series(index_level)},
    )


def _bond(coupon: float = C, schedule: CouponSchedule = SCHEDULE, lag: int = 3) -> InflationLinkedBond:
    return InflationLinkedBond(HICP, I0, RealLeg(schedule, coupon), Money(N, EUR), lag)


def _pv(bond: InflationLinkedBond, market: MarketSnapshot) -> float:
    out = price(bond, DiscountingModel(market))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


# ── oracle 1: the general closed form, computed in the test ─────────────────────────────────
def test_the_linker_matches_the_closed_form_with_accrual_fractions() -> None:
    """`I(t)/I0 · N[ Σ C·ψᵢ·Pr(payᵢ) + Pr(TM) ]`. The test computes every ψ and every `Pr` itself,
    so both sides are arithmetic and nothing is read back from the engine. Note the nominal rate is
    varied and must not move the price — 1.2.5 does not involve `Pn`."""
    for real in (0.005, 0.01, 0.03):
        for nominal in (0.02, 0.06):
            market = _market(nominal, real)
            total = 0.0
            for start, pay in zip(DATES, DATES[1:]):
                psi = (pay - start).days / 365.0
                total += C * psi * math.exp(-real * (pay - VAL).days / 365.0)
            total += math.exp(-real * (DATES[-1] - VAL).days / 365.0)
            assert _pv(_bond(), market) == pytest.approx(N * total, abs=1e-8, rel=1e-12), (real, nominal)


def test_the_nominal_curve_does_NOT_affect_the_price() -> None:
    """1.2.5 is a real-curve-only formula. If the nominal curve leaked in, this moves."""
    base = _pv(_bond(), _market(nominal=0.02))
    assert _pv(_bond(), _market(nominal=0.09)) == pytest.approx(base, abs=1e-9)


# ── oracle 1b: the EXACT reduction to the paper's annual form ───────────────────────────────
def test_with_unit_accruals_the_general_form_REDUCES_to_paper_eq_1_2_5() -> None:
    """Every period is exactly 365 days, so each `ψᵢ = 1` and the general form *is* 1.2.5 — the
    paper's own equation, with no accrual factor. One formula, not two: if `ψ` were applied to the
    principal, or omitted from the coupons, the two would diverge."""
    real = 0.01
    market = _market(real=real)
    paper = sum(C * math.exp(-real * (p - VAL).days / 365.0) for p in DATES[1:])
    paper += math.exp(-real * (DATES[-1] - VAL).days / 365.0)
    assert all((b - a).days == 365 for a, b in zip(DATES, DATES[1:]))  # the premise, asserted
    assert _pv(_bond(), market) == pytest.approx(N * paper, abs=1e-8, rel=1e-12)


# ── oracle 2: cross-product reconciliation against T2 ───────────────────────────────────────
def test_the_linker_IS_a_strip_of_zero_coupon_inflation_bonds_from_T2() -> None:
    """`ILB = Σ C·ψᵢ·ZCIB(Tᵢ) + ZCIB(TM)`, with each `ZCIB(T) = inflation_leg_value(T) + N·Pn(T)`.

    **Non-circular, and that is the point.** The two sides compose different code: the pricer sums
    `Pr.df` over the schedule directly, while this routes through tranche 2's independently-verified
    `inflation_leg_value` — whose `−Pn` the `+N·Pn(T)` undoes to recover `N·I(t)/I0·Pr(t,T)`. A
    divergent index-ratio application in one product but not the other breaks this while each
    product's own oracle still passes. It is a *consistency* guard; oracle 1 remains the
    correctness anchor."""
    market = _market()
    nominal_curve = market.curves.discount(EUR)

    def zcib(when: date) -> float:
        swap = ZeroCouponInflationSwap(HICP, I0, 0.0, Term(VAL, when, DC), Money(N, EUR))
        return inflation_leg_value(swap, market) + N * nominal_curve.df(when)

    strip = sum(
        C * ((pay - start).days / 365.0) * zcib(pay) for start, pay in zip(DATES, DATES[1:])
    )
    strip += zcib(DATES[-1])
    assert _pv(_bond(), market) == pytest.approx(strip, abs=1e-6, rel=1e-12)


# ── oracle 3: the zero-coupon limit — an exact reduction ────────────────────────────────────
def test_a_zero_coupon_linker_IS_the_unit_real_value_eq_1_2_4() -> None:
    """`C = 0` and one payment: the bond is `I(t)/I0 · N · Pr(t,TM)`, which is 1.2.4 scaled. A
    double-counted principal or a mis-built schedule shows up here and nowhere else."""
    one_period = CouponSchedule((VAL, DATES[-1]), DC)
    real = 0.01
    want = N * math.exp(-real * (DATES[-1] - VAL).days / 365.0)
    assert _pv(_bond(coupon=0.0, schedule=one_period), _market(real=real)) == pytest.approx(want, abs=1e-8)


# ── oracle 4: with no inflation, a linker IS a nominal bond ─────────────────────────────────
def test_with_Pr_equal_Pn_and_the_index_at_base_it_prices_as_a_NOMINAL_bond() -> None:
    """Catches the index ratio applied to the wrong leg, and `Pr`/`Pn` transposed."""
    r = 0.025
    market = _market(nominal=r, real=r, index_level=I0)
    nominal_bond = sum(
        C * ((p - s).days / 365.0) * math.exp(-r * (p - VAL).days / 365.0)
        for s, p in zip(DATES, DATES[1:])
    ) + math.exp(-r * (DATES[-1] - VAL).days / 365.0)
    assert _pv(_bond(), market) == pytest.approx(N * nominal_bond, abs=1e-8)


# ── oracle 5: the index ratio scales the WHOLE price ────────────────────────────────────────
def test_doubling_the_index_doubles_the_whole_price_coupons_and_principal() -> None:
    """1.2.5 is linear in `I(t)/I0`. Applying the ratio to coupons but not principal (or the
    reverse) would scale by less than two."""
    base = _pv(_bond(), _market())
    assert _pv(_bond(), _market(index_level=2.0 * I0)) == pytest.approx(2.0 * base, abs=1e-8)


# ── oracle 6: the valuation-date partition, now across MANY flows ───────────────────────────
def test_coupons_on_or_before_the_valuation_date_are_EXCLUDED() -> None:
    """Invariant 6 across a schedule rather than a single exchange: a seasoned linker's past
    coupons are the shell's business. Priced at a date after two coupons have paid, the bond must
    equal the same bond with those two dates removed."""
    later = DATES[2] + timedelta(days=10)
    market = MarketSnapshot(
        valuation_date=later,
        curves=CurveSet(
            {NOMINAL: DiscountCurve.flat(later, 0.03, DC), REAL: DiscountCurve.flat(later, 0.01, DC)}
        ),
        fixings={HICP: _level_series(110.0)},
    )
    full = _pv(_bond(), market)
    remaining = _pv(_bond(schedule=CouponSchedule((DATES[2], *DATES[3:]), DC)), market)
    assert full == pytest.approx(remaining, abs=1e-8)


def test_a_MATURED_linker_is_PV_ZERO_not_a_failure() -> None:
    after = DATES[-1] + timedelta(days=1)
    market = MarketSnapshot(
        valuation_date=after,
        curves=CurveSet({REAL: DiscountCurve.flat(after, 0.01, DC)}),
        fixings={HICP: _level_series(120.0)},
    )
    out = price(_bond(), DiscountingModel(market))
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == pytest.approx(0.0, abs=1e-12)


# ── oracle 7: failures, and the representation EXERCISING the lag ───────────────────────────
def test_a_missing_real_curve_or_fixing_is_a_failure() -> None:
    no_real = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({NOMINAL: DiscountCurve.flat(VAL, 0.03, DC)}),
        fixings={HICP: _level_series(I0)},
    )
    assert isinstance(price(_bond(), DiscountingModel(no_real)), PricingFailure)
    no_fixing = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({REAL: DiscountCurve.flat(VAL, 0.01, DC)}),
        fixings={},
    )
    assert isinstance(price(_bond(), DiscountingModel(no_fixing)), PricingFailure)


def test_the_round_trip_EXERCISES_the_indexation_lag() -> None:
    """The lag is contract data the representation must carry: two linkers on one index with
    different indexation lags are **different contracts** (3-month TIPS vs 8-month UK ILG), so a
    payload without it is ambiguous. That round-trip is its present consumer — which is what makes
    it a carried field rather than an unread one."""
    for lag in (3, 8):
        bond = _bond(lag=lag)
        restored = InflationLinkedBond.from_dict(bond.to_dict())
        assert restored == bond
        assert restored.indexation_lag == lag
    assert _bond(lag=3) != _bond(lag=8)  # the lag is part of the contract's identity


def test_the_schedule_day_count_is_round_tripped_too() -> None:
    assert CouponSchedule.from_dict(SCHEDULE.to_dict()) == SCHEDULE


def test_an_unknown_schema_version_is_refused() -> None:
    bad = _bond().to_dict()
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="reads up to"):
        InflationLinkedBond.from_dict(bad)
