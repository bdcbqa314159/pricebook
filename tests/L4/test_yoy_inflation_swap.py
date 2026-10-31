"""The year-on-year inflation swap under Jarrow-Yildirim. BERHAN-2012 eq 2.5.13 (and 1.2.18).

    YYIIS(0,T,Psi,N) = N*psi_1*[Pr(0,T1) - Pn(0,T1)]
                     + N*sum_{i=2..M} psi_i*[ Pn(0,Ti-1)*Pr(0,Ti)/Pr(0,Ti-1)*e^C - Pn(0,Ti) ]

The **first** cash flow is priced by the zero-coupon inflation leg formula (1.2.13) because its
previous fixing is already known -- the thesis says so explicitly -- which makes the
cross-product reconciliation against tranche 2 a statement of the formula's own structure rather
than an invented check.

Pricing is at **t = 0** (eq 2.5.13). The seasoned case (eq 2.5.12, `iota(t) > 1`) needs a
*historical* index fixing, i.e. an index **series** in the snapshot, which is a new market-data
shape and is deferred with the indexation-lag resolution.
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
    InflationIndex,
    Money,
    PricingFailure,
    PricingResult,
    Term,
    YoYLeg,
)
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models import (
    HullWhiteParams,
    JarrowYildirim,
    JYCorrelations,
    JYParams,
    yoy_convexity,
)
from pricebook.products import YearOnYearInflationSwap, ZeroCouponInflationSwap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
HICP = InflationIndex("HICPxt")
N, FIXED = 1_000_000.0, 0.021
DATES = tuple(VAL + timedelta(days=365 * k) for k in range(6))   # start + 5 annual periods
SCHEDULE = CouponSchedule(DATES, DC)
NOMINAL_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
REAL_KEY = CurveKey(CurveRole.REAL, EUR)

NOM_P = HullWhiteParams(0.05, 0.010)
REAL_P = HullWhiteParams(0.04, 0.008)
SI = 0.012
CORR = JYCorrelations(nominal_real=0.30, real_index=-0.20, nominal_index=0.15)


def _snapshot(nominal: float = 0.03, real: float = 0.01) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {
                NOMINAL_KEY: DiscountCurve.flat(VAL, nominal, DC),
                REAL_KEY: DiscountCurve.flat(VAL, real, DC),
            }
        ),
        index_fixings={HICP: 100.0},
    )


def _model(nominal: float = 0.03, real: float = 0.01, real_vol: float = 0.008) -> JarrowYildirim:
    return JarrowYildirim(
        _snapshot(nominal, real),
        JYParams(NOM_P, HullWhiteParams(REAL_P.kappa, real_vol), SI, CORR),
    )


def _swap(fixed: float = FIXED) -> YearOnYearInflationSwap:
    return YearOnYearInflationSwap(HICP, YoYLeg(SCHEDULE, fixed), Money(N, EUR))


def _pv(swap: YearOnYearInflationSwap, model: JarrowYildirim) -> float:
    out = price(swap, model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _t(d: date) -> float:
    return (d - VAL).days / 365.0


def _reference_leg(nominal: float, real: float, real_vol: float) -> float:
    """Eq 2.5.13's inflation leg, computed here from the curves and 2.5.11."""
    def pn(d: date) -> float:
        return math.exp(-nominal * _t(d))

    def pr(d: date) -> float:
        return math.exp(-real * _t(d))

    periods = SCHEDULE.periods()
    _, first_pay, psi1 = periods[0]
    total = psi1 * (pr(first_pay) - pn(first_pay))
    for start, pay, psi in periods[1:]:
        c = yoy_convexity(
            0.0, _t(start), _t(pay),
            JYParams(NOM_P, HullWhiteParams(REAL_P.kappa, real_vol), SI, CORR),
        )
        total += psi * (pn(start) * pr(pay) / pr(start) * math.exp(c) - pn(pay))
    return N * total


# ── oracle 1: eq 2.5.13 against an independent transcription ────────────────────────────────
def test_the_inflation_leg_matches_eq_2_5_13() -> None:
    from pricebook.engine.yoy import yoy_inflation_leg_value

    for nominal, real in ((0.03, 0.01), (0.05, 0.045), (0.02, 0.035)):
        model = _model(nominal, real)
        got = yoy_inflation_leg_value(_swap(), model)
        assert got == pytest.approx(_reference_leg(nominal, real, 0.008), abs=1e-7, rel=1e-12)


# ── oracle 2: THE ANCHOR — sigma_r = 0 gives the deterministic-real-rate limit ───────────────
def test_at_zero_real_vol_the_leg_IS_the_deterministic_limit() -> None:
    """`C = 0` exactly when `sigma_r = 0`, so every period collapses to
    `Pn(0,Ti-1)*Pr(0,Ti)/Pr(0,Ti-1) - Pn(0,Ti)` -- the no-convexity baseline the extraction doc
    names. Exact by construction: `sigma_r` is 2.5.11's overall factor."""
    from pricebook.engine.yoy import yoy_inflation_leg_value

    nominal, real = 0.03, 0.01
    def pn(d: date) -> float:
        return math.exp(-nominal * _t(d))

    def pr(d: date) -> float:
        return math.exp(-real * _t(d))

    periods = SCHEDULE.periods()
    _, first_pay, psi1 = periods[0]
    want = psi1 * (pr(first_pay) - pn(first_pay))
    for start, pay, psi in periods[1:]:
        want += psi * (pn(start) * pr(pay) / pr(start) - pn(pay))   # no e^C factor at all
    got = yoy_inflation_leg_value(_swap(), _model(nominal, real, real_vol=0.0))
    assert got == pytest.approx(N * want, abs=1e-7, rel=1e-12)


def test_the_convexity_MOVES_the_leg_so_the_anchor_is_not_vacuous() -> None:
    """Without this, oracle 2 could pass on an implementation that ignored `C` entirely."""
    from pricebook.engine.yoy import yoy_inflation_leg_value

    flat = yoy_inflation_leg_value(_swap(), _model(real_vol=0.0))
    stochastic = yoy_inflation_leg_value(_swap(), _model(real_vol=0.02))
    assert abs(stochastic - flat) > 1e-3 * abs(flat)


# ── oracle 3: the FIRST period IS tranche 2's ZCIIS leg ─────────────────────────────────────
def test_the_first_period_IS_the_T2_zero_coupon_inflation_leg() -> None:
    """The thesis prices the first cash flow by 1.2.13 because its previous fixing is known, so
    this reconciliation is the formula's own structure -- and it routes through tranche 2's
    independently-verified `inflation_leg_value`, which is different code from this pricer."""
    from pricebook.engine.yoy import yoy_inflation_leg_value

    model = _model()
    one_period = YearOnYearInflationSwap(
        HICP, YoYLeg(CouponSchedule(DATES[:2], DC), FIXED), Money(N, EUR)
    )
    _, first_pay, psi1 = CouponSchedule(DATES[:2], DC).periods()[0]
    zciis = ZeroCouponInflationSwap(HICP, 100.0, 0.0, Term(VAL, first_pay, DC), Money(N, EUR))
    assert yoy_inflation_leg_value(one_period, model) == pytest.approx(
        psi1 * inflation_leg_value(zciis, model.market), abs=1e-7, rel=1e-12
    )


# ── oracle 4: the net swap, and its par rate ────────────────────────────────────────────────
def test_the_net_swap_is_the_leg_minus_the_fixed_leg() -> None:
    from pricebook.engine.yoy import yoy_inflation_leg_value

    model = _model()
    nominal_curve = model.market.curves.discount(EUR)
    fixed = N * sum(FIXED * psi * nominal_curve.df(pay) for _, pay, psi in SCHEDULE.periods())
    assert _pv(_swap(), model) == pytest.approx(
        yoy_inflation_leg_value(_swap(), model) - fixed, abs=1e-7
    )


def test_a_swap_struck_at_its_PAR_rate_prices_to_zero() -> None:
    """Solved from the leg, so it checks that the two legs share one annuity and one set of
    discount factors -- a consistency guard, telescoping by nature, kept for that reason only."""
    from pricebook.engine.yoy import yoy_inflation_leg_value, yoy_par_rate

    model = _model()
    par = yoy_par_rate(_swap(), model)
    assert 0.0 < par < 0.10
    assert _pv(_swap(fixed=par), model) == pytest.approx(0.0, abs=1e-6)
    assert yoy_inflation_leg_value(_swap(fixed=par), model) > 0.0   # not vacuously zero


# ── oracle 5: the valuation-date partition and failures ─────────────────────────────────────
def test_a_matured_swap_is_PV_ZERO_and_a_missing_curve_is_a_failure() -> None:
    after = DATES[-1] + timedelta(days=1)
    matured = JarrowYildirim(
        MarketSnapshot(
            valuation_date=after,
            curves=CurveSet({NOMINAL_KEY: DiscountCurve.flat(after, 0.03, DC)}),
        ),
        JYParams(NOM_P, REAL_P, SI, CORR),
    )
    out = price(_swap(), matured)
    assert isinstance(out, PricingResult) and out.pv.amount == pytest.approx(0.0, abs=1e-12)

    no_real = JarrowYildirim(
        MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet({NOMINAL_KEY: DiscountCurve.flat(VAL, 0.03, DC)}),
        ),
        JYParams(NOM_P, REAL_P, SI, CORR),
    )
    assert isinstance(price(_swap(), no_real), PricingFailure)


def test_the_yoy_swap_needs_NO_index_fixing_at_inception() -> None:
    """Eq 2.5.13 contains no index level -- the first term is `Pr - Pn` and the rest are forward
    ratios. So a YoY swap at inception prices off curves alone, which is worth asserting because
    requiring a fixing would be a plausible-looking error."""
    no_fixing = JarrowYildirim(
        MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet(
                {NOMINAL_KEY: DiscountCurve.flat(VAL, 0.03, DC),
                 REAL_KEY: DiscountCurve.flat(VAL, 0.01, DC)}
            ),
        ),
        JYParams(NOM_P, REAL_P, SI, CORR),
    )
    assert isinstance(price(_swap(), no_fixing), PricingResult)


def test_the_swap_round_trips_and_refuses_an_unknown_version() -> None:
    swap = _swap()
    assert YearOnYearInflationSwap.from_dict(swap.to_dict()) == swap
    bad = swap.to_dict()
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="reads up to"):
        YearOnYearInflationSwap.from_dict(bad)
