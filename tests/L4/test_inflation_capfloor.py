"""The inflation cap/floor under Jarrow-Yildirim. BERHAN-2012 eq 2.6.5 (and 1.2.22-1.2.23).

    ILCFLT(t,Ti-1,Ti,psi,K,N,w) = w*N*psi*Pn(t,Ti)*[ m*Phi(w*d1) - K*Phi(w*d2) ]
    m  = (Pn(t,Ti-1)/Pn(t,Ti))*(Pr(t,Ti)/Pr(t,Ti-1))*e^C            (2.6.2)
    d1 = [ln(m/K) + V^2/2]/V ,  d2 = d1 - V ,  K = 1 + kappa        (2.6.5)

`m` is the Ti-forward expectation of the index ratio and `C` is tranche 4's convexity -- composed,
not restated. `V^2` is eq 2.6.4, oracled separately in `tests/L3/test_jy_variance.py`.
"""

import math
from datetime import date, timedelta

import pytest
from scipy.stats import norm

from pricebook.engine import price
from pricebook.engine.yoy import yoy_inflation_leg_value
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    InflationIndex,
    Money,
    PricingFailure,
    PricingResult,
    YoYLeg,
)
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models import HullWhiteParams, JarrowYildirim, JYCorrelations, JYParams
from pricebook.products import InflationCapFloor, YearOnYearInflationSwap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
HICP = InflationIndex("HICPxt")
N = 1_000_000.0
DATES = tuple(VAL + timedelta(days=365 * k) for k in range(4))   # start + 3 annual periods
SCHEDULE = CouponSchedule(DATES, DC)
NOMINAL_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
REAL_KEY = CurveKey(CurveRole.REAL, EUR)

NOM_P = HullWhiteParams(0.05, 0.010)
REAL_P = HullWhiteParams(0.04, 0.008)
SI = 0.012
CORR = JYCorrelations(nominal_real=0.30, real_index=-0.20, nominal_index=0.15)
PARAMS = JYParams(NOM_P, REAL_P, SI, CORR)


def _model(params: JYParams = PARAMS, nominal: float = 0.03, real: float = 0.01) -> JarrowYildirim:
    return JarrowYildirim(
        MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet(
                {NOMINAL_KEY: DiscountCurve.flat(VAL, nominal, DC),
                 REAL_KEY: DiscountCurve.flat(VAL, real, DC)}
            ),
        ),
        params,
    )


def _cap(kappa: float = 0.02, kind: CapFloorKind = CapFloorKind.CAP) -> InflationCapFloor:
    return InflationCapFloor(HICP, SCHEDULE, kappa, Money(N, EUR), kind)


def _pv(product: InflationCapFloor, model: JarrowYildirim) -> float:
    out = price(product, model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _t(d: date) -> float:
    return (d - VAL).days / 365.0


def _reference_capfloor(kappa, kind, params=PARAMS, nominal=0.03, real=0.01):
    """Eq 2.6.5 per period, summed -- transcribed from the PDF, independent of the engine."""
    from pricebook.models import yoy_convexity, yoy_variance

    w = 1.0 if kind is CapFloorKind.CAP else -1.0
    strike = 1.0 + kappa
    total = 0.0
    for start, pay, psi in SCHEDULE.periods():
        t0, t1 = _t(start), _t(pay)
        pn0, pn1 = math.exp(-nominal * t0), math.exp(-nominal * t1)
        pr0, pr1 = math.exp(-real * t0), math.exp(-real * t1)
        c = yoy_convexity(0.0, t0, t1, params)
        m = (pn0 / pn1) * (pr1 / pr0) * math.exp(c)
        v = math.sqrt(yoy_variance(0.0, t0, t1, params))
        d1 = (math.log(m / strike) + 0.5 * v * v) / v
        d2 = d1 - v
        total += w * psi * pn1 * (m * norm.cdf(w * d1) - strike * norm.cdf(w * d2))
    return N * total


# ── ORACLE 1 (primary): eq 2.6.5 against an independent transcription ──────────────────────
def test_the_capfloor_matches_eq_2_6_5() -> None:
    for kappa in (-0.01, 0.0, 0.02, 0.05):
        for kind in (CapFloorKind.CAP, CapFloorKind.FLOOR):
            got = _pv(_cap(kappa, kind), _model())
            assert got == pytest.approx(_reference_capfloor(kappa, kind), rel=1e-11, abs=1e-8)


# ── ORACLE 2 (reduction) + ORACLE 3 (its companion) ────────────────────────────────────────
def test_with_both_rate_vols_zero_the_caplet_IS_plain_BLACK_on_the_forward_ratio() -> None:
    """`sigma_n = sigma_r = 0` gives `C = 0` and `V^2 = sigma_I^2*(Ti-Ti-1)`, so each caplet is a
    Black option on the *deterministic* forward ratio with vol `sigma_I*sqrt(delta)`. Exact, not a
    limit -- and it kills six of V^2's eight terms, which is why oracle 1 above is the correctness
    anchor and this is reduction-sanity."""
    flat = JYParams(HullWhiteParams(NOM_P.kappa, 0.0), HullWhiteParams(REAL_P.kappa, 0.0), SI, CORR)
    model = _model(flat)
    kappa, strike = 0.02, 1.02
    total = 0.0
    for start, pay, psi in SCHEDULE.periods():
        t0, t1 = _t(start), _t(pay)
        pn0, pn1 = math.exp(-0.03 * t0), math.exp(-0.03 * t1)
        pr0, pr1 = math.exp(-0.01 * t0), math.exp(-0.01 * t1)
        m = (pn0 / pn1) * (pr1 / pr0)                 # no e^C factor at all
        v = SI * math.sqrt(t1 - t0)                   # V = sigma_I*sqrt(delta)
        d1 = (math.log(m / strike) + 0.5 * v * v) / v
        total += psi * pn1 * (m * norm.cdf(d1) - strike * norm.cdf(d1 - v))
    assert _pv(_cap(kappa), model) == pytest.approx(N * total, rel=1e-11, abs=1e-8)


def test_the_rate_vols_MOVE_the_price_so_the_reduction_is_not_vacuous() -> None:
    """The charter's companion requirement: without it, the reduction passes an implementation
    that computed only `sigma_I^2*delta` and ignored seven of V^2's terms."""
    flat = JYParams(HullWhiteParams(NOM_P.kappa, 0.0), HullWhiteParams(REAL_P.kappa, 0.0), SI, CORR)
    # Rate vols well above the realistic set, so the effect is unambiguous rather than a threshold
    # tuned until it passed: at realistic vols the move is ~0.4% of the price, which is real but
    # too close to a round number to be convincing evidence on its own.
    loud = JYParams(HullWhiteParams(NOM_P.kappa, 0.04), HullWhiteParams(REAL_P.kappa, 0.03), SI, CORR)
    base = _pv(_cap(), _model(flat))
    assert _pv(_cap(), _model(loud)) > 1.2 * base
    assert _pv(_cap(), _model()) != pytest.approx(base, rel=1e-3)   # and the realistic set moves it too


# ── ORACLE 4: put-call parity, and the cross-product tie to tranche 4 ──────────────────────
def test_cap_minus_floor_is_the_forward_minus_strike_annuity() -> None:
    """From 2.6.1 with `Phi(x) + Phi(-x) = 1`: `cap - floor = N*sum(psi*Pn(Ti)*(m - K))`. Parity is
    invisible to magnitude checks and is where an `w` on the wrong term shows up."""
    from pricebook.models import yoy_convexity

    model = _model()
    for kappa in (0.0, 0.02, 0.04):
        strike = 1.0 + kappa
        want = 0.0
        for start, pay, psi in SCHEDULE.periods():
            t0, t1 = _t(start), _t(pay)
            pn0, pn1 = math.exp(-0.03 * t0), math.exp(-0.03 * t1)
            pr0, pr1 = math.exp(-0.01 * t0), math.exp(-0.01 * t1)
            m = (pn0 / pn1) * (pr1 / pr0) * math.exp(yoy_convexity(0.0, t0, t1, PARAMS))
            want += psi * pn1 * (m - strike)
        got = _pv(_cap(kappa), model) - _pv(_cap(kappa, CapFloorKind.FLOOR), model)
        assert got == pytest.approx(N * want, rel=1e-10, abs=1e-8)


def test_at_a_ZERO_strike_rate_cap_minus_floor_IS_the_T4_YOY_SWAP_LEG() -> None:
    """`K = 1` makes the parity right-hand side `N*sum(psi*Pn(Ti)*(m - 1))`, which is exactly
    tranche 4's YoY inflation leg (eq 2.5.13). **Cross-product, through different code** -- the
    cap/floor pricer and `yoy_inflation_leg_value` share no path."""
    model = _model()
    parity = _pv(_cap(0.0), model) - _pv(_cap(0.0, CapFloorKind.FLOOR), model)
    swap = YearOnYearInflationSwap(HICP, YoYLeg(SCHEDULE, 0.0), Money(N, EUR))
    assert parity == pytest.approx(yoy_inflation_leg_value(swap, model), rel=1e-10, abs=1e-8)


# ── ORACLE 6: the strike limits ─────────────────────────────────────────────────────────────
def test_a_deep_in_the_money_cap_approaches_the_forward_annuity() -> None:
    """`K -> 0`: every `Phi -> 1` and the caplet tends to `N*psi*Pn(Ti)*m`."""
    from pricebook.models import yoy_convexity

    model = _model()
    want = 0.0
    for start, pay, psi in SCHEDULE.periods():
        t0, t1 = _t(start), _t(pay)
        pn0, pn1 = math.exp(-0.03 * t0), math.exp(-0.03 * t1)
        pr0, pr1 = math.exp(-0.01 * t0), math.exp(-0.01 * t1)
        want += psi * pn1 * (pn0 / pn1) * (pr1 / pr0) * math.exp(yoy_convexity(0.0, t0, t1, PARAMS))
    # `rel=2e-3` is what "approaches" means here, not a loosened match: at K = 0.001 the two
    # Phi terms are 1 - O(1e-3) rather than exactly 1, so the limit is approached and not attained.
    assert _pv(_cap(-0.999), model) == pytest.approx(N * want, rel=2e-3)


def test_a_deep_out_of_the_money_cap_is_worthless_and_the_floor_is_not() -> None:
    model = _model()
    assert _pv(_cap(5.0), model) == pytest.approx(0.0, abs=1e-6)
    assert _pv(_cap(5.0, CapFloorKind.FLOOR), model) > 0.0


def test_a_cap_is_worth_more_than_a_higher_struck_cap() -> None:
    model = _model()
    prices = [_pv(_cap(k), model) for k in (0.0, 0.02, 0.04, 0.08)]
    assert prices == sorted(prices, reverse=True)
    assert all(p > 0.0 for p in prices)


# ── ORACLE 7: no vol at all is intrinsic ────────────────────────────────────────────────────
def test_with_NO_vol_anywhere_the_cap_is_the_intrinsic_on_the_forward() -> None:
    novol = JYParams(
        HullWhiteParams(NOM_P.kappa, 0.0), HullWhiteParams(REAL_P.kappa, 0.0), 0.0, CORR
    )
    model = _model(novol)
    want = 0.0
    for start, pay, psi in SCHEDULE.periods():
        t0, t1 = _t(start), _t(pay)
        pn0, pn1 = math.exp(-0.03 * t0), math.exp(-0.03 * t1)
        pr0, pr1 = math.exp(-0.01 * t0), math.exp(-0.01 * t1)
        m = (pn0 / pn1) * (pr1 / pr0)
        want += psi * pn1 * max(m - 1.02, 0.0)
    assert _pv(_cap(0.02), model) == pytest.approx(N * want, rel=1e-10, abs=1e-8)


# ── ORACLE 8: the carried contracts ─────────────────────────────────────────────────────────
def test_matured_is_PV_ZERO_and_a_missing_real_curve_is_a_failure() -> None:
    after = DATES[-1] + timedelta(days=1)
    matured = JarrowYildirim(
        MarketSnapshot(valuation_date=after,
                       curves=CurveSet({NOMINAL_KEY: DiscountCurve.flat(after, 0.03, DC)})),
        PARAMS,
    )
    out = price(_cap(), matured)
    assert isinstance(out, PricingResult) and out.pv.amount == pytest.approx(0.0, abs=1e-12)
    no_real = JarrowYildirim(
        MarketSnapshot(valuation_date=VAL,
                       curves=CurveSet({NOMINAL_KEY: DiscountCurve.flat(VAL, 0.03, DC)})),
        PARAMS,
    )
    assert isinstance(price(_cap(), no_real), PricingFailure)


def test_the_capfloor_round_trips_and_refuses_an_unknown_version() -> None:
    for kind in (CapFloorKind.CAP, CapFloorKind.FLOOR):
        product = _cap(0.025, kind)
        assert InflationCapFloor.from_dict(product.to_dict()) == product
    bad = _cap().to_dict()
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="reads up to"):
        InflationCapFloor.from_dict(bad)
