"""Market Model I — an LMM for nominal and real forward rates. BERHAN-2012 ch 3.

    (X_i, Y_i) ~ bivariate normal, mean (mu_x, mu_y), correlation rho_nr                 (3.1.7-8)
      mu_x = -1/2 sigma_n^2 (T_{i-1} - t),            sigma_x = sigma_n sqrt(T_{i-1} - t)
      mu_y = [-1/2 sigma_r^2 - rho_Ir sigma_I sigma_r](T_{i-1} - t),
                                                      sigma_y = sigma_r sqrt(T_{i-1} - t)

    m_i = E[ (1 + tau F_n e^{rho_nr sigma_x z - 1/2 sigma_x^2 rho_nr^2})
             / (1 + tau F_r e^{mu_y + sigma_y z}) ]                                      (3.1.11)

    YoY leg later period = psi_i Pn(t,T_i) [ m_i - 1 ]

**The transcription is corroborated by a SECOND PATH INSIDE THE SOURCE, which is stronger than two
renderings.** The `-layout` rendering of 3.1.11 is column-mangled, so the dual-rendering rule could
not settle it. But page 25 gives the integral in its PRE-substitution form -- over `y` with the
conditional density, before `z = (y - mu_y)/sigma_y` -- and the two must agree. They do, to 2.2e-16
(`test_the_integrand_matches_the_sources_PRE_substitution_form`). A misprint would have to appear
identically in both forms to survive that, which is a far stronger guard than a second rendering of
one printed line.

**Where the approximation actually lives, which is NOT in 3.1.11.** The spec expected the freezing
procedure to make the leg value approximate. It does not: given its per-period inputs, 3.1.11 is an
exact expectation. Mercurio's freezing governs the *recursion between periods* --

    sigma_I,i-1 ~= sigma_I,i + sigma_r tau F_r(0)/(1 + tau F_r(0))
                             - sigma_n tau F_n(0)/(1 + tau F_n(0))

-- where the exact relation carries the *stochastic* `F(t)` rather than its time-0 value. So the
approximation is a statement about the index-vol term structure, and its error is bounded here by
how far the frozen coefficient can move when the forward moves (`test_the_freezing_error_is_BOUNDED
_by_how_far_the_coefficient_moves`), which is a measured, justified number. A distributional bound
on it would need an LMM simulation -- a separate unit, and said so rather than faked.
"""

import math
from datetime import date, timedelta

import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pricebook.engine import price
from pricebook.engine.yoy import yoy_inflation_leg_value
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    IndexInterpolation,
    InflationIndex,
    Money,
    PricingResult,
    YoYLeg,
)
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models import HullWhiteParams, JarrowYildirim, JYCorrelations, JYParams
from pricebook.models.index_ratio import IndexRatioModel
from pricebook.models.market_model_i import (
    MarketModelI,
    MMICorrelations,
    MMIParams,
    index_vol_step,
)
from pricebook.products import InflationCapFloor, YearOnYearInflationSwap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
HICP = InflationIndex("HICPxt", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
N = 1_000_000.0
# SLOPED curves, so the forward rates differ by period and a per-period error cannot hide.
NOMINAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0), (0.026, 0.030, 0.034, 0.038))
REAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0), (0.008, 0.011, 0.013, 0.016))
MARKET = MarketSnapshot(
    valuation_date=VAL,
    curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): NOMINAL,
                     CurveKey(CurveRole.REAL, EUR): REAL}),
)
DATES = tuple(VAL + timedelta(days=365 * k) for k in range(5))

CORR = MMICorrelations(nominal_real=0.30, real_index=-0.20)
PARAMS = MMIParams(nominal_vol=0.25, real_vol=0.20, index_vol=0.012, correlations=CORR)
MMI = MarketModelI(MARKET, PARAMS)


def _forwards(start: date, pay: date) -> tuple[float, float, float]:
    """`(tau, tau*Fn, tau*Fr)` for a period, from the curves — `1 + tau*F = P(start)/P(pay)`."""
    tau = MEASURE.year_fraction(pay) - MEASURE.year_fraction(start)
    return (tau,
            NOMINAL.df(start) / NOMINAL.df(pay) - 1.0,
            REAL.df(start) / REAL.df(pay) - 1.0)


# ============================================================================================
# The transcription, against the source's own PRE-substitution form.
# ============================================================================================

def test_the_integrand_matches_the_sources_PRE_substitution_form():
    """Page 25's form is over `y` with `f_Y(y)`; 3.1.11 is over `z` after the substitution. They are
    the same quantity, so agreement is a transcription check a single misprint cannot survive."""
    start, pay = DATES[2], DATES[3]
    _, tfn, tfr = _forwards(start, pay)
    lead = MEASURE.year_fraction(start)
    sn, sr, si = PARAMS.nominal_vol, PARAMS.real_vol, PARAMS.index_vol
    rho_nr, rho_ir = CORR.nominal_real, CORR.real_index
    sx, sy = sn * math.sqrt(lead), sr * math.sqrt(lead)
    mux = -0.5 * sn * sn * lead
    muy = (-0.5 * sr * sr - rho_ir * si * sr) * lead

    def pre(y):
        num = 1.0 + tfn * math.exp(
            mux + rho_nr * sx * (y - muy) / sy + 0.5 * sx * sx * (1.0 - rho_nr**2)
        )
        return num / (1.0 + tfr * math.exp(y)) * norm.pdf(y, loc=muy, scale=sy)

    reference = quad(pre, muy - 12 * sy, muy + 12 * sy, limit=200)[0]
    assert MMI.expected_index_ratio(start, pay) == pytest.approx(reference, rel=1e-10)


def test_the_integrand_matches_an_independent_POST_substitution_transcription():
    start, pay = DATES[1], DATES[2]
    _, tfn, tfr = _forwards(start, pay)
    lead = MEASURE.year_fraction(start)
    sn, sr, si = PARAMS.nominal_vol, PARAMS.real_vol, PARAMS.index_vol
    rho_nr, rho_ir = CORR.nominal_real, CORR.real_index
    sx, sy = sn * math.sqrt(lead), sr * math.sqrt(lead)
    muy = (-0.5 * sr * sr - rho_ir * si * sr) * lead

    def post(z):
        num = 1.0 + tfn * math.exp(rho_nr * sx * z - 0.5 * sx * sx * rho_nr**2)
        return num / (1.0 + tfr * math.exp(muy + sy * z)) * norm.pdf(z)

    reference = quad(post, -12, 12, limit=200)[0]
    assert MMI.expected_index_ratio(start, pay) == pytest.approx(reference, rel=1e-10)


# ============================================================================================
# THE ANCHOR — the cross-model reduction. Three independent paths, one limit.
# ============================================================================================

def test_at_ZERO_vol_MMI_and_JY_and_the_MODEL_FREE_forward_all_agree_exactly():
    """The strongest oracle in the tranche: at zero volatility MM-I's integral collapses to the
    deterministic forward ratio, JY's `e^C` collapses to 1, and the curves give the same number
    directly. **Three independently-coded paths, one limit** — and no parameter is shared between
    them except the curves."""
    flat = MarketModelI(MARKET, MMIParams(0.0, 0.0, 0.0, MMICorrelations(0.0, 0.0)))
    jy_no_real_vol = JarrowYildirim(
        MARKET,
        JYParams(HullWhiteParams(0.05, 0.010), HullWhiteParams(0.04, 0.0), 0.012,
                 JYCorrelations(0.30, -0.20, 0.15)),
    )
    for k in range(1, 4):
        start, pay = DATES[k], DATES[k + 1]
        model_free = (NOMINAL.df(start) / NOMINAL.df(pay)) * (REAL.df(pay) / REAL.df(start))
        assert flat.expected_index_ratio(start, pay) == pytest.approx(model_free, rel=1e-13)
        assert jy_no_real_vol.expected_index_ratio(start, pay) == pytest.approx(
            model_free, rel=1e-13
        )


def test_the_reduction_is_NOT_vacuous_the_vols_move_MMIs_ratio():
    """The mandatory companion: an implementation ignoring its parameters would pass the anchor."""
    start, pay = DATES[2], DATES[3]
    base = MMI.expected_index_ratio(start, pay)
    flat = MarketModelI(MARKET, MMIParams(0.0, 0.0, 0.0, MMICorrelations(0.0, 0.0)))
    assert base != pytest.approx(flat.expected_index_ratio(start, pay), rel=1e-6)
    moved = MarketModelI(
        MARKET, MMIParams(0.40, 0.30, 0.012, CORR)
    ).expected_index_ratio(start, pay)
    assert moved != pytest.approx(base, rel=1e-6)


def test_MMI_and_JY_DISAGREE_at_real_volatility_because_they_are_different_models():
    """They must not accidentally coincide away from the limit — that would mean one of them is not
    doing what it claims."""
    jy = JarrowYildirim(
        MARKET,
        JYParams(HullWhiteParams(0.05, 0.010), HullWhiteParams(0.04, 0.008), 0.012,
                 JYCorrelations(0.30, -0.20, 0.15)),
    )
    start, pay = DATES[2], DATES[3]
    assert MMI.expected_index_ratio(start, pay) != pytest.approx(
        jy.expected_index_ratio(start, pay), rel=1e-6
    )


# ============================================================================================
# Requirements that survive a misprint.
# ============================================================================================

def test_the_expected_ratio_is_always_POSITIVE():
    """The integrand is a ratio of two positive quantities, so no parameter set can make it
    negative. A requirement derived from the structure, not from the printed formula."""
    for sn in (0.0, 0.1, 0.5):
        for sr in (0.0, 0.1, 0.5):
            for rho in (-0.9, 0.0, 0.9):
                m = MarketModelI(
                    MARKET, MMIParams(sn, sr, 0.012, MMICorrelations(rho, -0.2))
                ).expected_index_ratio(DATES[2], DATES[3])
                assert m > 0.0, (sn, sr, rho)


def test_the_cap_floor_variance_is_the_index_vol_squared_times_the_PERIOD():
    """§3.2's inner expectation is Black with `sigma_I^2 (T_i - T_{i-1})` — the period, as for JY's
    simplified limit, and NOT the expiry (tranche 9's distinction)."""
    for k in range(1, 4):
        start, pay = DATES[k], DATES[k + 1]
        span = MEASURE.year_fraction(pay) - MEASURE.year_fraction(start)
        assert MMI.index_ratio_variance(start, pay) == pytest.approx(
            PARAMS.index_vol**2 * span, rel=1e-15
        )


def test_a_NEGATIVE_volatility_is_refused_at_construction():
    with pytest.raises(ValueError, match="cannot be negative"):
        MMIParams(-0.1, 0.2, 0.012, CORR)


def test_an_OUT_OF_RANGE_correlation_is_refused():
    with pytest.raises(ValueError, match="must lie"):
        MMICorrelations(1.5, 0.0)


# ============================================================================================
# The freezing procedure — where the approximation actually is, with a MEASURED bound.
# ============================================================================================

def test_the_index_vol_step_vanishes_when_BOTH_rate_vols_are_zero():
    """An exact reduction: the adjustment is a difference of terms each carrying a rate vol as a
    factor, so with no rate volatility the index-vol term structure is flat."""
    _, tfn, tfr = _forwards(DATES[2], DATES[3])
    assert index_vol_step(0.012, 0.0, 0.0, tfn, tfr) == 0.012


def test_the_index_vol_step_MOVES_with_the_rate_vols():
    _, tfn, tfr = _forwards(DATES[2], DATES[3])
    base = index_vol_step(0.012, 0.0, 0.0, tfn, tfr)
    assert index_vol_step(0.012, 0.25, 0.20, tfn, tfr) != pytest.approx(base, rel=1e-9)


def test_the_freezing_error_is_BOUNDED_by_how_far_the_coefficient_moves():
    """**The approximation's error bar, measured rather than asserted.** The freezing replaces the
    stochastic `tau F(t)/(1 + tau F(t))` with its time-0 value. That map is bounded and monotone in
    `F`, so shocking the forward by a stated amount bounds the coefficient's move, and hence the
    error in `sigma_I,i-1`.

    This is a SENSITIVITY bound, not a distributional one: a probability-weighted bound would need
    an LMM simulation of the forward rates, which this tranche does not build and does not pretend
    to. The number is reported in the tranche doc."""
    tau, tfn, tfr = _forwards(DATES[2], DATES[3])
    shock = 0.02 * tau       # a 200bp move in the forward rate
    base = index_vol_step(0.012, PARAMS.nominal_vol, PARAMS.real_vol, tfn, tfr)
    moved = [
        index_vol_step(0.012, PARAMS.nominal_vol, PARAMS.real_vol, tfn + dn, tfr + dr)
        for dn in (-shock, shock) for dr in (-shock, shock)
    ]
    worst = max(abs(m - base) for m in moved)
    assert worst > 0.0, "the coefficient is inert in the forwards — then nothing is being frozen"
    assert worst < 0.01, f"a 200bp shock moves sigma_I by {worst:.5f}, which needs explaining"


# ============================================================================================
# THE POINT OF TRANCHE 12(a): MM-I flows through the SAME pricers, with no change to them.
# ============================================================================================

def test_MMI_satisfies_the_protocol():
    assert isinstance(MMI, IndexRatioModel)


def test_the_YOY_pricer_prices_MMI_with_no_change_to_the_pricer():
    swap = YearOnYearInflationSwap(HICP, YoYLeg(CouponSchedule(DATES, DC), 0.02), Money(N, EUR))
    jy = JarrowYildirim(
        MARKET,
        JYParams(HullWhiteParams(0.05, 0.010), HullWhiteParams(0.04, 0.008), 0.012,
                 JYCorrelations(0.30, -0.20, 0.15)),
    )
    mmi_leg = yoy_inflation_leg_value(swap, MMI)
    jy_leg = yoy_inflation_leg_value(swap, jy)
    assert mmi_leg != 0.0
    assert mmi_leg != pytest.approx(jy_leg, rel=1e-6), "two models gave one answer"


def test_the_CAPFLOOR_pricer_prices_MMI_with_no_change_to_the_pricer():
    product = InflationCapFloor(HICP, CouponSchedule(DATES, DC), 0.02, Money(N, EUR),
                                CapFloorKind.FLOOR)
    out = price(product, MMI)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount > 0.0


def test_cap_MINUS_floor_under_MMI_is_still_the_YOY_LEG():
    """The cross-product identity from tranche 5, now with a second model behind it — so it tests
    the seam as well as the formula. At a zero strike, `cap − floor` is the year-on-year leg."""
    schedule = CouponSchedule(DATES, DC)
    cap = price(InflationCapFloor(HICP, schedule, 0.0, Money(N, EUR), CapFloorKind.CAP), MMI)
    floor = price(InflationCapFloor(HICP, schedule, 0.0, Money(N, EUR), CapFloorKind.FLOOR), MMI)
    swap = YearOnYearInflationSwap(HICP, YoYLeg(schedule, 0.0), Money(N, EUR))
    assert isinstance(cap, PricingResult) and isinstance(floor, PricingResult)
    assert cap.pv.amount - floor.pv.amount == pytest.approx(
        yoy_inflation_leg_value(swap, MMI), rel=1e-11
    )
