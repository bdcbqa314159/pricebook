"""Market Model II — modelling the forward inflation indices. BERHAN-2012 ch 4.

    Di(t) = sigma_I,i-1 [ tau_i sigma_n,i Fn/(1 + tau_i Fn) rho_I,n + sigma_I,i-1
                          - sigma_I,i rho_I,i ] (T_i-1 - t)
    m     = I(t,Ti)/I(t,Ti-1) e^{Di(t)} = [(1+tau Fn)/(1+tau Fr)] e^{Di(t)}
    V^2   = [sigma_I,i-1^2 + sigma_I,i^2 - 2 rho_I,i sigma_I,i-1 sigma_I,i](T_i-1 - t)   (4.2.2)
                                                      + sigma_I,i^2 (T_i - T_i-1)
    YoY leg later period = psi_i Pn(t,Ti) [ m - 1 ]                                      (4.1.6)

**MM-II's identity, in the source's own words: "contrary to Market Model I, the correction term does
not depend on the volatility of real rates."** There is no `sigma_r` anywhere in this model -- only
the real forward *levels* enter, through `1 + tau Fr`. That is its reason to exist, and
`test_NO_real_rate_volatility_can_reach_MMII` is the oracle for it.

**And "the formula above is exact only when the correlations between I(.,Ti-1) and Fn(.,Ti-1,Ti) are
assumed to be zero so that the nominal forward rate is zeroed out from Di"** -- a second exact
anchor that does NOT require zero volatility, which no other model in this library has.

## Two hazards in the source, both handled rather than guessed

**1. Eq 4.1.3's denominator disagrees with `Di`'s, in BOTH renderings.** 4.1.3 prints
`tau_i sigma_n,i Fn/(1 + Fn)` while `Di(t)` prints `.../(1 + tau_i Fn)`. So this is an internal
inconsistency in the source, not a rendering artefact -- the derivation-step corroboration rule
(charter section 4) fires and finds a *discrepancy* rather than a confirmation. `1 + tau_i Fn` is
taken as correct: it is the standard LMM coefficient, it is what Market Model I's freezing relation
uses consistently, `1 + Fn` adds a pure number to a rate, and `Di` is the formula actually used
downstream. Pinned by `test_the_denominator_carries_TAU_as_Di_prints_it`.

**2. `Vi(t)` is defined TWICE, for two different quantities.** Eq 4.2.2 defines it as
`sqrt([...][T_i-1 - t])`; a line before 4.2.4 redefines it as `sqrt(V_i^2(t) + sigma_I,i^2
(T_i - T_i-1))` -- the same symbol, augmented. The caplet needs the augmented one: 4.2.1 is the
distribution of the *forward*-index ratio at `T_i-1`, while the payoff is on the *spot* ratio at
`T_i`, so the extra `sigma_I,i^2 (T_i - T_i-1)` of diffusion from `T_i-1` to `T_i` must be there.
Dropping it is the missing-term trap, and `test_the_variance_is_NOT_just_sigma_I_squared_times_span`
is its guard.
"""

import math
from datetime import date, timedelta

import pytest

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
from pricebook.models import (
    HullWhiteParams,
    JarrowYildirim,
    JYCorrelations,
    JYParams,
    MarketModelI,
    MMICorrelations,
    MMIParams,
)
from pricebook.models.index_ratio import IndexRatioModel
from pricebook.models.market_model_ii import MarketModelII, MMIICorrelations, MMIIParams
from pricebook.products import InflationCapFloor, YearOnYearInflationSwap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
HICP = InflationIndex("HICPxt", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
N = 1_000_000.0
NOMINAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0), (0.026, 0.030, 0.034, 0.038))
REAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0), (0.008, 0.011, 0.013, 0.016))
MARKET = MarketSnapshot(
    valuation_date=VAL,
    curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): NOMINAL,
                     CurveKey(CurveRole.REAL, EUR): REAL}),
)
# **SEMI-ANNUAL, and the `tau` in `Di`'s denominator is why.** On annual ACT/365F periods `tau` is
# exactly 1.0, so `tau*Fn` and `Fn` are the same number and the two readings of eq 4.1.3's
# denominator (source hazard 1) are indistinguishable — my first fixture was annual and the
# discriminating test could not fail. With 180-day periods `tau ~ 0.4932`, so the `tau` is oracled
# here and everywhere else in this file. Fifth appearance of the degenerate-fixture trap (charter
# section 4), and the first time it hit a *source-discrepancy* oracle rather than a formula one.
DATES = tuple(VAL + timedelta(days=180 * k) for k in range(5))

CORR = MMIICorrelations(index_nominal=0.25, index_index=0.80)
PARAMS = MMIIParams(index_vol=0.012, nominal_vol=0.25, correlations=CORR)
MMII = MarketModelII(MARKET, PARAMS)


def _curve_ratio(start: date, pay: date) -> float:
    """`I(t,Ti)/I(t,Ti-1) = (1+tau Fn)/(1+tau Fr)`, straight off the curves (eq 3.1.5)."""
    return (NOMINAL.df(start) / NOMINAL.df(pay)) * (REAL.df(pay) / REAL.df(start))


def _tau_fn(start: date, pay: date) -> float:
    return NOMINAL.df(start) / NOMINAL.df(pay) - 1.0


def _reference_di(start: date, pay: date, p: MMIIParams = PARAMS) -> float:
    """`Di(t)`, transcribed independently. One index vol, so `sigma_I,i-1 = sigma_I,i`."""
    lead = MEASURE.year_fraction(start)
    tfn = _tau_fn(start, pay)
    si, sn = p.index_vol, p.nominal_vol
    rho_in, rho_ii = p.correlations.index_nominal, p.correlations.index_index
    return si * (sn * tfn / (1.0 + tfn) * rho_in + si - si * rho_ii) * lead


# ============================================================================================
# ORACLE 7 — the pointwise transcription (charter §4: properties never replace this).
# ============================================================================================

@pytest.mark.parametrize("k", [1, 2, 3])
def test_m_matches_an_independent_transcription_of_4_1_6(k):
    start, pay = DATES[k], DATES[k + 1]
    expected = _curve_ratio(start, pay) * math.exp(_reference_di(start, pay))
    assert MMII.expected_index_ratio(start, pay) == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_variance_matches_an_independent_transcription_of_4_2_2(k):
    start, pay = DATES[k], DATES[k + 1]
    lead = MEASURE.year_fraction(start)
    span = MEASURE.year_fraction(pay) - lead
    si, rho_ii = PARAMS.index_vol, CORR.index_index
    expected = (si * si + si * si - 2.0 * rho_ii * si * si) * lead + si * si * span
    assert MMII.index_ratio_variance(start, pay) == pytest.approx(expected, rel=1e-15)


def test_the_denominator_carries_TAU_as_Di_prints_it():
    """**Source hazard 1, pinned.** Eq 4.1.3 prints `1 + Fn` and `Di` prints `1 + tau_i Fn`, in both
    renderings — an internal inconsistency, so the derivation-step rule finds a discrepancy rather
    than a confirmation. `1 + tau Fn` is taken as correct (the standard LMM coefficient, what MM-I's
    freezing uses, and dimensionally sound). This asserts the two readings differ measurably, so the
    choice is not cosmetic."""
    start, pay = DATES[2], DATES[3]
    lead = MEASURE.year_fraction(start)
    tfn = _tau_fn(start, pay)
    si, sn = PARAMS.index_vol, PARAMS.nominal_vol
    rho_in, rho_ii = CORR.index_nominal, CORR.index_index
    with_tau = si * (sn * tfn / (1.0 + tfn) * rho_in + si - si * rho_ii) * lead
    # `tau*Fn/(1+Fn)`: the 4.1.3 print, with Fn = tau_fn/tau recovered.
    span = MEASURE.year_fraction(pay) - lead
    fn = tfn / span
    without_tau = si * (sn * tfn / (1.0 + fn) * rho_in + si - si * rho_ii) * lead
    assert abs(with_tau - without_tau) / abs(with_tau) > 1e-3, (with_tau, without_tau)
    assert MMII.expected_index_ratio(start, pay) == pytest.approx(
        _curve_ratio(start, pay) * math.exp(with_tau), rel=1e-15
    )


def test_the_variance_is_NOT_just_sigma_I_squared_times_span():
    """**Source hazard 2, pinned — the missing-term trap.** `Vi(t)` is defined twice in ch 4 for two
    different quantities; the caplet needs the augmented one, because 4.2.1 is the distribution of
    the FORWARD-index ratio at `T_i-1` while the payoff is on the SPOT ratio at `T_i`. Reusing
    MM-I's plain `sigma_I^2 * span` would silently drop the `Vi^2(t)` lead term."""
    start, pay = DATES[2], DATES[3]
    lead = MEASURE.year_fraction(start)
    span = MEASURE.year_fraction(pay) - lead
    mmi_shaped = PARAMS.index_vol**2 * span
    actual = MMII.index_ratio_variance(start, pay)
    assert actual > mmi_shaped
    assert abs(actual / mmi_shaped - 1.0) > 0.1, (actual, mmi_shaped)
    # and the lead term carries the correlation with the right sign: more correlated -> smaller
    less_correlated = MarketModelII(
        MARKET, MMIIParams(PARAMS.index_vol, PARAMS.nominal_vol,
                           MMIICorrelations(CORR.index_nominal, 0.10))
    )
    assert less_correlated.index_ratio_variance(start, pay) > actual


def test_at_PERFECT_index_correlation_the_lead_term_vanishes():
    """`rho_I,I = 1` collapses `sigma_I,i-1^2 + sigma_I,i^2 - 2 rho sigma sigma` to zero when the two
    index vols are equal — so the variance reduces to exactly `sigma_I^2 * span`, MM-I's shape. An
    exact reduction, and the one case where the two models' variances coincide."""
    perfect = MarketModelII(
        MARKET, MMIIParams(PARAMS.index_vol, PARAMS.nominal_vol,
                           MMIICorrelations(CORR.index_nominal, 1.0))
    )
    for k in (1, 2, 3):
        start, pay = DATES[k], DATES[k + 1]
        span = MEASURE.year_fraction(pay) - MEASURE.year_fraction(start)
        assert perfect.index_ratio_variance(start, pay) == pytest.approx(
            PARAMS.index_vol**2 * span, rel=1e-14
        )


# ============================================================================================
# ORACLE 1 — the FOUR-path zero-vol reduction.
# ============================================================================================

def test_at_ZERO_vol_FOUR_independent_paths_agree():
    """MM-II's `m`, MM-I's `m`, JY's `m` and the curve ratio directly — four independently-coded
    paths, one limit, sharing nothing but the curves."""
    mmii0 = MarketModelII(MARKET, MMIIParams(0.0, 0.0, MMIICorrelations(0.0, 0.0)))
    mmi0 = MarketModelI(MARKET, MMIParams(0.0, 0.0, 0.0, MMICorrelations(0.0, 0.0)))
    jy0 = JarrowYildirim(
        MARKET,
        JYParams(HullWhiteParams(0.05, 0.010), HullWhiteParams(0.04, 0.0), 0.012,
                 JYCorrelations(0.30, -0.20, 0.15)),
    )
    for k in (1, 2, 3):
        start, pay = DATES[k], DATES[k + 1]
        target = _curve_ratio(start, pay)
        assert mmii0.expected_index_ratio(start, pay) == pytest.approx(target, rel=1e-14)
        assert mmi0.expected_index_ratio(start, pay) == pytest.approx(target, rel=1e-13)
        assert jy0.expected_index_ratio(start, pay) == pytest.approx(target, rel=1e-13)


# ============================================================================================
# ORACLE 2 — the SECOND exact anchor, unique to MM-II: exact at rho_I,n = 0.
# ============================================================================================

def test_at_ZERO_index_nominal_correlation_the_NOMINAL_FORWARD_drops_out_of_Di():
    """The source: "the formula above is exact only when the correlations between `I(.,Ti-1)` and
    `Fn(.,Ti-1,Ti)` are assumed to be zero so that the nominal forward rate is zeroed out from
    `Di`". So at `rho_I,n = 0` the freezing approximation is not used at all, and `Di` reduces to a
    closed form with no forward rate in it — an exact anchor that does NOT need zero volatility,
    which no other model here has."""
    exact = MarketModelII(
        MARKET, MMIIParams(PARAMS.index_vol, PARAMS.nominal_vol, MMIICorrelations(0.0, 0.80))
    )
    for k in (1, 2, 3):
        start, pay = DATES[k], DATES[k + 1]
        lead = MEASURE.year_fraction(start)
        si = PARAMS.index_vol
        closed_form = si * (si - si * 0.80) * lead      # no Fn, no nominal vol
        assert exact.expected_index_ratio(start, pay) == pytest.approx(
            _curve_ratio(start, pay) * math.exp(closed_form), rel=1e-15
        )


def test_at_zero_index_nominal_correlation_the_NOMINAL_VOL_is_inert():
    """The same statement from the other side: with `rho_I,n = 0`, changing `sigma_n` cannot move
    anything — because the only term it appears in is multiplied by that correlation."""
    a = MarketModelII(MARKET, MMIIParams(0.012, 0.10, MMIICorrelations(0.0, 0.80)))
    b = MarketModelII(MARKET, MMIIParams(0.012, 0.90, MMIICorrelations(0.0, 0.80)))
    start, pay = DATES[2], DATES[3]
    assert a.expected_index_ratio(start, pay) == b.expected_index_ratio(start, pay)


# ============================================================================================
# ORACLE 5 — MM-II's IDENTITY: no real-rate volatility, anywhere.
# ============================================================================================

def test_NO_real_rate_volatility_can_reach_MMII():
    """**The whole reason MM-II exists**, in the source's words: "contrary to Market Model I, the
    correction term does not depend on the volatility of real rates."

    Proved structurally rather than by inspection: `MMIIParams` has no real-vol field at all, so
    there is nothing a caller could set. The positive half is that the real *curve* still enters —
    through the forward levels `1 + tau Fr` — so the model is not simply ignoring the real economy."""
    assert not any("real" in f and "vol" in f for f in PARAMS.__dataclass_fields__)
    assert set(PARAMS.__dataclass_fields__) == {"index_vol", "nominal_vol", "correlations"}
    assert set(CORR.__dataclass_fields__) == {"index_nominal", "index_index"}

    # The real curve DOES matter — a different real curve gives a different `m`.
    shifted = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): NOMINAL,
                         CurveKey(CurveRole.REAL, EUR): REAL.bumped(0.0020)}),
    )
    start, pay = DATES[2], DATES[3]
    assert MarketModelII(shifted, PARAMS).expected_index_ratio(start, pay) != pytest.approx(
        MMII.expected_index_ratio(start, pay), rel=1e-9
    )


def test_MMI_DOES_depend_on_real_rate_vol_but_only_WEAKLY_and_the_size_is_the_finding():
    """The companion that gives the oracle above its meaning: MM-I's `m` moves with `sigma_r` while
    MM-II has no such parameter at all. Without it, "MM-II has no real vol" could be true of a model
    with no parameters whatsoever.

    **And the measured size is itself the finding.** MM-I's real-vol dependence runs through
    `mu_y = [-sigma_r^2/2 - rho_rI sigma_I sigma_r]*(T_i-1 - t)`, so it grows with `|rho_rI|`, with
    `sigma_I` and with the lead — but it is small: sweeping `sigma_r` from 5% to 50%,

        lead 0.49y, rho_rI=-0.20 -> 7.7e-09      lead 0.49y, rho_rI=-0.90 -> 1.1e-05
        lead 0.99y, rho_rI=-0.20 -> 1.6e-06      lead 0.99y, rho_rI=-0.90 -> 2.4e-05
        lead 1.48y, rho_rI=-0.20 -> 6.6e-06      lead 1.48y, rho_rI=-0.90 -> 3.7e-05

    My first version of this test used `rho_rI = -0.20` at `rel=1e-6` and sat right on the boundary —
    it failed not because the dependence is absent but because it is *weak*. So MM-II's "no real-rate
    vol" is a genuine structural simplification whose **pricing** impact is modest at these
    parameters, which is worth stating rather than hiding behind a loose tolerance. The oracle asserts
    the TREND (growing in `|rho_rI|` and in the lead), which is stronger than one inequality."""
    def m(sigma_r: float, rho_ri: float, k: int) -> float:
        return MarketModelI(
            MARKET, MMIParams(0.25, sigma_r, 0.012, MMICorrelations(0.30, rho_ri))
        ).expected_index_ratio(DATES[k], DATES[k + 1])

    def gap(rho_ri: float, k: int) -> float:
        low, high = m(0.05, rho_ri, k), m(0.50, rho_ri, k)
        return abs(high - low) / low

    # It is present at all -- which is the half MM-II contrasts with.
    assert gap(-0.90, 3) > 1e-5, gap(-0.90, 3)
    # And it grows with the correlation's magnitude, and with the lead.
    assert gap(-0.90, 3) > gap(-0.20, 3), (gap(-0.90, 3), gap(-0.20, 3))
    assert gap(-0.90, 3) > gap(-0.90, 1), (gap(-0.90, 3), gap(-0.90, 1))
    # But it is modest -- the honest magnitude, asserted so a later change that inflates it shows.
    assert gap(-0.90, 3) < 1e-3, gap(-0.90, 3)


# ============================================================================================
# ORACLES 3, 4 — non-vacuity and cross-model disagreement.
# ============================================================================================

def test_the_vols_and_correlations_MOVE_MMIIs_m():
    start, pay = DATES[2], DATES[3]
    base = MMII.expected_index_ratio(start, pay)
    for params in (
        MMIIParams(0.030, 0.25, CORR),
        MMIIParams(0.012, 0.60, CORR),
        MMIIParams(0.012, 0.25, MMIICorrelations(0.90, 0.80)),
        MMIIParams(0.012, 0.25, MMIICorrelations(0.25, 0.10)),
    ):
        assert MarketModelII(MARKET, params).expected_index_ratio(start, pay) != pytest.approx(
            base, rel=1e-9
        )


def test_THREE_models_DISAGREE_away_from_the_limits():
    start, pay = DATES[2], DATES[3]
    jy = JarrowYildirim(
        MARKET,
        JYParams(HullWhiteParams(0.05, 0.010), HullWhiteParams(0.04, 0.008), 0.012,
                 JYCorrelations(0.30, -0.20, 0.15)),
    )
    mmi = MarketModelI(MARKET, MMIParams(0.25, 0.20, 0.012, MMICorrelations(0.30, -0.20)))
    values = [m.expected_index_ratio(start, pay) for m in (jy, mmi, MMII)]
    rounded = {round(v, 9) for v in values}
    assert len(rounded) == 3, f"models silently agree: {values}"


# ============================================================================================
# ORACLE 8 — a THIRD model through the tranche 12(a) seam, pricers untouched.
# ============================================================================================

def test_MMII_satisfies_the_protocol():
    assert isinstance(MMII, IndexRatioModel)


def test_the_pricers_price_MMII_with_no_change_to_them():
    product = InflationCapFloor(HICP, CouponSchedule(DATES, DC), 0.02, Money(N, EUR),
                                CapFloorKind.FLOOR)
    out = price(product, MMII)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount > 0.0
    swap = YearOnYearInflationSwap(HICP, YoYLeg(CouponSchedule(DATES, DC), 0.02), Money(N, EUR))
    assert yoy_inflation_leg_value(swap, MMII) != 0.0


def test_cap_MINUS_floor_under_MMII_is_still_the_YOY_LEG():
    """Tranche 5's cross-product identity, now carrying the THIRD model through the same seam."""
    schedule = CouponSchedule(DATES, DC)
    cap = price(InflationCapFloor(HICP, schedule, 0.0, Money(N, EUR), CapFloorKind.CAP), MMII)
    floor = price(InflationCapFloor(HICP, schedule, 0.0, Money(N, EUR), CapFloorKind.FLOOR), MMII)
    swap = YearOnYearInflationSwap(HICP, YoYLeg(schedule, 0.0), Money(N, EUR))
    assert isinstance(cap, PricingResult) and isinstance(floor, PricingResult)
    assert cap.pv.amount - floor.pv.amount == pytest.approx(
        yoy_inflation_leg_value(swap, MMII), rel=1e-11
    )


# ============================================================================================
# ORACLE 9 — the freezing bound: magnitude and TREND, never a smallness threshold.
# ============================================================================================

def test_the_freezing_error_GROWS_with_maturity_as_the_source_warns():
    """The source: "the approximation used when freezing the drift may be rough for longer
    maturities." Asserted as a TREND plus a measured magnitude, not an arbitrary bound — the
    tranche 12(b) lesson that a smallness threshold dresses an error as benign."""
    shock = 0.0020
    sensitivities = []
    for k in (1, 2, 3):
        start, pay = DATES[k], DATES[k + 1]
        lead = MEASURE.year_fraction(start)
        tfn = _tau_fn(start, pay)
        si, sn = PARAMS.index_vol, PARAMS.nominal_vol
        rho_in = CORR.index_nominal
        span = MEASURE.year_fraction(pay) - lead
        base = si * sn * tfn / (1.0 + tfn) * rho_in * lead
        moved = si * sn * (tfn + shock * span) / (1.0 + tfn + shock * span) * rho_in * lead
        sensitivities.append(abs(moved - base))
    assert all(s > 0.0 for s in sensitivities), sensitivities
    assert sensitivities == sorted(sensitivities), (
        f"the freezing sensitivity does not grow with maturity: {sensitivities}"
    )
    assert sensitivities[-1] / sensitivities[0] > 2.0, sensitivities


# ============================================================================================
# ORACLE 10 — construction contracts.
# ============================================================================================

def test_a_NEGATIVE_volatility_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        MMIIParams(-0.01, 0.25, CORR)


def test_an_OUT_OF_RANGE_correlation_is_refused():
    with pytest.raises(ValueError, match="must lie"):
        MMIICorrelations(0.0, 1.5)
