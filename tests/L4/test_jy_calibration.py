"""Calibrating Jarrow-Yildirim's vols to a YoY floor strip. BERHAN-2012 ch 5.

Two-stage, and only the second stage is fitted here: the curve stage is pinned by tranche 2
(nominal curve + real curve from ZCIIS break-evens, model-free) and is NOT refitted against vol
quotes -- refitting curves against vol quotes launders a curve error into a vol.

Fixed: `kappa_n`, `kappa_r`, the three correlations, and `sigma_n` (the thesis sources it from
*nominal* ATM caps, an engine this library has no reason to own yet -- so it is a named input).
Free: `(sigma_r, sigma_I)` against a strip of floors, overdetermined.

**The stock's `jy_calibrate` is NOT the model here, and that is the headline of tranche 6.** It
fits vols to a ZCIIS strip -- an instrument carrying no vol information at all under correct JY
(`Pr/Pn - 1`, eq 1.2.13/1.2.16; it is *because* it is model-free that tranche 2 built the real
curve from it). Its objective is non-flat only through a model-implied Vasicek `theta=r0` curve
and a convexity misapplied to the zero-coupon swap. See `docs/tranches/06_jy_calibration.md` §0.

The oracle ladder here, and why each rung is present:
  A1  parameter recovery from synthetic quotes       -- the headline check, and VACUOUS ALONE
  A2a per-parameter sensitivity: not FLAT            -- A1's first discriminator
  A2b multi-start uniqueness + conditioning: not DEGENERATE -- A1's second (Cowork, thread 0035)
  A3  residual at the truth, and the fit descends
  A4  cross-stage: the fit is invariant to how the real curve was EXPRESSED
  A5  failure is a value
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine.jy_calibration import (
    JYCalibration,
    JYFixedParams,
    calibrate_jy_to_yoy_floors,
    jy_params_from,
)
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    InflationIndex,
    Money,
    PricingFailure,
    PricingResult,
)
from pricebook.market import (
    CurveKey,
    CurveRole,
    CurveSet,
    DiscountCurve,
    MarketSnapshot,
    real_curve_from_breakeven,
)
from pricebook.models import JarrowYildirim, JYCorrelations
from pricebook.products import InflationCapFloor

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
HICP = InflationIndex("HICPxt")
N = 1_000_000.0
NOMINAL_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
REAL_KEY = CurveKey(CurveRole.REAL, EUR)

# SLOPED curves, deliberately. A flat pair has twice produced an oracle that could not fail
# (a clock advance and an expiry bump agreeing to the last bit; interpolation indistinguishable
# from any scheme). Here the strip's maturity structure is what identifies two parameters, so a
# degenerate curve is exactly the wrong fixture.
NOM_PILLARS = ((0.0, 0.028), (2.0, 0.032), (5.0, 0.036), (10.0, 0.038))
REAL_PILLARS = ((0.0, 0.008), (2.0, 0.011), (5.0, 0.014), (10.0, 0.015))

CORR = JYCorrelations(nominal_real=0.30, real_index=-0.20, nominal_index=0.15)
FIXED = JYFixedParams(kappa_nominal=0.05, kappa_real=0.04, sigma_nominal=0.010, correlations=CORR)

# The truth the synthetic strip is generated from, and what A1 must recover.
SIGMA_R_TRUE = 0.0080
SIGMA_I_TRUE = 0.0120
TRUTH = (SIGMA_R_TRUE, SIGMA_I_TRUE)
MATURITIES = (2, 3, 4, 5, 6, 7)   # annual periods per floor -- six quotes, two free parameters


def _curve(pillars: tuple[tuple[float, float], ...]) -> DiscountCurve:
    from pricebook.foundation.time_measure import TimeMeasure

    return DiscountCurve(
        TimeMeasure(VAL, DC), tuple(t for t, _ in pillars), tuple(r for _, r in pillars)
    )


def _market(real: DiscountCurve | None = None) -> MarketSnapshot:
    curves = {NOMINAL_KEY: _curve(NOM_PILLARS), REAL_KEY: real or _curve(REAL_PILLARS)}
    return MarketSnapshot(valuation_date=VAL, curves=CurveSet(curves))


def _floor(n_periods: int, kappa: float = 0.02) -> InflationCapFloor:
    dates = tuple(VAL + timedelta(days=365 * k) for k in range(n_periods + 1))
    return InflationCapFloor(
        HICP, CouponSchedule(dates, DC), kappa, Money(N, EUR), CapFloorKind.FLOOR
    )


def _price(product: InflationCapFloor, market: MarketSnapshot, sigma_r, sigma_i) -> float:
    from pricebook.engine import price

    model = JarrowYildirim(market, jy_params_from(FIXED, sigma_r, sigma_i))
    out = price(product, model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _strip(market: MarketSnapshot, sigma_r=SIGMA_R_TRUE, sigma_i=SIGMA_I_TRUE):
    """The synthetic quote strip: six floors priced at a known parameter pair."""
    return [(_floor(n), _price(_floor(n), market, sigma_r, sigma_i)) for n in MATURITIES]


def _fit(market: MarketSnapshot, start=(0.015, 0.025), quotes=None) -> JYCalibration:
    out = calibrate_jy_to_yoy_floors(quotes or _strip(market), market, FIXED, start)
    assert isinstance(out, JYCalibration), out
    return out


# --------------------------------------------------------------------------------------------
# A1 -- parameter recovery. The headline, and on its own it proves nothing: see A2a/A2b.
# --------------------------------------------------------------------------------------------

def test_a1_recovers_the_generating_parameters():
    market = _market()
    fit = _fit(market, start=(0.015, 0.025))   # ~2x sigma_r, ~2x sigma_I away from truth
    assert fit.params.real.sigma == pytest.approx(SIGMA_R_TRUE, abs=1e-6)
    assert fit.params.index_vol == pytest.approx(SIGMA_I_TRUE, abs=1e-6)
    assert fit.n_quotes == len(MATURITIES)
    assert fit.rms_residual < 1e-6


def test_a1_fixed_parameters_pass_through_untouched():
    """The fit must move only the two free parameters -- a calibrator that quietly moves `sigma_n`
    or a correlation is fitting something it was told not to."""
    fit = _fit(_market())
    assert fit.params.nominal.sigma == FIXED.sigma_nominal
    assert fit.params.nominal.kappa == FIXED.kappa_nominal
    assert fit.params.real.kappa == FIXED.kappa_real
    assert fit.params.correlations == CORR


# --------------------------------------------------------------------------------------------
# A2a -- the objective is not FLAT. This is the stock's failure mode, as a test.
# --------------------------------------------------------------------------------------------

def test_a2a_each_free_parameter_moves_the_strip():
    market = _market()
    base = [pv for _, pv in _strip(market)]
    for label, (sr, si) in {
        "sigma_r +10%": (SIGMA_R_TRUE * 1.1, SIGMA_I_TRUE),
        "sigma_I +10%": (SIGMA_R_TRUE, SIGMA_I_TRUE * 1.1),
    }.items():
        moved = [pv for _, pv in _strip(market, sr, si)]
        worst = max(abs(a - b) for a, b in zip(base, moved))
        assert worst > 1e-4 * N, f"{label} barely moves the strip ({worst:.3e}) -- not identifiable"


def test_a2a_the_strip_is_not_a_row_of_identical_numbers():
    """A populated baseline is not a discriminating one: a strip of six equal quotes would satisfy
    a least-squares fit at any number of parameter pairs."""
    pvs = [pv for _, pv in _strip(_market())]
    assert len(set(round(pv, 6) for pv in pvs)) == len(pvs)
    assert all(pv > 0.0 for pv in pvs)


# --------------------------------------------------------------------------------------------
# A2b -- the objective is not DEGENERATE. Cowork, thread 0035: A2a proves "not flat" but for a
# TWO-parameter fit it cannot prove the optimum is UNIQUE. sigma_r and sigma_I can each move the
# strip a lot while moving it in nearly COLLINEAR directions -- a valley in which sigma_r up,
# sigma_I down cancels -- and then per-parameter sensitivity passes while the recovered pair is
# not identified, with A1 landing near truth only by luck of its start.
# --------------------------------------------------------------------------------------------

def test_a2b_multi_start_recovery_lands_on_one_point():
    """Uniqueness, threshold-free: four distinct starts must agree. If the objective had a valley
    rather than a basin, these would spread along it."""
    market = _market()
    quotes = _strip(market)
    starts = [(0.004, 0.006), (0.015, 0.025), (0.002, 0.030), (0.020, 0.004)]
    fits = [_fit(market, start=s, quotes=quotes) for s in starts]
    for s, fit in zip(starts, fits):
        assert fit.params.real.sigma == pytest.approx(SIGMA_R_TRUE, abs=1e-6), f"from {s}"
        assert fit.params.index_vol == pytest.approx(SIGMA_I_TRUE, abs=1e-6), f"from {s}"


def test_a2b_the_sensitivity_jacobian_is_well_conditioned():
    """The mechanism behind A2b: the two parameters' sensitivity columns must be INDEPENDENT, not
    merely non-zero. `sigma_I` enters `V^2` as `sigma_I^2 * delta` (flat within a period) while
    `sigma_r` enters through the `beta_r` terms and scales all of the convexity `C`, so across a
    maturity strip they separate -- but "should separate" is what the oracle has to prove."""
    market = _market()
    base = [pv for _, pv in _strip(market)]
    h = 1e-6
    col_r = [(b - a) / h for a, b in zip(base, [pv for _, pv in _strip(market, SIGMA_R_TRUE + h, SIGMA_I_TRUE)])]
    col_i = [(b - a) / h for a, b in zip(base, [pv for _, pv in _strip(market, SIGMA_R_TRUE, SIGMA_I_TRUE + h)])]

    # Condition number of the 2x2 Gram matrix, via its eigenvalues. kappa = 1 means orthogonal
    # columns; kappa -> infinity means collinear (the degenerate case this test exists to exclude).
    grr = sum(x * x for x in col_r)
    gii = sum(x * x for x in col_i)
    gri = sum(x * y for x, y in zip(col_r, col_i))
    trace, det = grr + gii, grr * gii - gri * gri
    disc = math.sqrt(max(trace * trace / 4.0 - det, 0.0))
    lo, hi = trace / 2.0 - disc, trace / 2.0 + disc
    assert lo > 0.0, "the Gram matrix is singular -- the two columns are collinear"
    assert hi / lo < 1e4, f"ill-conditioned sensitivity: kappa={hi / lo:.3e}"


# --------------------------------------------------------------------------------------------
# A3 -- the residual at the truth, and the fit actually descends.
# --------------------------------------------------------------------------------------------

def test_a3_the_objective_vanishes_at_the_generating_parameters():
    market = _market()
    quotes = _strip(market)
    at_truth = max(abs(_price(p, market, *TRUTH) - q) for p, q in quotes)
    assert at_truth < 1e-9, at_truth


def test_a3_the_fit_strictly_improves_on_a_deliberately_wrong_start():
    market = _market()
    quotes = _strip(market)
    start = (0.020, 0.004)
    before = math.sqrt(
        sum((_price(p, market, *start) - q) ** 2 for p, q in quotes) / len(quotes)
    )
    fit = _fit(market, start=start, quotes=quotes)
    assert before > 1e-3 * N, "the wrong start is not actually wrong -- the test proves nothing"
    assert fit.rms_residual < before / 1e6


# --------------------------------------------------------------------------------------------
# A4 -- cross-stage. The real curve built FROM break-evens must calibrate to the same vols as the
# curve those break-evens were implied from: break-evens -> curve -> fit, a round trip across
# tranche 2's inversion and tranche 6's fit, with different code on each side.
#
# NOT the version first proposed ("calibrated vols leave the ZCIIS fair rate unchanged"): the
# ZCIIS pricer takes a `DiscountingModel` and structurally cannot see a vol, so that check cannot
# fail and is worth nothing. Cowork endorsed the rejection (thread 0035).
# --------------------------------------------------------------------------------------------

def test_a4_the_fit_is_invariant_to_how_the_real_curve_was_expressed():
    direct = _market()
    nominal = _curve(NOM_PILLARS)
    real_direct = _curve(REAL_PILLARS)

    # Invert 1.2.16 to the break-evens this real curve implies, then rebuild it through tranche 2.
    quote_dates = tuple(VAL + timedelta(days=round(365 * t)) for t, _ in REAL_PILLARS[1:])
    breakevens = {
        d: math.expm1(nominal.zero_rate(t) - real_direct.zero_rate(t))
        for d, (t, _) in zip(quote_dates, REAL_PILLARS[1:])
    }
    rebuilt = _market(real_curve_from_breakeven(nominal, breakevens))

    # The two curves must agree before the fits can be compared -- otherwise this test passes by
    # comparing two calibrations to two different markets.
    for n in MATURITIES:
        d = VAL + timedelta(days=365 * n)
        assert rebuilt.curves.real(EUR).df(d) == pytest.approx(
            real_direct.df(d), rel=1e-12
        ), f"the break-even round trip moved the curve at {n}y"

    quotes = _strip(direct)
    fit = _fit(rebuilt, quotes=quotes)
    assert fit.params.real.sigma == pytest.approx(SIGMA_R_TRUE, abs=1e-6)
    assert fit.params.index_vol == pytest.approx(SIGMA_I_TRUE, abs=1e-6)


# --------------------------------------------------------------------------------------------
# A5 -- failure is a value (engine contract 4), never a NaN and never a raise.
# --------------------------------------------------------------------------------------------

def test_a5_no_quotes_is_a_failure_value():
    out = calibrate_jy_to_yoy_floors([], _market(), FIXED, (0.01, 0.01))
    assert isinstance(out, PricingFailure)
    assert "quote" in out.reason


def test_a5_an_unpriceable_quote_is_a_failure_value():
    """A market with no real curve cannot price a floor, so it cannot calibrate to one. The
    calibrator must return that as a value rather than let a `KeyError` out of the solver."""
    no_real = MarketSnapshot(
        valuation_date=VAL, curves=CurveSet({NOMINAL_KEY: _curve(NOM_PILLARS)})
    )
    out = calibrate_jy_to_yoy_floors(
        [(_floor(3), 1234.0)], no_real, FIXED, (0.01, 0.01)
    )
    assert isinstance(out, PricingFailure)


def test_a5_a_non_positive_start_is_a_failure_value():
    out = calibrate_jy_to_yoy_floors(_strip(_market()), _market(), FIXED, (0.0, 0.01))
    assert isinstance(out, PricingFailure)
