"""The full six-parameter JY calibration, and reproducing WHY it fails. BERHAN-2012 §5.3.2.

§5.3.2 frees all six of `{kappa_r, sigma_r, sigma_I, rho_nr, rho_rI, rho_nI}` against the
year-on-year floor **surface** and reports the fit fails: *"we cannot fit to the whole surface"*,
*"the presence of a strike skew makes calibration unfeasible for non-ATM contracts"*.

**So "the fit matches the market" is NOT the oracle** -- it would assert something the source says
is false. The deliverable is: the machinery is correct, and the documented failure is reproduced
AND explained.

**There are TWO distinct failures, and the thesis names only one.**

1. **UNDERDETERMINATION, and it is structural.** In `yoy_variance` the index correlations enter
   **only as products with `sigma_I`**:

       total += si*si*span
       total += 2*rho_ni * (sn*si/kn) * (span - beta_n)
       total -= 2*rho_ri * (sr*si/kr) * (span - beta_r)

   so `V^2` is a quadratic in `sigma_I` whose linear coefficient is a single POOLED combination of
   `(rho_nI, rho_rI)`; and `rho_nI` is **absent from `yoy_convexity` entirely**, so a floor price
   sees it only through that pooled product. Three of the six are therefore not identifiable from
   YoY floors **at any strike, by construction** -- which holds on perfect ATM data, *before any
   skew enters*. BERHAN does not say this.

2. **MISSPECIFICATION -- the flat smile.** JY is Gaussian, so its implied YoY-cap vol is constant
   across strikes. That is what makes a market skew unfittable, and it is the failure §5.3.2 names.

**No condition number is asserted anywhere.** The smallest singular value of the six-parameter
Jacobian sits below the finite-difference noise floor -- measured, it swings three orders of
magnitude with the step (2.5e7 -> 2.1e10) and differs again under central differences. A threshold
on it would assert noise and flake across the two CI operating systems. The discriminator is the
**multi-start spread**, which is behavioural, step-independent and identical on both OSes.
`s5/s1 ~ 3.4e-6` is the only stable number and is documented, never gated (charter §4).

Oracle 1 leans on T4/T5's forward being independently oracled against eqs 2.5.11/2.6.5 -- the
recovery here proves the *solver*, not the formula.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine.jy_calibration import JYFixedParams
from pricebook.engine.jy_full_calibration import (
    JYFreeParams,
    JYSurfaceCalibration,
    calibrate_jy_to_floor_surface,
    jy_params_from_free,
)
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    IndexInterpolation,
    InflationIndex,
    Money,
    PricingFailure,
    PricingResult,
)
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models import HullWhiteParams, JarrowYildirim, JYCorrelations
from pricebook.products import InflationCapFloor

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
HICP = InflationIndex("HICPxt", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
N = 1.0
KEY_N = CurveKey(CurveRole.DISCOUNT, EUR)
KEY_R = CurveKey(CurveRole.REAL, EUR)
MARKET = MarketSnapshot(
    valuation_date=VAL,
    curves=CurveSet({KEY_N: DiscountCurve.flat(VAL, 0.03, DC),
                     KEY_R: DiscountCurve.flat(VAL, 0.01, DC)}),
)

# (kappa_n, sigma_n) come FIXED from §5.3.1's fit (tranche 10b) and are never re-fitted here --
# refitting them against YoY floors would launder the nominal-cap calibration into the inflation fit.
FIXED = JYFixedParams(kappa_nominal=0.08, kappa_real=0.0, sigma_nominal=0.009,
                      correlations=JYCorrelations(0.0, 0.0, 0.0))

TRUE_FREE = JYFreeParams(
    real=HullWhiteParams(0.04, 0.008),
    index_vol=0.012,
    correlations=JYCorrelations(nominal_real=0.30, real_index=-0.20, nominal_index=0.15),
)
IDENTIFIABLE = ("kappa_real", "sigma_real", "nominal_real")
UNIDENTIFIABLE = ("index_vol", "real_index", "nominal_index")

ATM = 0.02
MATURITIES = (2, 3, 4, 5, 6, 7)
SKEW_STRIKES = (0.00, 0.01, 0.02, 0.03, 0.04)

STARTS = (
    (0.10, 0.020, 0.030, 0.00, 0.00, 0.00),
    (0.02, 0.004, 0.006, -0.50, 0.50, -0.50),
    (0.30, 0.050, 0.050, 0.80, -0.80, 0.80),
)


def _floor(n_periods: int, strike: float) -> InflationCapFloor:
    dates = tuple(VAL + timedelta(days=365 * k) for k in range(n_periods + 1))
    return InflationCapFloor(
        HICP, CouponSchedule(dates, DC), strike, Money(N, EUR), CapFloorKind.FLOOR
    )


def _pv(n_periods: int, strike: float, free: JYFreeParams) -> float:
    from pricebook.engine import price

    model = JarrowYildirim(MARKET, jy_params_from_free(FIXED, free))
    out = price(_floor(n_periods, strike), model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _atm_quotes(free: JYFreeParams = TRUE_FREE):
    return [(_floor(n, ATM), _pv(n, ATM, free)) for n in MATURITIES]


def _surface_quotes(free: JYFreeParams = TRUE_FREE):
    return [(_floor(n, k), _pv(n, k, free)) for n in MATURITIES for k in SKEW_STRIKES]


def _fit(quotes, start) -> JYSurfaceCalibration:
    out = calibrate_jy_to_floor_surface(quotes, MARKET, FIXED, start)
    assert isinstance(out, JYSurfaceCalibration), out
    return out


def _as_six(free: JYFreeParams) -> dict:
    return {
        "kappa_real": free.real.kappa,
        "sigma_real": free.real.sigma,
        "index_vol": free.index_vol,
        "nominal_real": free.correlations.nominal_real,
        "real_index": free.correlations.real_index,
        "nominal_index": free.correlations.nominal_index,
    }


# ============================================================================================
# ORACLE 1 -- the solver works on what IS determined.
# ============================================================================================

def test_O1_the_identifiable_subset_is_recovered_from_every_start():
    """`(kappa_r, sigma_r, rho_nr)` are determined by the ATM strip, so every start must find them.
    This proves the SOLVER; the forward it inverts was independently oracled in T4/T5 against
    eqs 2.5.11 and 2.6.5."""
    quotes = _atm_quotes()
    truth = _as_six(TRUE_FREE)
    for start in STARTS:
        got = _as_six(_fit(quotes, start).free)
        for name in IDENTIFIABLE:
            assert got[name] == pytest.approx(truth[name], rel=1e-5), f"{name} from {start}"


def test_O1_the_fit_from_the_TRUTH_reproduces_the_truth_exactly():
    fit = _fit(_atm_quotes(), (0.04, 0.008, 0.012, 0.30, -0.20, 0.15))
    got, truth = _as_six(fit.free), _as_six(TRUE_FREE)
    for name in got:
        assert got[name] == pytest.approx(truth[name], abs=1e-9), name
    assert fit.rms_residual < 1e-12


def test_O1_the_quotes_are_DISTINCT_so_the_fit_sees_a_real_strip():
    pvs = [pv for _, pv in _atm_quotes()]
    assert len(set(round(p, 12) for p in pvs)) == len(pvs), pvs
    assert all(p > 0.0 for p in pvs)


# ============================================================================================
# ORACLE 2 -- THE DISCRIMINATOR. The multi-start spread, behavioural and platform-independent.
# ============================================================================================

def test_O2_three_parameters_CONVERGE_and_three_DO_NOT_on_perfect_ATM_data():
    """**The underdetermination, as a test.** Every fit reports an essentially perfect residual, and
    they still disagree about `(sigma_I, rho_rI, rho_nI)`. A calibration reporting all six converged
    would be hiding this valley."""
    quotes = _atm_quotes()
    fits = [_as_six(_fit(quotes, s).free) for s in STARTS]
    truth = _as_six(TRUE_FREE)
    spread = {n: max(f[n] for f in fits) - min(f[n] for f in fits) for n in truth}

    for name in IDENTIFIABLE:
        assert spread[name] < abs(truth[name]) * 1e-4, (
            f"{name} was expected to converge, spread {spread[name]:.3e}"
        )
    for name in UNIDENTIFIABLE:
        assert spread[name] > abs(truth[name]) * 0.05, (
            f"{name} was expected NOT to converge, yet its spread is only {spread[name]:.3e} — "
            "either the fixture lost its power or the model changed"
        )


def test_O2_the_non_convergence_coexists_with_a_PERFECT_residual():
    """The sharp edge of the finding: these are not failed fits. Each lands on an rms indistinguish-
    able from zero while sitting at a different parameter set — which is exactly why a residual is
    not evidence of identification."""
    quotes = _atm_quotes()
    for start in STARTS:
        fit = _fit(quotes, start)
        assert fit.rms_residual < 1e-8, (start, fit.rms_residual)


def test_O2_rho_nominal_index_is_the_WORST_determined():
    """It is absent from the convexity `C` altogether, so a floor price sees it only through the
    pooled `sigma_I * rho` term in `V^2` — the weakest channel of the three."""
    quotes = _atm_quotes()
    fits = [_as_six(_fit(quotes, s).free) for s in STARTS]
    spread = {n: max(f[n] for f in fits) - min(f[n] for f in fits) for n in _as_six(TRUE_FREE)}
    assert spread["nominal_index"] > spread["real_index"], spread
    assert spread["nominal_index"] > 0.1, spread


def test_O2_rho_nominal_index_is_ABSENT_from_the_convexity():
    """The structural reason, asserted against the code rather than described: moving `rho_nI`
    cannot change `C` at all, while moving `rho_rI` or `rho_nr` does."""
    from pricebook.models import yoy_convexity

    base = yoy_convexity(0.0, 2.0, 3.0, jy_params_from_free(FIXED, TRUE_FREE))
    bumped_ni = TRUE_FREE.with_correlations(
        JYCorrelations(0.30, -0.20, 0.90)
    )
    bumped_ri = TRUE_FREE.with_correlations(
        JYCorrelations(0.30, 0.60, 0.15)
    )
    assert yoy_convexity(0.0, 2.0, 3.0, jy_params_from_free(FIXED, bumped_ni)) == base
    assert yoy_convexity(0.0, 2.0, 3.0, jy_params_from_free(FIXED, bumped_ri)) != base


def test_O2_the_index_correlations_enter_the_variance_ONLY_pooled_with_sigma_I():
    """At `sigma_I = 0` both index correlations become inert in `V^2`, because each appears only as
    a product with it. That is the pooling, proved rather than asserted."""
    from pricebook.models import yoy_variance

    zero_si = JYFreeParams(TRUE_FREE.real, 0.0, TRUE_FREE.correlations)
    other = JYFreeParams(TRUE_FREE.real, 0.0, JYCorrelations(0.30, 0.70, -0.80))
    a = yoy_variance(0.0, 2.0, 3.0, jy_params_from_free(FIXED, zero_si))
    b = yoy_variance(0.0, 2.0, 3.0, jy_params_from_free(FIXED, other))
    assert a == b, "the index correlations are not purely pooled with sigma_I"
    assert yoy_variance(0.0, 2.0, 3.0, jy_params_from_free(FIXED, TRUE_FREE)) != a


# ============================================================================================
# ORACLE 3 -- the flat smile. A GUARD, and near-tautological for our code. Not a discriminator.
# ============================================================================================

def test_O3_the_implied_smile_is_FLAT_across_strikes():
    """JY is Gaussian: eq 2.6.5 is Black with `V^2` (eq 2.6.4), and `V^2` contains no strike — so the
    implied vol is identical at every strike.

    **Near-tautological for this implementation**, and said so rather than dressed up: it cannot
    discriminate a wrong `V^2`, only a strike leaking into one. Kept as a documentation and
    regression guard, because it is the structural reason the market skew is unfittable."""
    from pricebook.engine.capfloor import caplet_value
    from pricebook.models.black import black_implied_vol

    model = JarrowYildirim(MARKET, jy_params_from_free(FIXED, TRUE_FREE))
    nominal, real = MARKET.curves.discount(EUR), MARKET.curves.real(EUR)
    anchor = nominal.time_measure.anchor
    t0, t1 = 2.0, 3.0
    ds, dp = anchor + timedelta(days=730), anchor + timedelta(days=1095)
    fwd = (nominal.df(ds) / nominal.df(dp)) * (real.df(dp) / real.df(ds)) * math.exp(
        model.convexity(0.0, t0, t1)
    )
    vols = []
    for k in SKEW_STRIKES:
        value = caplet_value(_floor(3, k), model, t0, t1, 1.0)
        vols.append(black_implied_vol(value / nominal.df(dp), fwd, 1.0 + k,
                                      t1 - t0, CapFloorKind.FLOOR))
    assert max(vols) - min(vols) < 1e-9, vols
    assert len(set(round(v, 12) for v in vols)) == 1, vols


# ============================================================================================
# ORACLE 4 -- the two failures are DISTINCT, and the thesis names only the second.
# ============================================================================================

def test_O4_the_underdetermination_holds_on_ATM_ONLY_data_before_any_skew():
    """**The half BERHAN does not report.** Its diagnosis is the strike skew; this fixture has ONE
    strike, so no skew exists — and three parameters are still unidentified."""
    quotes = _atm_quotes()
    assert len({q[0].strike for q in quotes}) == 1, "the fixture must be single-strike"
    fits = [_as_six(_fit(quotes, s).free) for s in STARTS]
    worst = max(
        max(f[n] for f in fits) - min(f[n] for f in fits) for n in UNIDENTIFIABLE
    )
    assert worst > 0.05, worst


def test_O4_adding_strikes_does_NOT_fix_the_underdetermination():
    """And it does not make it worse either: the extra rows are vega-scaled near-copies of the ATM
    ones, so they add magnitude and not information. Asserted because the spec's first draft
    claimed the opposite."""
    fits = [_as_six(_fit(_surface_quotes(), s).free) for s in STARTS]
    spread = {n: max(f[n] for f in fits) - min(f[n] for f in fits) for n in _as_six(TRUE_FREE)}
    for name in IDENTIFIABLE:
        assert spread[name] < 1e-3, (name, spread[name])
    assert max(spread[n] for n in UNIDENTIFIABLE) > 0.05, spread


def test_O4_the_fixture_surface_really_HAS_a_skew_to_fit():
    """The premise for the smile half: a single-strike surface could not show a smile failure at
    all (charter §4's degenerate-fixture clause)."""
    quotes = _surface_quotes()
    strikes = {q[0].strike for q in quotes}
    assert len(strikes) == len(SKEW_STRIKES) > 1
    by_strike = {k: [pv for f, pv in quotes if f.strike == k] for k in strikes}
    assert all(len(v) == len(MATURITIES) for v in by_strike.values())


# ============================================================================================
# ORACLE 5 -- failure is a value.
# ============================================================================================

def test_O5_no_quotes_is_a_FAILURE():
    out = calibrate_jy_to_floor_surface([], MARKET, FIXED, STARTS[0])
    assert isinstance(out, PricingFailure)
    assert "quote" in out.reason


def test_O5_an_out_of_range_correlation_start_is_a_FAILURE():
    out = calibrate_jy_to_floor_surface(_atm_quotes(), MARKET, FIXED,
                                        (0.04, 0.008, 0.012, 1.5, 0.0, 0.0))
    assert isinstance(out, PricingFailure)


def test_O5_a_non_positive_vol_start_is_a_FAILURE():
    out = calibrate_jy_to_floor_surface(_atm_quotes(), MARKET, FIXED,
                                        (0.04, 0.0, 0.012, 0.3, -0.2, 0.15))
    assert isinstance(out, PricingFailure)


def test_O5_an_unpriceable_quote_is_a_FAILURE_not_a_raise():
    bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({KEY_N: DiscountCurve.flat(VAL, 0.03, DC)}))
    out = calibrate_jy_to_floor_surface(_atm_quotes(), bare, FIXED, STARTS[0])
    assert isinstance(out, PricingFailure)


def test_O5_the_output_IS_a_usable_model_input():
    fit = _fit(_atm_quotes(), STARTS[0])
    model = JarrowYildirim(MARKET, jy_params_from_free(FIXED, fit.free))
    from pricebook.engine import price

    assert isinstance(price(_floor(4, ATM), model), PricingResult)
