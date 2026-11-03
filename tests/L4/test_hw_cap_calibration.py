"""Fitting `(kappa_n, sigma_n)` to ATM nominal cap vols. BERHAN-2012 §5.3.1, MCINERNEY-ZASTAWNIAK 3.26.

BERHAN's scheme, verbatim: "for each maturity Ti, we observe the ATM Cap (Black) volatility quote
sigma_ATM_i and the accompanying ATM strike level K_ATM_i. We can then fit the nominal volatility
parameters kappa_n, sigma_n by performing a least squares optimization over

    Cap_HullWhite(t, Ti, psi_i, K_ATM_i, sigma_n, kappa_n) - Cap_Black(t, Ti, psi_i, K_ATM_i, sigma_ATM_i)"

**This closes the two-of-six gap.** `sigma_n` has been a hand-supplied input since tranche 6, because
BERHAN sources it here and declines to publish the forward model ("beyond the scope of this
thesis"). McInerney 3.5 supplied the model in (a); 3.26 confirms the scheme.

**The Black caplet's variance horizon is the EXPIRY, not the period -- the opposite of the
year-on-year case, and that is why the parameter is named `variance_years`.** A rate caplet's
underlying `L(T_{i-1},T_i)` is observed at `T_{i-1}`, so its variance is `sigma^2 * T_{i-1}`. An
inflation floorlet on `I(T_i)/I(T_{i-1})` accumulates over the period, so its variance is
`sigma_I^2 * (T_i - T_{i-1})`. Tranche 9 measured that confusion at up to 50%; here the two
instruments genuinely differ, and `test_the_black_caplet_variance_uses_the_EXPIRY...` pins it.

Identifiability is a real question, not a formality: `kappa_n` shapes the caplet-vol TERM STRUCTURE
while `sigma_n` sets its LEVEL, so across a maturity strip they should separate -- but tranche 6
showed a two-parameter fit can report a perfect residual at a wrong answer, so B2 proves it.
"""

import math

import pytest

from pricebook.engine.hw_cap_calibration import (
    AtmCapVolQuote,
    HullWhiteCalibration,
    black_cap_implied_vol,
    black_cap_price,
    calibrate_hull_white_to_atm_caps,
)
from pricebook.engine.nominal_cap import price_nominal_cap
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
from pricebook.market import (
    CurveKey,
    CurveRole,
    CurveSet,
    DiscountCurve,
    MarketSnapshot,
    swap_annuity,
)
from pricebook.models.hull_white import HullWhiteModel, HullWhiteParams
from pricebook.products import NominalCap

from datetime import date, timedelta

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
KEY = CurveKey(CurveRole.DISCOUNT, EUR)
N = 1.0   # the fit is notional-invariant; unit notional keeps the residuals readable

# SLOPED, and SEMI-ANNUAL so no accrual sits at 1.0 (charter §4's degenerate-fixture clause).
CURVE = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0, 10.0),
                      (0.024, 0.028, 0.032, 0.036, 0.039))
MARKET = MarketSnapshot(valuation_date=VAL, curves=CurveSet({KEY: CURVE}))

KAPPA_TRUE, SIGMA_TRUE = 0.08, 0.0090
TRUTH = (KAPPA_TRUE, SIGMA_TRUE)
# A MATURITY STRIP is what separates the two parameters. One maturity cannot: kappa only shows
# across the term structure, so a single-maturity strip is the degenerate fixture B2 must reject.
TENORS = (2, 4, 8, 12, 20)   # in semi-annual periods: 1y, 2y, 4y, 6y, 10y


def _schedule(n_periods: int) -> CouponSchedule:
    return CouponSchedule(
        tuple(VAL + timedelta(days=180 * k) for k in range(n_periods + 1)), DC
    )


def _atm_strike(n_periods: int) -> float:
    """The par swap rate of the same tenor — the conventional ATM cap strike.

    Composes tranche 8's `swap_annuity`, which is its second consumer: `(1 − df(Tn)) / Σ τ·df`."""
    schedule = _schedule(n_periods)
    return (1.0 - CURVE.df(schedule.dates[-1])) / swap_annuity(CURVE, schedule)


def _cap(n_periods: int, strike: float | None = None) -> NominalCap:
    k = _atm_strike(n_periods) if strike is None else strike
    return NominalCap(_schedule(n_periods), k, Money(N, EUR), CapFloorKind.CAP)


def _hw_price(n_periods: int, params: HullWhiteParams) -> float:
    out = price_nominal_cap(_cap(n_periods), HullWhiteModel(MARKET, params))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _quotes(params: HullWhiteParams = HullWhiteParams(*TRUTH)) -> list[AtmCapVolQuote]:
    """Quotes generated from a known `(kappa, sigma)`: price each cap under Hull-White, then imply
    the flat Black vol that reproduces it. At the generating parameters the residual is exactly 0."""
    out = []
    for n in TENORS:
        schedule = _schedule(n)
        strike = _atm_strike(n)
        vol = black_cap_implied_vol(_hw_price(n, params), schedule, strike, CURVE)
        out.append(AtmCapVolQuote(schedule, strike, vol))
    return out


def _fit(quotes=None, start=(0.03, 0.02)) -> HullWhiteCalibration:
    out = calibrate_hull_white_to_atm_caps(quotes or _quotes(), MARKET, start)
    assert isinstance(out, HullWhiteCalibration), out
    return out


# ============================================================================================
# The Black cap side -- the quoting convention the fit measures against.
# ============================================================================================

def test_the_black_cap_is_the_SUM_of_its_black_caplets():
    quote = _quotes()[2]
    expected = 0.0
    from pricebook.models.black import black_price

    for start, pay, accrual in quote.schedule.periods():
        expiry = MEASURE.year_fraction(start)
        forward = (CURVE.df(start) / CURVE.df(pay) - 1.0) / accrual
        expected += N * accrual * CURVE.df(pay) * black_price(
            forward, quote.strike, quote.black_vol**2 * expiry, CapFloorKind.CAP
        )
    assert black_cap_price(quote, CURVE) == pytest.approx(expected, rel=1e-14)


def test_the_black_caplet_variance_uses_the_EXPIRY_not_the_period():
    """**The opposite of the inflation case, and the reason `variance_years` is named that.** A rate
    caplet's underlying fixes at `T_{i-1}`, so its variance is `sigma^2 * T_{i-1}`; using the period
    length `T_i - T_{i-1}` instead would be the tranche 9 defect with the sign of the error
    reversed. Pinned by showing the two give different prices on a multi-period cap."""
    from pricebook.models.black import black_price

    quote = _quotes()[3]
    with_expiry = with_period = 0.0
    for start, pay, accrual in quote.schedule.periods():
        t0, t1 = MEASURE.year_fraction(start), MEASURE.year_fraction(pay)
        forward = (CURVE.df(start) / CURVE.df(pay) - 1.0) / accrual
        disc = N * accrual * CURVE.df(pay)
        with_expiry += disc * black_price(
            forward, quote.strike, quote.black_vol**2 * t0, CapFloorKind.CAP
        )
        with_period += disc * black_price(
            forward, quote.strike, quote.black_vol**2 * (t1 - t0), CapFloorKind.CAP
        )
    assert black_cap_price(quote, CURVE) == pytest.approx(with_expiry, rel=1e-14)
    assert abs(with_period / with_expiry - 1.0) > 0.1, "the two horizons barely differ — no power"


def test_the_black_cap_vol_round_trips_through_its_implied_vol():
    quote = _quotes()[1]
    price = black_cap_price(quote, CURVE)
    assert black_cap_implied_vol(price, quote.schedule, quote.strike, CURVE) == pytest.approx(
        quote.black_vol, rel=1e-9
    )


def test_the_black_cap_is_MONOTONE_in_its_vol():
    schedule, strike = _schedule(8), _atm_strike(8)
    prices = [black_cap_price(AtmCapVolQuote(schedule, strike, v), CURVE)
              for v in (0.05, 0.10, 0.20, 0.50)]
    assert prices == sorted(prices), prices
    assert len(set(round(p, 14) for p in prices)) == 4


# ============================================================================================
# B1 -- parameter recovery. Vacuous without B2.
# ============================================================================================

def test_B1_recovers_the_generating_parameters():
    fit = _fit(start=(0.03, 0.02))
    assert fit.params.kappa == pytest.approx(KAPPA_TRUE, rel=1e-6)
    assert fit.params.sigma == pytest.approx(SIGMA_TRUE, rel=1e-6)
    assert fit.n_quotes == len(TENORS)
    assert fit.rms_residual < 1e-10


def test_B1_the_quotes_are_a_real_SURFACE_not_a_flat_vol():
    """The premise: a Hull-White model implies a non-flat cap-vol term structure, which is what
    `kappa` is identified from. If these came out equal, `kappa` would be invisible."""
    vols = [q.black_vol for q in _quotes()]
    assert len(set(round(v, 8) for v in vols)) == len(vols), vols
    assert max(vols) / min(vols) > 1.05, f"the term structure is nearly flat: {vols}"


# ============================================================================================
# B2 -- identifiability. The T6 pattern: not flat AND not degenerate.
# ============================================================================================

def test_B2_each_parameter_MOVES_the_strip():
    """Necessary, not sufficient — the tranche 6 lesson. Proves the objective is not flat."""
    base = [_hw_price(n, HullWhiteParams(*TRUTH)) for n in TENORS]
    for label, params in {
        "kappa +20%": HullWhiteParams(KAPPA_TRUE * 1.2, SIGMA_TRUE),
        "sigma +20%": HullWhiteParams(KAPPA_TRUE, SIGMA_TRUE * 1.2),
    }.items():
        moved = [_hw_price(n, params) for n in TENORS]
        worst = max(abs(a - b) / a for a, b in zip(base, moved))
        assert worst > 1e-3, f"{label} barely moves the strip ({worst:.2e})"


def test_B2_multi_start_recovery_lands_on_ONE_point():
    """Uniqueness, threshold-free. A valley rather than a basin would spread these out."""
    quotes = _quotes()
    for start in [(0.01, 0.004), (0.03, 0.02), (0.30, 0.002), (0.005, 0.05)]:
        fit = _fit(quotes, start)
        assert fit.params.kappa == pytest.approx(KAPPA_TRUE, rel=1e-5), f"from {start}"
        assert fit.params.sigma == pytest.approx(SIGMA_TRUE, rel=1e-5), f"from {start}"


def test_B2_the_sensitivity_jacobian_is_WELL_CONDITIONED():
    """The mechanism: the two parameters' sensitivity columns must be INDEPENDENT, not merely
    non-zero. `kappa` bends the term structure, `sigma` lifts it — so across a maturity strip the
    columns should separate. "Should" is what the oracle proves."""
    h_k, h_s = 1e-7, 1e-9
    base = [_hw_price(n, HullWhiteParams(*TRUTH)) for n in TENORS]
    col_k = [(_hw_price(n, HullWhiteParams(KAPPA_TRUE + h_k, SIGMA_TRUE)) - b) / h_k
             for n, b in zip(TENORS, base)]
    col_s = [(_hw_price(n, HullWhiteParams(KAPPA_TRUE, SIGMA_TRUE + h_s)) - b) / h_s
             for n, b in zip(TENORS, base)]
    gkk = sum(x * x for x in col_k)
    gss = sum(x * x for x in col_s)
    gks = sum(x * y for x, y in zip(col_k, col_s))
    trace, det = gkk + gss, gkk * gss - gks * gks
    disc = math.sqrt(max(trace * trace / 4.0 - det, 0.0))
    lo, hi = trace / 2.0 - disc, trace / 2.0 + disc
    assert lo > 0.0, "the Gram matrix is singular — the columns are collinear"
    assert hi / lo < 1e8, f"ill-conditioned: kappa={hi / lo:.3e}"


def test_B2_a_SINGLE_maturity_strip_is_NOT_identifiable():
    """The degenerate fixture, asserted as unusable rather than merely avoided — charter §4. One
    maturity is one equation in two unknowns: `kappa` only shows across the term structure, so
    per-parameter sensitivity still passes while the pair is unidentified."""
    global TENORS
    original = TENORS
    try:
        TENORS = (8,)
        quotes = _quotes()
        fits = [_fit(quotes, s) for s in [(0.01, 0.004), (0.30, 0.002), (0.05, 0.03)]]
        recovered = {(round(f.params.kappa, 5), round(f.params.sigma, 5)) for f in fits}
        assert len(recovered) > 1, (
            f"a one-maturity strip recovered one point from every start ({recovered}) — then this "
            "test has no power and B2's multi-start check proves less than it appears to"
        )
    finally:
        TENORS = original


# ============================================================================================
# B3 -- the objective vanishes at the truth and descends from a wrong start.
# ============================================================================================

def test_B3_the_residual_vanishes_at_the_generating_parameters():
    quotes = _quotes()
    worst = max(
        abs(_hw_price(n, HullWhiteParams(*TRUTH)) - black_cap_price(q, CURVE))
        for n, q in zip(TENORS, quotes)
    )
    assert worst < 1e-12, worst


def test_B3_a_deliberately_wrong_start_strictly_improves():
    quotes = _quotes()
    start = (0.30, 0.002)
    before = math.sqrt(
        sum((_hw_price(n, HullWhiteParams(*start)) - black_cap_price(q, CURVE)) ** 2
            for n, q in zip(TENORS, quotes)) / len(quotes)
    )
    fit = _fit(quotes, start)
    assert before > 1e-6, "the wrong start is not actually wrong — the test proves nothing"
    assert fit.rms_residual < before / 1e4


# ============================================================================================
# B4 -- failure is a value.
# ============================================================================================

def test_B4_no_quotes_is_a_FAILURE():
    out = calibrate_hull_white_to_atm_caps([], MARKET, (0.05, 0.01))
    assert isinstance(out, PricingFailure)
    assert "quote" in out.reason


def test_B4_a_non_positive_start_is_a_FAILURE():
    out = calibrate_hull_white_to_atm_caps(_quotes(), MARKET, (0.0, 0.01))
    assert isinstance(out, PricingFailure)


def test_B4_an_unpriceable_quote_is_a_FAILURE_not_a_raise():
    bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({}))
    out = calibrate_hull_white_to_atm_caps(_quotes(), bare, (0.05, 0.01))
    assert isinstance(out, PricingFailure)


def test_B4_the_calibration_output_IS_a_usable_model_input():
    """`params` is a complete `HullWhiteParams`, so a fit feeds `HullWhiteModel` directly with no
    unpacking step in which a caller could pair a fitted sigma with the wrong kappa."""
    fit = _fit()
    model = HullWhiteModel(MARKET, fit.params)
    out = price_nominal_cap(_cap(8), model)
    assert isinstance(out, PricingResult), out
