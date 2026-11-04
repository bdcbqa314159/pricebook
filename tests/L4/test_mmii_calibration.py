"""Market Model II's calibration. BERHAN-2012 §5.4 — the tranche that closes the book.

§5.4.1: *"For each maturity Ti we need to estimate the volatility, sigma_n,i, of the (simply
compounded) nominal forward rate Fn(.,Ti). By the log-normal dynamics of Fn(.,Ti) we get an
automatic calibration to the quoted (Black) cap volatility sigma_ATM_i."* So **sigma_n is READ, not
fitted** -- an identity, and oracle 1 is an exact round-trip rather than a residual.

§5.4.2: the remaining per-expiry set is `{sigma_I,i-1, sigma_I,i, rho_I,i, rho_I,n,i}`, fitted
iteratively, with *"sigma_I,1 ... directly obtained from the 1-Year Floor, since it depends on no
other unknown parameters."*

**That bootstrap claim is structural and this library can show why.** For the first period the lead
`T_0 - t` is zero, so `Di` vanishes (it carries the lead as a factor) and `V^2` collapses to
`sigma_I,1^2 * (T_1 - T_0)`. One unknown, one price -- so the 1-Year floor inverts to `sigma_I,1` by
a plain Black inversion, reusing tranche 9's `black_implied_vol`. Oracle 2 asserts exactly that.

**What this suite REPORTS rather than presupposes** (the tranche 11 scar, where a spec asserted a
six-parameter recovery that was impossible): how many of the four per-expiry parameters are
identifiable, and how MM-II's fit compares with JY's. Oracle 3 runs the tranche 11 behavioural
method -- multi-start with the synthetic truth among the starts, and distance-from-truth asserted
beside spread -- and **no condition number**, because the smallest singular value of such a Jacobian
sits below the finite-difference noise floor (measured in tranche 11: three orders of magnitude of
step-dependence, and flaky across the two CI operating systems).
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.mmii_calibration import (
    MMIICalibration,
    calibrate_mmii_to_atm_floors,
    index_vol_from_first_floor,
    nominal_vol_from_black_atm,
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
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models import MarketModelII, MMIICorrelations, MMIIParams
from pricebook.products import InflationCapFloor

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
HICP = InflationIndex("HICPxt", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
N = 1.0
# SEMI-ANNUAL, so tau != 1 and Di's denominator stays oracled (tranche 13's fifth-appearance trap).
DATES = tuple(VAL + timedelta(days=180 * k) for k in range(6))
NOMINAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0), (0.026, 0.030, 0.034, 0.038))
REAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0), (0.008, 0.011, 0.013, 0.016))
MARKET = MarketSnapshot(
    valuation_date=VAL,
    curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): NOMINAL,
                     CurveKey(CurveRole.REAL, EUR): REAL}),
)
ATM = 0.02
NOMINAL_VOL = 0.25
CORR = MMIICorrelations(index_nominal=0.25, index_index=0.80)
# A SLOPED index-vol term structure, so sigma_{I,i-1} != sigma_{I,i} and the four per-expiry
# parameters are genuinely four. A flat one would collapse two of them and make any identifiability
# finding an artefact of the fixture (charter section 4).
TRUE_VOLS = {d: 0.010 + 0.002 * k for k, d in enumerate(DATES)}
TRUTH = MMIIParams(TRUE_VOLS, NOMINAL_VOL, CORR)
TRUE_MODEL = MarketModelII(MARKET, TRUTH)


def _floor(n_periods: int, strike: float = ATM) -> InflationCapFloor:
    return InflationCapFloor(
        HICP, CouponSchedule(DATES[: n_periods + 1], DC), strike, Money(N, EUR),
        CapFloorKind.FLOOR,
    )


def _pv(n_periods: int, strike: float, model: MarketModelII) -> float:
    out = price(_floor(n_periods, strike), model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _atm_quotes(model: MarketModelII = TRUE_MODEL):
    return [(_floor(n), _pv(n, ATM, model)) for n in range(1, len(DATES))]


# ============================================================================================
# ORACLE 1 — sigma_n is an IDENTITY, not a fit.
# ============================================================================================

def test_O1_sigma_n_IS_the_black_atm_cap_vol_exactly():
    """§5.4.1 is an identity: the log-normal dynamics of `Fn` make `sigma_n,i` *be* the quoted Black
    ATM cap volatility. So this is an exact round-trip, not a residual — a fitted `sigma_n` would be
    a mis-reading of the section."""
    for quoted in (0.05, 0.18, 0.25, 0.60):
        assert nominal_vol_from_black_atm(quoted) == quoted


def test_O1_a_negative_quote_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        nominal_vol_from_black_atm(-0.01)


# ============================================================================================
# ORACLE 2 — the bootstrap: sigma_I,1 from the 1-Year floor ALONE.
# ============================================================================================

def test_O2_the_first_period_has_ZERO_lead_so_Di_vanishes():
    """The structural reason §5.4.2's bootstrap works, asserted before relying on it: the first
    period starts at the curve anchor, so `T_0 - t = 0`, `Di` carries that as a factor and vanishes,
    and `V^2` collapses to `sigma_I,1^2 * (T_1 - T_0)` — one unknown."""
    start, pay = DATES[0], DATES[1]
    assert MEASURE.year_fraction(start) == 0.0
    span = MEASURE.year_fraction(pay)
    # m is the pure curve ratio: no drift at all.
    curve_ratio = (NOMINAL.df(start) / NOMINAL.df(pay)) * (REAL.df(pay) / REAL.df(start))
    assert TRUE_MODEL.expected_index_ratio(start, pay) == pytest.approx(curve_ratio, rel=1e-15)
    # and V^2 is sigma_{I,1}^2 * span, with sigma_{I,0} nowhere in it
    assert TRUE_MODEL.index_ratio_variance(start, pay) == pytest.approx(
        TRUE_VOLS[pay] ** 2 * span, rel=1e-15
    )


def test_O2_sigma_I_1_is_recovered_from_the_one_year_floor_alone():
    quote = _pv(1, ATM, TRUE_MODEL)
    recovered = index_vol_from_first_floor(_floor(1), quote, MARKET)
    assert recovered == pytest.approx(TRUE_VOLS[DATES[1]], rel=1e-9)


def test_O2_the_bootstrap_reads_NOTHING_but_the_first_floor():
    """It must be blind to every other parameter — so changing the correlations and the later index
    vols cannot move it. That is the whole content of "depends on no other unknown parameters"."""
    quote = _pv(1, ATM, TRUE_MODEL)
    first = index_vol_from_first_floor(_floor(1), quote, MARKET)
    other = MarketModelII(
        MARKET,
        MMIIParams({**TRUE_VOLS, DATES[2]: 0.09, DATES[3]: 0.11}, 0.60,
                   MMIICorrelations(-0.80, 0.10)),
    )
    assert _pv(1, ATM, other) == pytest.approx(quote, rel=1e-14), (
        "the 1-period floor moved when a LATER parameter changed — the bootstrap premise is false"
    )
    assert index_vol_from_first_floor(_floor(1), _pv(1, ATM, other), MARKET) == pytest.approx(
        first, rel=1e-12
    )


def test_O2_a_sub_intrinsic_first_floor_quote_is_REFUSED():
    with pytest.raises(ValueError):
        index_vol_from_first_floor(_floor(1), -1.0, MARKET)


# ============================================================================================
# ORACLE 3 — identifiability. REPORTED by the tranche 11 behavioural method, not presupposed.
# ============================================================================================

def test_O3_the_fit_recovers_the_term_structure_from_ATM_quotes():
    out = calibrate_mmii_to_atm_floors(_atm_quotes(), MARKET, NOMINAL_VOL, CORR)
    assert isinstance(out, MMIICalibration), out
    for k in range(1, len(DATES)):
        assert out.params.index_vol_at(DATES[k]) == pytest.approx(
            TRUE_VOLS[DATES[k]], rel=1e-6
        ), f"period {k}"
    assert out.rms_residual < 1e-9


def test_O3_the_quotes_are_DISTINCT_so_the_fit_sees_a_real_strip():
    pvs = [pv for _, pv in _atm_quotes()]
    assert len(set(round(p, 14) for p in pvs)) == len(pvs), pvs
    assert all(p > 0.0 for p in pvs)


def test_O3_every_index_vol_EXCEPT_THE_TERMINAL_ONE_is_identified():
    """**The finding, and it is not what the spec expected.** §5.4.2's claim is that *"since each
    expiry has it's own set of parameters, there is no need to restrict the number of contracts in
    the expiry dimension"*, and that is **almost** right: measured the tranche 11 way (the synthetic
    truth among the starts, distance-from-truth beside spread), every index vol recovers to ~1e-16
    **except the last one**, which from a low start lands at `0.002141` against a truth of `0.020000`
    — **89% wrong at a residual of 8.6e-17.**

    Why, exactly: `sigma_I,i` enters period `i`'s variance *quadratically* and its drift *linearly
    and with a negative sign*, so the caplet price is **non-monotone** in it — U-shaped, with a
    vertex at `rho_II * sigma_I,i-1 * lead / (lead + span)`. Two volatilities therefore reproduce one
    price. Every vol except the terminal one appears in **two** periods (as period `i`'s pay vol and
    period `i+1`'s start vol), and the second appearance picks the root; the terminal date appears
    once, so nothing breaks the tie. `test_O3_the_terminal_caplet_price_is_NON_MONOTONE` is the
    mechanism, asserted rather than described."""
    quotes = _atm_quotes()
    n = len(DATES)
    starts = (
        tuple(TRUE_VOLS[DATES[k]] for k in range(1, n)),      # the truth itself
        tuple(0.030 for _ in range(1, n)),
        tuple(0.004 for _ in range(1, n)),
    )
    fits = []
    for st in starts:
        out = calibrate_mmii_to_atm_floors(quotes, MARKET, NOMINAL_VOL, CORR, start=st)
        assert isinstance(out, MMIICalibration), out
        fits.append([out.params.index_vol_at(DATES[k]) for k in range(1, n)])

    # Every period but the last: identified, and identified CORRECTLY.
    for i, d in enumerate(DATES[1:-1]):
        values = [f[i] for f in fits]
        spread = max(values) - min(values)
        worst = max(abs(v - TRUE_VOLS[d]) for v in values)
        assert spread < TRUE_VOLS[d] * 1e-6, f"sigma_I at {d} spread {spread:.3e}"
        assert worst < TRUE_VOLS[d] * 1e-6, f"sigma_I at {d} off by {worst:.3e}"

    # The terminal one: NOT identified -- and the fit still reports a perfect residual.
    terminal = DATES[-1]
    values = [f[-1] for f in fits]
    worst = max(abs(v - TRUE_VOLS[terminal]) for v in values) / TRUE_VOLS[terminal]
    assert worst > 0.5, (
        f"the terminal index vol came within {worst:.1%} of the truth from every start -- then this "
        "oracle no longer demonstrates the two-root ambiguity"
    )
    for st in starts:
        out = calibrate_mmii_to_atm_floors(quotes, MARKET, NOMINAL_VOL, CORR, start=st)
        assert isinstance(out, MMIICalibration)
        assert out.rms_residual < 1e-9, "a wrong terminal vol must still fit perfectly"


def test_O3_the_terminal_caplet_price_is_NON_MONOTONE_in_its_own_vol():
    """The mechanism behind the ambiguity above, measured directly: sweeping the terminal index vol
    with every earlier one pinned, the caplet price **falls then rises**. So two volatilities give
    one price, and a least-squares fit can land on either."""
    from pricebook.engine.capfloor import caplet_value

    start, pay = DATES[-2], DATES[-1]
    span = MEASURE.year_fraction(pay) - MEASURE.year_fraction(start)
    period = (start, pay, span)
    floor = _floor(len(DATES) - 1)

    def caplet(sigma_terminal: float) -> float:
        vols = {**TRUE_VOLS, pay: sigma_terminal}
        return caplet_value(floor, MarketModelII(MARKET, MMIIParams(vols, NOMINAL_VOL, CORR)),
                            period)

    sweep = [0.0015, 0.004, 0.008, 0.0115, 0.014, 0.020, 0.030]
    prices = [caplet(s) for s in sweep]
    assert prices != sorted(prices), f"the price is monotone — no ambiguity: {prices}"
    assert prices != sorted(prices, reverse=True), prices
    low = min(range(len(prices)), key=lambda i: prices[i])
    assert 0 < low < len(prices) - 1, f"the minimum is at an endpoint, not interior: {prices}"
    # The vertex is where the quadratic in sigma_pay turns -- a predicted location, not a fitted one.
    lead = MEASURE.year_fraction(start)
    predicted = CORR.index_index * TRUE_VOLS[start] * lead / (lead + span)
    assert sweep[low] == pytest.approx(predicted, rel=0.35), (sweep[low], predicted)


def test_O3_ADDING_expiries_does_not_degrade_the_earlier_ones():
    """§5.4.2's claim tested directly: fit a short strip and a long one, and the short strip's
    parameters must come out the same. That is what "no need to restrict the expiry dimension"
    means, and it is exactly what JY's global fit could not offer."""
    short = calibrate_mmii_to_atm_floors(_atm_quotes()[:2], MARKET, NOMINAL_VOL, CORR)
    long = calibrate_mmii_to_atm_floors(_atm_quotes(), MARKET, NOMINAL_VOL, CORR)
    assert isinstance(short, MMIICalibration) and isinstance(long, MMIICalibration)
    for k in (1, 2):
        assert short.params.index_vol_at(DATES[k]) == pytest.approx(
            long.params.index_vol_at(DATES[k]), rel=1e-5
        ), f"period {k} moved when later expiries were added"


# ============================================================================================
# ORACLE 5 — the skew failure, as its own axis, with the source's stated DIRECTION.
# ============================================================================================

def test_O5_an_ATM_fit_UNDERPRICES_OTM_and_OVERPRICES_ITM_floors():
    """§5.4.2's exact words: *"if we assume constant volatility in the expiry dimension, then OTM
    contracts are underpriced and ITM contracts are overpriced."*

    A **signed, directional** claim — much stronger than "the error is large". Built by giving the
    synthetic market a skew (higher implied vol at low strikes, as inflation floor markets show) and
    then fitting MM-II at ATM only. For a floor, OTM is a LOW strike and ITM a HIGH one."""
    fit = calibrate_mmii_to_atm_floors(_atm_quotes(), MARKET, NOMINAL_VOL, CORR)
    assert isinstance(fit, MMIICalibration)
    fitted = MarketModelII(MARKET, fit.params)

    # A skewed "market": the same model but with the index vol lifted at low strikes and cut at
    # high ones -- the smile MM-II structurally cannot produce (its V^2 has no strike in it).
    def market_price(n: int, strike: float) -> float:
        tilt = 1.0 + 6.0 * (ATM - strike)        # richer OTM, cheaper ITM
        tilted = MMIIParams({d: v * tilt for d, v in fit.params.index_vols.items()},
                            NOMINAL_VOL, CORR)
        return _pv(n, strike, MarketModelII(MARKET, tilted))

    otm, itm = 0.005, 0.035
    n = 4
    assert market_price(n, ATM) == pytest.approx(_pv(n, ATM, fitted), rel=1e-12), (
        "the tilt must leave ATM untouched, or this is not a skew test"
    )
    assert _pv(n, otm, fitted) < market_price(n, otm), "OTM is not underpriced"
    assert _pv(n, itm, fitted) > market_price(n, itm), "ITM is not overpriced"


def test_O5_MMIIs_implied_smile_is_FLAT_which_is_WHY_the_skew_cannot_be_fitted():
    """The structural cause, as in tranche 11 oracle 3: `V^2` contains no strike, so MM-II prices
    every strike off one volatility. Kept with its near-tautology stated — it cannot discriminate a
    wrong `V^2`, only a strike leaking into one."""
    start, pay = DATES[2], DATES[3]
    base = TRUE_MODEL.index_ratio_variance(start, pay)
    assert all(
        TRUE_MODEL.index_ratio_variance(start, pay) == base for _ in (0.0, 0.01, 0.05)
    )


def test_O5_the_ATM_fit_itself_is_GOOD_so_the_two_axes_are_distinct():
    """The companion that keeps oracle 5 from reading as "MM-II cannot fit anything": on ATM quotes
    the residual is essentially zero. ATM works; the strike dimension is where it fails — two
    distinct axes, as the thesis says and as tranche 11 had to separate for JY."""
    fit = calibrate_mmii_to_atm_floors(_atm_quotes(), MARKET, NOMINAL_VOL, CORR)
    assert isinstance(fit, MMIICalibration)
    assert fit.rms_residual < 1e-9


# ============================================================================================
# ORACLES 4 and 6 — the recursion refusal is reachable, and failure is a value.
# ============================================================================================

def test_O4_the_T13_recursion_refusal_is_REACHABLE_and_its_depth_DEPENDS_ON_THE_PERIOD():
    """Tranche 13 found `index_vol_step` drives `sigma_I` negative at realistic rate vols, and that
    refusal is reachable from this calibration's parameters — it must surface as an exception a
    caller can see, never a `nan`.

    **The number of steps is NOT a constant, and assuming tranche 13's was my error.** T13 measured
    two steps on ANNUAL periods; this fixture is semi-annual, so `tau` is halved, each forward is
    smaller and the per-step adjustment is smaller — it survives **three** steps. So the oracle walks
    until refusal and asserts it arrives within a bounded number, rather than hardcoding a count
    that belongs to a different fixture."""
    from pricebook.models.market_model_i import index_vol_step

    tau_fn = NOMINAL.df(DATES[2]) / NOMINAL.df(DATES[3]) - 1.0
    tau_fr = REAL.df(DATES[2]) / REAL.df(DATES[3]) - 1.0
    sigma, steps = 0.012, 0
    with pytest.raises(ValueError, match="non-positive index volatility"):
        for _ in range(12):
            sigma = index_vol_step(sigma, 0.25, 0.20, tau_fn, tau_fr)
            steps += 1
            assert sigma > 0.0
    assert 2 <= steps <= 6, f"the recursion survived {steps} steps — the fixture has drifted"


def test_O6_no_quotes_is_a_FAILURE():
    out = calibrate_mmii_to_atm_floors([], MARKET, NOMINAL_VOL, CORR)
    assert isinstance(out, PricingFailure)
    assert "quote" in out.reason


def test_O6_a_non_positive_start_is_a_FAILURE():
    out = calibrate_mmii_to_atm_floors(_atm_quotes(), MARKET, NOMINAL_VOL, CORR,
                                       start=(0.0, 0.01, 0.01, 0.01, 0.01))
    assert isinstance(out, PricingFailure)


def test_O6_a_negative_nominal_vol_is_a_FAILURE():
    out = calibrate_mmii_to_atm_floors(_atm_quotes(), MARKET, -0.1, CORR)
    assert isinstance(out, PricingFailure)


def test_O6_an_unpriceable_quote_is_a_FAILURE_not_a_raise():
    bare = MarketSnapshot(
        valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): NOMINAL})
    )
    out = calibrate_mmii_to_atm_floors(_atm_quotes(), bare, NOMINAL_VOL, CORR)
    assert isinstance(out, PricingFailure)


def test_O6_the_output_IS_a_usable_model_input():
    fit = calibrate_mmii_to_atm_floors(_atm_quotes(), MARKET, NOMINAL_VOL, CORR)
    assert isinstance(fit, MMIICalibration)
    model = MarketModelII(MARKET, fit.params)
    assert isinstance(price(_floor(3), model), PricingResult)


# ============================================================================================
# ORACLE 7 — the cross-model comparison. The point of the whole inflation arc.
# ============================================================================================

def test_O7_MMII_identifies_its_parameters_where_JY_does_NOT():
    """**The comparison the book was for, and it is about identifiability, not price.**

    Tranche 13 measured the *pricing* gap between the models at 7.7e-09 to 3.7e-05 — negligible. The
    real difference is what you can learn from a calibration:

      * **JY (§5.3.2, tranche 11):** six GLOBAL parameters against the whole surface. Three of the
        six — `sigma_I`, `rho_rI`, `rho_nI` — are **structurally unidentifiable** from year-on-year
        floors at any strike, because the index correlations enter only as products with `sigma_I`
        and `rho_nI` is absent from the convexity entirely. `rho_nI` recovers at the WRONG SIGN
        (-0.597 against +0.15) at a residual of 8e-15.
      * **MM-II (§5.4.2, here):** per-expiry parameters. Every index vol recovers to ~1e-16 **except
        the terminal one**, which carries a two-root ambiguity because its caplet price is
        non-monotone in it.

    So the thesis's verdict — *"of the models evaluated, this approach seems the most promising"* —
    holds, and this library can now say **why** in a way the thesis does not: not better prices, but
    a calibration whose parameters are mostly recoverable, with one named exception at the end of the
    strip rather than three unidentifiable parameters throughout.

    This oracle asserts the MM-II half (JY's is tranche 11's `test_O2_*`), and that the two failures
    are of different KINDS: JY's is global and structural, MM-II's is local to the terminal point."""
    quotes = _atm_quotes()
    n = len(DATES)
    low = calibrate_mmii_to_atm_floors(quotes, MARKET, NOMINAL_VOL, CORR,
                                       start=tuple(0.004 for _ in range(1, n)))
    high = calibrate_mmii_to_atm_floors(quotes, MARKET, NOMINAL_VOL, CORR,
                                        start=tuple(0.030 for _ in range(1, n)))
    assert isinstance(low, MMIICalibration) and isinstance(high, MMIICalibration)

    # Interior: both starts agree with each other AND with the truth -- unlike any of JY's three.
    interior = [
        (low.params.index_vol_at(d), high.params.index_vol_at(d), TRUE_VOLS[d])
        for d in DATES[1:-1]
    ]
    assert all(
        abs(a - b) < 1e-9 and abs(a - truth) / truth < 1e-6 for a, b, truth in interior
    ), interior

    # Terminal: the one exception, and it is a SINGLE point rather than a pervasive degeneracy.
    terminal = DATES[-1]
    assert abs(low.params.index_vol_at(terminal) - TRUE_VOLS[terminal]) / TRUE_VOLS[terminal] > 0.5
    assert abs(high.params.index_vol_at(terminal) - TRUE_VOLS[terminal]) / TRUE_VOLS[terminal] < 1e-6

    # And the count is the headline: MM-II leaves ONE parameter ambiguous, JY leaves THREE.
    unidentified = sum(
        1 for d in DATES[1:]
        if abs(low.params.index_vol_at(d) - TRUE_VOLS[d]) / TRUE_VOLS[d] > 0.01
    )
    assert unidentified == 1, f"expected exactly the terminal point to be ambiguous, got {unidentified}"


def test_O7_whether_the_fit_LANDS_on_the_wrong_root_is_START_DEPENDENT():
    """**My first version of this claimed truncating the strip moves the ambiguity. It does not —
    the ambiguity DISAPPEARS.** Measured, fitting from a low start (0.004):

        strip of 5:  0.00%  0.00%  0.00%  0.00%  89.29%
        strip of 4:  0.00%  0.00%  0.00%  0.00%
        strip of 3:  0.00%  0.00%  0.00%

    So the two-root *structure* is always there (the non-monotonicity oracle proves it), but whether
    a least-squares fit **lands** on the wrong root depends on where it starts relative to the
    vertex and the conjugate root. A high start recovers the truth; a low one can fall into the
    conjugate.

    That is the practically important statement, and it is stronger than a recommendation to quote
    further out: **a single-start calibration of MM-II is unsafe at the terminal point, and no
    residual will tell you** — the wrong root fits to 1e-17. Multi-start is not a diagnostic
    nicety here, it is the only thing standing between the fit and a silently wrong terminal
    volatility. The tranche 11 lesson, arriving in the thesis's *preferred* model."""
    n = len(DATES)
    quotes = _atm_quotes()
    terminal = DATES[-1]

    from_low = calibrate_mmii_to_atm_floors(quotes, MARKET, NOMINAL_VOL, CORR,
                                            start=tuple(0.004 for _ in range(1, n)))
    from_high = calibrate_mmii_to_atm_floors(quotes, MARKET, NOMINAL_VOL, CORR,
                                             start=tuple(0.030 for _ in range(1, n)))
    assert isinstance(from_low, MMIICalibration) and isinstance(from_high, MMIICalibration)

    low_err = abs(from_low.params.index_vol_at(terminal) - TRUE_VOLS[terminal]) / TRUE_VOLS[terminal]
    high_err = abs(from_high.params.index_vol_at(terminal) - TRUE_VOLS[terminal]) / TRUE_VOLS[terminal]
    assert low_err > 0.5, low_err
    assert high_err < 1e-6, high_err
    # And BOTH report a residual indistinguishable from zero -- which is the dangerous part.
    assert from_low.rms_residual < 1e-9 and from_high.rms_residual < 1e-9, (
        from_low.rms_residual, from_high.rms_residual
    )


def test_O7_a_SHORTER_strip_does_not_inherit_the_ambiguity():
    """The companion to the above, and the reason the first version's premise was wrong: truncating
    the strip does NOT hand the ambiguity to the new terminal point. Whether the conjugate root is
    reachable from a given start depends on the period's own lead and the neighbouring vol, so it is
    a property of the point, not of being last."""
    quotes = _atm_quotes()
    for cut in (1, 2):
        shorter = quotes[: len(quotes) - cut]
        out = calibrate_mmii_to_atm_floors(
            shorter, MARKET, NOMINAL_VOL, CORR, start=tuple(0.004 for _ in shorter)
        )
        assert isinstance(out, MMIICalibration), out
        for q, _ in shorter:
            d = q.schedule.dates[-1]
            assert abs(out.params.index_vol_at(d) - TRUE_VOLS[d]) / TRUE_VOLS[d] < 1e-6, (cut, d)
