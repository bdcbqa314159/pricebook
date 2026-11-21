"""The Normal/Bachelier and displaced-diffusion caplet pricers. MCINERNEY-ZASTAWNIAK §8.2, §8.4.

Both models are a REPARAMETERISATION of the Black caplet, and the chapter's point is the smile each
one implies. So the oracles come in three kinds, and they are deliberately not interchangeable:

**The pointwise transcriptions** (p.147 for Bachelier, p.151-152 for displaced-diffusion), read in
`-raw` and `-layout` and reconciled. These are the independent correctness anchors. The Bachelier
one is transcribed in the source's own printed shape -- `(s/sqrt(2*pi))*exp(-d^2/2)` -- and NOT as
`s*phi(d)`, because `norm_pdf` is what the implementation calls: restating the implementation's own
composition is the predecessor's 5.14% bug (§4).

**The chapter's own printed numbers** (Example 8.1, p.152): Black gives **0.003789** at `F = K = 5%`,
`sigma = 0.2`, `T = 1`, discounting at 5% continuous; the displaced-diffusion formula with
`alpha = 0.05` and eq (8.9)'s matched `sigma_D = 0.1` gives **0.003793**. A trusted mark, the lowest
rung of §4's ladder -- but it is the only *external* anchor the displaced-diffusion piece has, and it
is the one that matters, for the reason in the next paragraph.

**What is NOT an oracle here, stated so it is not mistaken for one.** Two ties in this file are
tautologies rather than checks, and tranche 70's lesson is that a reduction's strength is an
argument about types:

  - The displaced-diffusion reduction at `displacement = 0` is **bit-identical** to `black_price`,
    because `F + 0.0 == F` exactly -- the composition never passes through a `log`/`exp`, so there is
    no tranche-61 ULP to band. It proves `black_price(F,K,v) == black_price(F,K,v)` and nothing more.
    Banding it would have been worse than useless: a tolerance makes a tautology LOOK like a
    measurement. The displaced-diffusion correctness therefore rests on the pointwise `d+/d-`
    transcription and on Example 8.1's printed 0.003793.
  - The Bachelier caplet and tranche 25's `expected_positive_exposure` are now the **same function**
    (§6's rule of two fired at this tranche; the body is L0's `normal_positive_part`). Their
    agreement is true by construction. It is asserted anyway -- as a guard that the delegation is
    not quietly re-implemented on one side -- and labelled.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.smile_caplet import bachelier_caplet, displaced_caplet
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    Money,
    PricingResult,
)
from pricebook.foundation.distributions import norm_cdf
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.black import black_implied_vol, black_price
from pricebook.models.hull_white import CapletPeriod
from pricebook.models.smile import (
    BachelierModel,
    DisplacedDiffusionModel,
    atm_matched_displaced_vol,
)
from pricebook.products import NominalCap
from pricebook.risk.credit_valuation_adjustment import expected_positive_exposure

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
N = 10_000_000.0
KEY = CurveKey(CurveRole.DISCOUNT, EUR)

# SEMI-ANNUAL periods, so no accrual is 1.0 (§4's first degenerate shape), and the schedule
# STRADDLES the valuation date: the k=-1 period pays ON `VAL`, which invariant 6 makes historical.
# That straddle is §4's TENTH degenerate shape, which has now bitten three times -- a fixture
# beginning at the valuation date leaves the historical-flow filter with nothing to filter. The
# premise is asserted, not believed, in `test_the_STRADDLE_premise...` below.
DATES = tuple(VAL + timedelta(days=180 * k) for k in range(-1, 5))
# SLOPED, so the caplets are not all equal and a mis-indexed strip would show. A curve's first
# pillar must be `t=0` (`market/curve.py` says so), and it extrapolates flat behind that -- so the
# historical period's discount factors are still computable (both exceed 1), which is what makes
# "dropping the filter changes the answer" a measurable premise rather than a hope.
CURVE = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 4.0), (0.026, 0.030, 0.034, 0.038))

NORMAL_VOL = 0.0085          # a RATE per root-year: ~85bp, the scale a normal model is quoted at
LOGNORMAL_VOL = 0.22
SHIFT = 0.015


def _snapshot(vd: date = VAL, curve: DiscountCurve = CURVE) -> MarketSnapshot:
    return MarketSnapshot(valuation_date=vd, curves=CurveSet({KEY: curve}))


def _cap(strike: float = 0.03, kind: CapFloorKind = CapFloorKind.CAP) -> NominalCap:
    return NominalCap(CouponSchedule(DATES, DC), strike, Money(N, EUR), kind)


def _pv(product: NominalCap, model: object) -> float:
    out = price(product, model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _period(forward: float, df_pay: float, accrual: float = 1.0) -> CapletPeriod:
    """A period whose SIMPLE FORWARD is exactly `forward` — the inverse of `forward_rate()`.

    Example 8.1 posits `F(0) = 5%` and a 5% flat discount independently; it does not derive the
    forward from the curve (the simple forward of a flat 5% continuous curve is 5.127%, not 5%). So
    the printed marks are oracled on the closed form at its own arguments, which is where §4 wants
    a formula oracled anyway."""
    return CapletPeriod(
        df_start=df_pay * (1.0 + accrual * forward), df_pay=df_pay, accrual=accrual
    )


# The helper's round trip is not bit-exact: `(df_pay*(1+tau*F)/df_pay - 1)/tau` recovers `F` to
# about 1e-14 relative, not to the bit. Identities that take the forward as an INPUT (parity, the
# zero-shift tautology) therefore read it back with `forward_rate()` so both sides price the same
# number -- that is reading an input, not reading the answer back, which is what §4's independence
# rule forbids. The pointwise transcriptions above assert a VALUE and band it at `rel=1e-14`, which
# covers the same round trip without needing the code to supply anything.


# ============================================================================================
# 1. The POINTWISE transcriptions — the independent correctness anchors.
# ============================================================================================

# `(F - K)/s` spans deep out of the money, at the money and deep in. A single at-the-money strike
# would leave `N(d1)` pinned at 0.5 and the two terms indistinguishable in weight.
BACHELIER_CASES = (
    (0.051, 0.030, 0.0120),
    (0.031, 0.028, 0.0085),
    (0.030, 0.030, 0.0085),
    (0.028, 0.031, 0.0085),
    (0.020, 0.055, 0.0040),
)


@pytest.mark.parametrize("forward,strike,stdev", BACHELIER_CASES)
def test_the_BACHELIER_caplet_matches_the_p147_transcription(forward, strike, stdev):
    # Cpl^Normal_i(t) = tau_i * B(t,T_i) * [ (F_i(t) - K) N(d1) + (s_i/sqrt(2 pi)) exp(-d1^2/2) ]
    #            d1   = (F_i(t) - K) / s_i ,   s_i^2 = int_t^{T_{i-1}} sigma_i^2(s) ds
    # Reconciled across `-raw` and `-layout` of pdf p.157 (book p.147). The second term is written
    # in the source's own shape, not as `s*phi(d1)`, so this does not restate `norm_pdf`.
    accrual, df_pay = 0.4932, 0.94
    d1 = (forward - strike) / stdev
    want = accrual * df_pay * (
        (forward - strike) * norm_cdf(d1)
        + (stdev / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)
    )
    got = bachelier_caplet(
        _period(forward, df_pay, accrual), strike, stdev * stdev, CapFloorKind.CAP
    )
    assert got == pytest.approx(want, rel=1e-14), (got, want)


@pytest.mark.parametrize("forward,strike,stdev", BACHELIER_CASES)
def test_the_two_TERMS_are_both_material_at_every_case(forward, strike, stdev):
    # The non-degeneracy premise for the transcription above: if the `exp(-d1^2/2)` term were
    # negligible anywhere, dropping it would survive there. §4's named survivor shape.
    #
    # **The threshold is 4% because that is what was MEASURED, and materiality is judged against
    # the oracle's band rather than a round number.** The term's share of the price across this
    # fixture runs 4.88% (`ratio = +1.75`, the in-the-money point, where `N(d1)` is already 0.96 and
    # the first term carries nearly everything) through 100% at the money to 7949% deep out of it,
    # where the two terms cancel and the second is all that is left. The transcription above bands
    # at `rel=1e-14`, so even the weakest point sits twelve orders of magnitude above what a green
    # there can resolve: the term is material at every point in the fixture, and 4% is the measured
    # floor stated rather than a goalpost moved to fit.
    accrual, df_pay = 0.4932, 0.94
    unit = _period(forward, df_pay, accrual)
    recovered = unit.forward_rate()
    d1 = (recovered - strike) / stdev
    whole = bachelier_caplet(unit, strike, stdev * stdev, CapFloorKind.CAP)
    first = accrual * df_pay * (recovered - strike) * norm_cdf(d1)
    assert abs(whole - first) > 0.04 * abs(whole), (forward, strike, whole, first)


DISPLACED_CASES = ((0.051, 0.030), (0.031, 0.028), (0.030, 0.030), (0.020, 0.055))


@pytest.mark.parametrize("forward,strike", DISPLACED_CASES)
def test_the_DISPLACED_caplet_matches_the_p151_transcription(forward, strike):
    #  Cpl_i(t) = tau_i B(t,T_i) ( (F_i(t)+alpha) N(d+) - (K+alpha) N(d-) )
    #  d+ = [ ln((F_i(t)+alpha)/(K+alpha)) + (1/2) int sigma^D^2 ] / sqrt( int sigma^D^2 )
    # Reconciled across `-raw` and `-layout` of pdf pp.161-162 (book pp.151-152). `d-` is `d+`
    # less the root variance, which the source prints on p.152.
    accrual, df_pay, years = 0.4932, 0.94, 1.7
    variance = LOGNORMAL_VOL * LOGNORMAL_VOL * years
    root = math.sqrt(variance)
    shifted_f, shifted_k = forward + SHIFT, strike + SHIFT
    d_plus = (math.log(shifted_f / shifted_k) + 0.5 * variance) / root
    d_minus = d_plus - root
    want = accrual * df_pay * (shifted_f * norm_cdf(d_plus) - shifted_k * norm_cdf(d_minus))
    got = displaced_caplet(
        _period(forward, df_pay, accrual), strike, variance, SHIFT, CapFloorKind.CAP
    )
    assert got == pytest.approx(want, rel=1e-14), (got, want)


# ============================================================================================
# 2. The chapter's own PRINTED numbers — Example 8.1, p.152.
# ============================================================================================

def test_example_8_1_reproduces_the_printed_BLACK_price():
    # "the caplet price at time 0 is given by Black's formula (5.8) as 0.003789"
    forward = strike = 0.05
    df_pay = math.exp(-0.05 * 1.0)
    got = df_pay * black_price(forward, strike, 0.2 * 0.2 * 1.0, CapFloorKind.CAP)
    assert round(got, 6) == 0.003789, got


def test_example_8_1_reproduces_the_printed_DISPLACED_DIFFUSION_price():
    # "if alpha = 0.05, then sigma_D(t) = 0.1, producing a caplet price of 0.003793"
    forward = strike = 0.05
    df_pay = math.exp(-0.05 * 1.0)
    matched = atm_matched_displaced_vol(0.2, forward, 0.05)
    got = displaced_caplet(
        _period(forward, df_pay), strike, matched * matched * 1.0, 0.05, CapFloorKind.CAP
    )
    assert round(got, 6) == 0.003793, got


def test_the_printed_pair_DIFFER_so_the_displacement_is_doing_something():
    # The two marks round to different numbers at the printed precision, which is the premise that
    # makes the second one an oracle on the shift rather than a second reading of Black.
    assert 0.003793 != 0.003789


# ============================================================================================
# 3. Requirement oracles — true for ALL parameters, so they survive a transcription error.
# ============================================================================================

@pytest.mark.parametrize("forward,strike,stdev", BACHELIER_CASES)
def test_BACHELIER_put_call_parity(forward, strike, stdev):
    accrual, df_pay = 0.4932, 0.94
    unit = _period(forward, df_pay, accrual)
    cap = bachelier_caplet(unit, strike, stdev * stdev, CapFloorKind.CAP)
    floor = bachelier_caplet(unit, strike, stdev * stdev, CapFloorKind.FLOOR)
    want = accrual * df_pay * (unit.forward_rate() - strike)
    assert cap - floor == pytest.approx(want, abs=1e-18)


@pytest.mark.parametrize("forward,strike", DISPLACED_CASES)
def test_DISPLACED_put_call_parity(forward, strike):
    # This is the oracle that catches shifting `F` and forgetting `K` — the classic
    # displaced-diffusion bug, which leaves parity off by exactly the shift.
    accrual, df_pay, variance = 0.4932, 0.94, 0.0484 * 1.7
    unit = _period(forward, df_pay, accrual)
    cap = displaced_caplet(unit, strike, variance, SHIFT, CapFloorKind.CAP)
    floor = displaced_caplet(unit, strike, variance, SHIFT, CapFloorKind.FLOOR)
    want = accrual * df_pay * (unit.forward_rate() - strike)
    assert cap - floor == pytest.approx(want, rel=1e-13)


@pytest.mark.parametrize("forward,strike", DISPLACED_CASES)
def test_both_models_collapse_to_the_INTRINSIC_value_as_the_variance_vanishes(forward, strike):
    accrual, df_pay = 0.4932, 0.94
    unit = _period(forward, df_pay, accrual)
    intrinsic = accrual * df_pay * max(forward - strike, 0.0)
    assert bachelier_caplet(unit, strike, 1e-24, CapFloorKind.CAP) == pytest.approx(
        intrinsic, abs=1e-12
    )
    assert displaced_caplet(unit, strike, 1e-24, SHIFT, CapFloorKind.CAP) == pytest.approx(
        intrinsic, abs=1e-12
    )


def test_both_prices_RISE_with_their_own_volatility():
    unit = _period(0.031, 0.94, 0.4932)
    bachelier = [bachelier_caplet(unit, 0.030, v * v, CapFloorKind.CAP)
                 for v in (0.004, 0.0085, 0.015, 0.030)]
    displaced = [displaced_caplet(unit, 0.030, v * v, SHIFT, CapFloorKind.CAP)
                 for v in (0.10, 0.22, 0.40, 0.70)]
    assert bachelier == sorted(bachelier), bachelier
    assert displaced == sorted(displaced), displaced
    assert bachelier[-1] > 2.0 * bachelier[0]        # materially, not on rounding noise
    assert displaced[-1] > 2.0 * displaced[0]


# ============================================================================================
# 4. The two TAUTOLOGIES, labelled — see the module docstring.
# ============================================================================================

@pytest.mark.parametrize("forward,strike", DISPLACED_CASES)
def test_a_ZERO_displacement_is_BIT_IDENTICAL_to_black_which_is_a_tautology(forward, strike):
    # `F + 0.0 == F` exactly, so this is `black_price(F,K,v) == black_price(F,K,v)`. Asserted as a
    # tautology with `==` and NOT banded: a tolerance here would dress a tautology as a measurement.
    # The shift's correctness lives in the p.151 transcription and Example 8.1, not here.
    accrual, df_pay, variance = 0.4932, 0.94, 0.0484 * 1.7
    unit = _period(forward, df_pay, accrual)
    got = displaced_caplet(unit, strike, variance, 0.0, CapFloorKind.CAP)
    want = black_price(unit.forward_rate(), strike, variance, CapFloorKind.CAP)
    assert got == accrual * df_pay * want


@pytest.mark.parametrize("forward,strike,stdev", BACHELIER_CASES)
def test_the_bachelier_call_IS_tranche_25s_exposure_by_CONSTRUCTION(forward, strike, stdev):
    # Not independence — one function, two names (§6's rule of two fired here). This guards the
    # DELEGATION: it fails the day someone re-implements either side.
    accrual, df_pay = 0.4932, 0.94
    unit = _period(forward, df_pay, accrual)
    got = bachelier_caplet(unit, strike, stdev * stdev, CapFloorKind.CAP)
    want = expected_positive_exposure(unit.forward_rate() - strike, stdev)
    assert got == accrual * df_pay * want


# ============================================================================================
# 5. The SMILE — measured, never pre-asserted (§4's obligation-not-form clause).
# ============================================================================================

SMILE_STRIKES = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)


def _implied(price_: float, forward: float, strike: float, years: float, df: float) -> float:
    return black_implied_vol(price_ / df, forward, strike, years, CapFloorKind.CAP)


def test_the_DISPLACED_DIFFUSION_skew_is_MEASURED_and_is_monotonically_DECREASING():
    # The source's own claim, p.152: "the displaced-diffusion model produces a monotonically
    # decreasing curve". Measured by inverting the model's own prices through Black, not asserted
    # from the algebra — and reported, so the shape is on the record.
    forward, years, df = 0.05, 1.0, math.exp(-0.05)
    matched = atm_matched_displaced_vol(0.2, forward, 0.05)
    unit = _period(forward, df)
    vols = []
    for strike in SMILE_STRIKES:
        px = displaced_caplet(unit, strike, matched * matched * years, 0.05, CapFloorKind.CAP)
        vols.append(_implied(px, forward, strike, years, df))
    assert vols == sorted(vols, reverse=True), vols
    # and the skew SPANS something: a flat curve would pass the ordering above trivially.
    assert vols[0] - vols[-1] > 0.02, vols


def test_the_BACHELIER_smile_is_MEASURED_and_is_also_decreasing_in_the_strike():
    # Bachelier's shape is reported rather than taken from the chapter, which plots the CEV and
    # displaced-diffusion skews and not this one. A normal forward has strike-independent ABSOLUTE
    # vol, so the Black-equivalent RELATIVE vol must fall as the strike rises; the oracle is that
    # measured fall, with the ATM tie below as its one algebraic anchor.
    forward, years, df = 0.05, 1.0, math.exp(-0.05)
    unit = _period(forward, df)
    vols = [
        _implied(bachelier_caplet(unit, k, NORMAL_VOL ** 2 * years, CapFloorKind.CAP),
                 forward, k, years, df)
        for k in SMILE_STRIKES
    ]
    assert vols == sorted(vols, reverse=True), vols
    assert vols[0] - vols[-1] > 0.02, vols


def test_at_the_money_the_normal_and_black_vols_tie_through_a_KNOWN_factor():
    # The one thing the closed forms force, so the only slope-free assertion worth making. At
    # `F = K` the Bachelier price is `tau*df*s/sqrt(2*pi)` and Black's is `tau*df*F*(2 Phi(V/2) - 1)`;
    # to first order in `V` the latter is `tau*df*F*V/sqrt(2*pi)`, so `sigma_N ~ sigma_B * F`.
    forward, years, df = 0.05, 1.0, math.exp(-0.05)
    unit = _period(forward, df)
    px = bachelier_caplet(unit, forward, NORMAL_VOL ** 2 * years, CapFloorKind.CAP)
    implied = _implied(px, forward, forward, years, df)
    assert implied == pytest.approx(NORMAL_VOL / forward, rel=5e-3)
    # and the first-order tie is NOT exact, which is why it is banded and not asserted as equality
    assert implied != NORMAL_VOL / forward


# ============================================================================================
# 6. The engine's composition — the strip, the curve, and invariant 6.
# ============================================================================================

def test_the_STRIP_is_the_SUM_of_its_live_caplets_for_both_models():
    curve = CURVE
    for model, pricer, extra in (
        (BachelierModel(_snapshot(), NORMAL_VOL), bachelier_caplet, ()),
        (DisplacedDiffusionModel(_snapshot(), LOGNORMAL_VOL, SHIFT), displaced_caplet, (SHIFT,)),
    ):
        cap = _cap()
        vol = NORMAL_VOL if not extra else LOGNORMAL_VOL
        expected = 0.0
        for start, pay, accrual in cap.schedule.periods():
            if pay <= VAL:
                continue
            unit = CapletPeriod(curve.df(start), curve.df(pay), accrual)
            years = MEASURE.year_fraction(start)
            expected += N * pricer(unit, cap.strike, vol * vol * years, *extra,
                                   CapFloorKind.CAP)
        assert _pv(cap, model) == pytest.approx(expected, rel=1e-14), type(model)


def test_the_STRADDLE_premise_the_historical_period_is_worth_a_DIFFERENT_amount_excluded():
    # §4's tenth degenerate shape, asserted rather than believed written -- the bite that armed the
    # proactive sweep was a fixture that only *reported* straddling. Two halves, both needed:
    #   (a) the schedule really does straddle, and
    #   (b) the excluded caplet is worth something, so a dropped filter CHANGES the answer.
    periods = list(_cap().schedule.periods())
    historical = [p for p in periods if p[1] <= VAL]
    assert len(historical) == 1, periods                        # (a)
    assert len(periods) - len(historical) == 4, periods
    start, pay, accrual = historical[0]
    assert MEASURE.year_fraction(start) < 0.0 and accrual != pytest.approx(1.0)
    leaked = N * bachelier_caplet(                              # (b)
        CapletPeriod(CURVE.df(start), CURVE.df(pay), accrual),
        0.03, NORMAL_VOL ** 2 * abs(MEASURE.year_fraction(start)), CapFloorKind.CAP,
    )
    assert abs(leaked) > 1e-4 * N, leaked


def test_a_fully_SETTLED_strip_is_PV_ZERO_and_not_a_failure_for_both_models():
    after = DATES[-1] + timedelta(days=1)
    curve = DiscountCurve(TimeMeasure(after, DC), (0.0, 1.0, 4.0), (0.026, 0.030, 0.034))
    for model in (
        BachelierModel(_snapshot(after, curve), NORMAL_VOL),
        DisplacedDiffusionModel(_snapshot(after, curve), LOGNORMAL_VOL, SHIFT),
    ):
        assert _pv(_cap(), model) == 0.0, type(model)


def test_a_FLOOR_strip_prices_through_the_same_engine_for_both_models():
    for model in (
        BachelierModel(_snapshot(), NORMAL_VOL),
        DisplacedDiffusionModel(_snapshot(), LOGNORMAL_VOL, SHIFT),
    ):
        cap = _pv(_cap(0.03, CapFloorKind.CAP), model)
        floor = _pv(_cap(0.03, CapFloorKind.FLOOR), model)
        assert cap > 0.0 and floor > 0.0, (cap, floor)
        # the strip-level parity: cap - floor is the forward-rate swap, no option formula either side
        swap = 0.0
        for start, pay, accrual in _cap().schedule.periods():
            if pay <= VAL:
                continue
            forward = (CURVE.df(start) / CURVE.df(pay) - 1.0) / accrual
            swap += N * accrual * CURVE.df(pay) * (forward - 0.03)
        assert cap - floor == pytest.approx(swap, rel=1e-12), type(model)


# ============================================================================================
# 7. A SECOND independent killer for the variance construction.
#
# The first sweep of this tranche killed all ten planted mutants but flagged three as
# SINGLE-KILLER oracles (§9): the variance horizon moved from the fixing date to the payment
# date, and each model's variance made linear in vol instead of quadratic. All three died only to
# `test_the_STRIP_is_the_SUM_of_its_live_caplets_for_both_models`, so deleting that one test would
# silently reopen three holes — and the horizon one is the hazard this module's own docstring
# names in prose ("with a flat vol it would over-state every caplet by sqrt(T_i/T_i-1), with every
# requirement oracle still green"). Naming a hazard and then leaving it on one test is the gap §9
# asks the sweep to report, so it is closed here rather than recorded.
#
# The instrument is an AT-THE-MONEY closed form, which shares no line with the strip walk: at
# `F = K` the Bachelier caplet collapses to `tau*df*s/sqrt(2*pi)` exactly (`d1 = 0`, `N(0) = 1/2`,
# so the first term vanishes), and the displaced-diffusion one to `tau*df*(F+a)*(2 N(V/2) - 1)`.
# Both read the horizon off the SCHEDULE independently and both depend on `sigma` through
# `sigma*sqrt(T_fix)`, so a wrong horizon or a missing square breaks them.
# ============================================================================================

def test_the_at_the_money_BACHELIER_strip_hits_its_exact_closed_form():
    # The fixture's fixing horizon is 0, which would make this vacuous -- so the premise is
    # asserted and the period is re-struck one period later, where the horizon is real.
    dates = (DATES[2], DATES[3])
    schedule = CouponSchedule(dates, DC)
    start, pay, accrual = next(iter(schedule.periods()))
    years = MEASURE.year_fraction(start)
    assert years > 0.4, years                          # the premise: a non-zero fixing horizon
    forward = (CURVE.df(start) / CURVE.df(pay) - 1.0) / accrual
    cap = NominalCap(schedule, forward, Money(N, EUR), CapFloorKind.CAP)
    want = N * accrual * CURVE.df(pay) * (
        NORMAL_VOL * math.sqrt(years) / math.sqrt(2.0 * math.pi)
    )
    assert _pv(cap, BachelierModel(_snapshot(), NORMAL_VOL)) == pytest.approx(want, rel=1e-13)


def test_the_at_the_money_DISPLACED_strip_hits_its_exact_closed_form():
    dates = (DATES[2], DATES[3])
    schedule = CouponSchedule(dates, DC)
    start, pay, accrual = next(iter(schedule.periods()))
    years = MEASURE.year_fraction(start)
    assert years > 0.4, years
    forward = (CURVE.df(start) / CURVE.df(pay) - 1.0) / accrual
    cap = NominalCap(schedule, forward, Money(N, EUR), CapFloorKind.CAP)
    root = LOGNORMAL_VOL * math.sqrt(years)
    want = N * accrual * CURVE.df(pay) * (forward + SHIFT) * (2.0 * norm_cdf(0.5 * root) - 1.0)
    model = DisplacedDiffusionModel(_snapshot(), LOGNORMAL_VOL, SHIFT)
    assert _pv(cap, model) == pytest.approx(want, rel=1e-13)


def test_the_fixing_and_PAYMENT_horizons_differ_enough_to_tell_the_two_apart():
    # The premise that gives the two oracles above their power over the horizon mutation: if the
    # fixing and payment year fractions were close, using the wrong one would barely move the price.
    start, pay = DATES[2], DATES[3]
    fixing, payment = MEASURE.year_fraction(start), MEASURE.year_fraction(pay)
    assert math.sqrt(payment / fixing) > 1.3, (fixing, payment)
