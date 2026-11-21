"""The CEV caplet pricer (L4). BROCKHAUS-2016 §6.1 (Cox 1996); MCINERNEY-ZASTAWNIAK §8.3.

**The source of the formula is BROCKHAUS-2016, not the arc's own text, and that is finding one.**
MCINERNEY-ZASTAWNIAK §8.3 presents the CEV SDE (8.5), the absorbing boundary, two *simulated
histograms* (Figures 8.1/8.2) and then pivots to the displaced-diffusion approximation as the
practical answer. It prints **no closed form**, no chi-squared argument triple, and no worked
example; the whole book's only other mention of the non-central chi-squared is a one-line taxonomy
remark about Cox-Ingersoll-Ross in ch 3. §4 forbids fitting numbers to a figure, so the caplet is
transcribed from BROCKHAUS-2016 §6.1 p.79-80, which prints Cox's (1996) formula and is already
`worked-from`.

**Finding two: the printed formula is WRONG, and both renderings print it identically.** As printed:

    C(K,T) = S0 Q(2w; 2 + 1/(1-rho), 2z) - K [1 - Q(2w; 2 - 1/(1-rho), 2z)]

Two defects. (a) Both chi-squared terms take `2w` as the argument and `2z` as the noncentrality;
Schroder's result **swaps** them between the terms. (b) The first term is missing its `1 -`. The
tell is not a second rendering — `-raw` and `-layout` agree, so this is the source's own misprint,
the case §4 says only a requirement oracle can catch. Three requirements catch it:

  - **`K -> 0` must give `S0`** (the forward is a martingale). As printed it gives 0.
  - **`K -> inf` must give 0.** As printed it gives `S0`.
  - **At `rho = 0.5` — the value the other source calls "market practice" — the printed form is
    UNDEFINED**: `2 - 1/(1-0.5)` is exactly **zero** degrees of freedom.

The corrected form was **ruled by Monte Carlo**, not by recall:

    C(K,T) = S0 * Qbar(2w; 2 + 1/(1-rho), 2z) - K * Q(2z; 1/(1-rho), 2w)

with `Qbar` the survival function, `z = F^(2(1-rho)) / (2(1-rho)^2 V)` and `w` the same in `K`. It
matches a 600k-path Euler scheme with absorption to under 1 SE at three strikes. The MC anchor is
committed below because it is the only oracle here that does not share the formula with the code.

**Finding three: the two reductions are ASYMMETRIC, and neither is what was expected.**

  - **`beta -> 1` (Black) is numerically UNREACHABLE** — `1/(1-beta)` diverges and the formula
    returns `nan` by `beta = 0.99999`. So it earns an **O(1-beta) convergence rate over a measured
    regime** (§4's tranche-66 clause, with tranche 67's "measure the regime for THIS model"): the
    ratio to Black runs 1.347 -> 1.0303 -> 1.00299 -> 1.000299 as `1-beta` falls by decades, and the
    formula is numerically sound to `1-beta = 1e-4` and not beyond.
  - **`beta = 0` (Bachelier) is reachable exactly, but is NOT an identity.** CEV at `beta = 0` is a
    normal forward *with an absorbing barrier at zero*, and Bachelier has none, so they agree only
    where the absorbed mass is negligible. Measured against `F/sigma_N`: agreement to 3e-10 at
    `F/sigma_N = 5.9`, and divergence to **0.50% at 1.25** and **12.7% at 0.62**. The oracle is
    therefore equality *in a stated regime* plus a companion asserting the divergence outside it —
    both halves, so the regime is measured rather than assumed.

**And MCINERNEY-ZASTAWNIAK still anchors something**, structurally rather than by its pixels: the
text states in prose that the absorbed fraction is "over 10%" at `F(0) = 1%` and "typically low" at
`F(0) = 5%` (both at `beta = 0.5`, `sigma = 0.1`, `t = 1`). The closed form's absorption probability
`1 - chi2(2z; 1/(1-beta))` gives **13.53%** and **0.0045%**. That is §4's "reproduce its structure,
not its pixels" satisfied with the figures' own stated numbers.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.cev_caplet import cev_absorption_probability, cev_call, cev_caplet
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    Money,
    PricingResult,
)
from pricebook.foundation.distributions import normal_positive_part
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.black import black_implied_vol, black_price
from pricebook.models.cev import CevModel
from pricebook.models.hull_white import CapletPeriod
from pricebook.products import NominalCap

scipy_stats = pytest.importorskip("scipy.stats")
numpy = pytest.importorskip("numpy")

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
N = 10_000_000.0
KEY = CurveKey(CurveRole.DISCOUNT, EUR)
# Semi-annual, and STRADDLING the valuation date: the k=-1 period pays on `VAL`, so invariant 6 has
# something to exclude (§4's tenth shape, asserted below rather than believed).
DATES = tuple(VAL + timedelta(days=180 * k) for k in range(-1, 5))
CURVE = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 4.0), (0.026, 0.030, 0.034, 0.038))
BETA = 0.5
CEV_VOL = 0.1          # units of rate^(1-beta) — not comparable to a Black vol


def _snapshot(vd: date = VAL, curve: DiscountCurve = CURVE) -> MarketSnapshot:
    return MarketSnapshot(valuation_date=vd, curves=CurveSet({KEY: curve}))


def _cap(strike: float = 0.03, kind: CapFloorKind = CapFloorKind.CAP) -> NominalCap:
    return NominalCap(CouponSchedule(DATES, DC), strike, Money(N, EUR), kind)


def _pv(product: NominalCap, model: object) -> float:
    out = price(product, model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _period(forward: float, df_pay: float, accrual: float = 1.0) -> CapletPeriod:
    return CapletPeriod(df_pay * (1.0 + accrual * forward), df_pay, accrual)


def _scipy_cox(forward: float, strike: float, variance: float, beta: float) -> float:
    """Cox's formula, RULED form, on scipy's chi-squareds — independent of our L0 primitive."""
    power = 2.0 * (1.0 - beta)
    scale = 2.0 * (1.0 - beta) ** 2 * variance
    z, w = forward**power / scale, strike**power / scale
    return forward * scipy_stats.ncx2.sf(2 * w, 2 + 1 / (1 - beta), 2 * z) - strike * (
        scipy_stats.ncx2.cdf(2 * z, 1 / (1 - beta), 2 * w)
    )


def _as_printed(forward: float, strike: float, variance: float, beta: float) -> float:
    """The formula EXACTLY as BROCKHAUS-2016 prints it, so the requirement oracles below can show
    it failing. Kept in the test, never in `src/` — it is the refuted candidate, not an option."""
    power = 2.0 * (1.0 - beta)
    scale = 2.0 * (1.0 - beta) ** 2 * variance
    z, w = forward**power / scale, strike**power / scale
    first = scipy_stats.ncx2.cdf(2 * w, 2 + 1 / (1 - beta), 2 * z)
    second = scipy_stats.ncx2.cdf(2 * w, 2 - 1 / (1 - beta), 2 * z)
    return forward * first - strike * (1.0 - second)


# ============================================================================================
# 1. The POINTWISE transcription, on a reference independent of our chi-squared primitive.
# ============================================================================================

CASES = (
    (0.05, 0.03, 0.01, 0.5),
    (0.05, 0.05, 0.01, 0.5),
    (0.05, 0.08, 0.01, 0.5),
    (0.03, 0.04, 0.004, 0.3),
    (0.06, 0.05, 0.02, 0.7),
)


@pytest.mark.parametrize("forward,strike,variance,beta", CASES)
def test_the_CEV_call_matches_the_p79_transcription_on_scipys_chi_squareds(
    forward, strike, variance, beta
):
    got, want = cev_call(forward, strike, variance, beta), _scipy_cox(
        forward, strike, variance, beta
    )
    assert got == pytest.approx(want, rel=1e-10), (got, want)


# ============================================================================================
# 2. The REQUIREMENTS that ruled the form — and the printed form failing each.
# ============================================================================================

def test_a_ZERO_strike_call_is_the_FORWARD_because_the_forward_is_a_martingale():
    # True for all parameters, so it survives a transcription error — and it is the first of the
    # three requirements that condemned the printed form.
    for forward, _, variance, beta in CASES:
        assert cev_call(forward, 1e-12, variance, beta) == pytest.approx(forward, rel=1e-9)


def test_the_PRINTED_form_fails_the_zero_strike_requirement():
    # The companion that proves the requirement discriminates (tranche 15): asserted here so the
    # ruling is in the suite, not only in the log.
    assert _as_printed(0.05, 1e-12, 0.01, 0.5) == pytest.approx(0.0, abs=1e-15)


def test_a_far_out_of_the_money_call_goes_to_ZERO_not_to_the_forward():
    for forward, _, variance, beta in CASES:
        assert cev_call(forward, 100.0, variance, beta) == pytest.approx(0.0, abs=1e-12)


def test_the_PRINTED_form_fails_the_deep_out_of_the_money_requirement():
    assert _as_printed(0.05, 100.0, 0.01, 0.5) == pytest.approx(0.05, rel=1e-6)


def test_the_PRINTED_form_is_UNDEFINED_at_the_sources_own_market_practice_beta():
    # `2 - 1/(1-0.5)` is exactly zero degrees of freedom. MCINERNEY-ZASTAWNIAK p.148: "Market
    # practice is to put beta = 0.5."
    assert 2.0 - 1.0 / (1.0 - 0.5) == 0.0
    assert math.isnan(_as_printed(0.05, 0.05, 0.01, 0.5))


# ============================================================================================
# 3. The MONTE CARLO anchor — the only oracle here that does not share the formula.
# ============================================================================================

def _mc_cev(forward: float, strike: float, variance: float, beta: float,
            paths: int = 120_000, steps: int = 1000) -> tuple[float, float]:
    """Euler on `dF = F^beta dW` over the TOTAL VARIANCE, absorbed at zero (BROCKHAUS p.79).

    The time change is why the horizon is the variance and not the expiry: eq (6.9)'s driftless
    process has unit volatility, so `V` IS its clock."""
    rng = numpy.random.default_rng(20261121)
    step = variance / steps
    root = math.sqrt(step)
    level = numpy.full(paths, forward)
    for _ in range(steps):
        live = level > 0.0
        level[live] += numpy.power(level[live], beta) * rng.standard_normal(
            int(live.sum())
        ) * root
        level[level < 0.0] = 0.0
    payoff = numpy.maximum(level - strike, 0.0)
    return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(paths))


# `4*SE + 0.5%`: the two error sources banded SEPARATELY (§4). The 0.5% is the measured ceiling on
# the Euler-with-absorption discretisation bias — at 600k paths the deviations across 250/1000/4000
# steps all sat inside +/-0.28%, below the 0.18-0.32% sampling noise, so 0.5% is an upper bound and
# not a fitted number. RESOLUTION: at 120k paths the SE is ~0.54% of price, so a green here rules
# out a defect larger than about **2.7%** — which is why the pointwise and requirement oracles above
# carry the precision and this one carries the independence.
@pytest.mark.parametrize("strike", (0.03, 0.05, 0.07))
def test_the_closed_form_matches_an_INDEPENDENT_simulation_with_absorption(strike):
    mean, se = _mc_cev(0.05, strike, 0.04, 0.5)
    exact = cev_call(0.05, strike, 0.04, 0.5)
    assert abs(mean - exact) < 4.0 * se + 0.005 * exact, (mean, se, exact)


def test_the_simulation_ABSORBS_a_material_fraction_so_the_barrier_is_exercised():
    # The premise: a simulation in which nothing reaches zero would agree with a barrier-free
    # formula too, and the barrier is what distinguishes CEV from Bachelier at `beta = 0`.
    assert cev_absorption_probability(0.05, 0.04, 0.5) > 0.05


# ============================================================================================
# 4. MCINERNEY-ZASTAWNIAK's figures, reproduced STRUCTURALLY (§4: never from the pixels).
# ============================================================================================

def test_the_absorbed_fraction_reproduces_the_sources_two_PROSE_claims():
    # p.148, beta = 0.5, sigma = 0.1, t = 1. The text's own numbers, not the plots'.
    low = cev_absorption_probability(0.01, 0.1**2 * 1.0, 0.5)
    high = cev_absorption_probability(0.05, 0.1**2 * 1.0, 0.5)
    assert low > 0.10, f'"over 10%" at F(0)=1%, got {low:.2%}'          # Figure 8.2
    assert low == pytest.approx(0.1353, abs=5e-4), low
    assert high < 0.001, f'"typically low" at F(0)=5%, got {high:.2%}'  # Figure 8.1
    assert high == pytest.approx(4.5e-5, rel=0.05), high


def test_the_absorption_probability_matches_a_CENTRAL_chi_squared_independently():
    for forward, variance, beta in ((0.05, 0.01, 0.5), (0.01, 0.01, 0.5), (0.04, 0.02, 0.3)):
        z = forward ** (2.0 * (1.0 - beta)) / (2.0 * (1.0 - beta) ** 2 * variance)
        want = scipy_stats.chi2.sf(2.0 * z, 1.0 / (1.0 - beta))
        assert cev_absorption_probability(forward, variance, beta) == pytest.approx(
            want, rel=1e-10
        )


# ============================================================================================
# 5. The two ASYMMETRIC reductions.
# ============================================================================================

def test_the_BLACK_limit_is_unreachable_so_it_is_a_RATE_over_a_measured_regime():
    # §4's tranche-66 clause. The regime is measured for THIS model (tranche 67): sound to
    # `1 - beta = 1e-4`, `nan` by 1e-5.
    forward, strike, variance = 0.05, 0.05, 0.04
    target = black_price(forward, strike, variance, CapFloorKind.CAP)
    errors = []
    for gap in (1e-1, 1e-2, 1e-3, 1e-4):
        errors.append(abs(cev_call(forward, strike, variance, 1.0 - gap) / target - 1.0))
    assert errors == sorted(errors, reverse=True), errors
    for coarse, fine in zip(errors, errors[1:]):
        assert 5.0 < coarse / fine < 20.0, errors        # first order in (1 - beta), decade by decade
    assert errors[-1] < 1e-3, errors


def test_OUTSIDE_that_regime_the_closed_form_DEGRADES_so_the_regime_is_stated_not_assumed():
    assert math.isnan(cev_call(0.05, 0.05, 0.04, 1.0 - 1e-5))


def test_the_BACHELIER_limit_holds_only_where_ABSORPTION_is_negligible():
    # Reachable exactly at `beta = 0` — but an identity only in a stated regime, because CEV
    # absorbs at zero and Bachelier does not. Both halves asserted.
    forward, normal_vol = 0.05, 0.0085
    variance = normal_vol * normal_vol
    assert cev_absorption_probability(forward, variance, 0.0) < 1e-6         # the regime premise
    for strike in (0.03, 0.05, 0.07):
        got = cev_call(forward, strike, variance, 0.0)
        want = normal_positive_part(forward - strike, normal_vol)
        assert got == pytest.approx(want, rel=1e-6), (strike, got, want)


def test_and_it_FAILS_outside_that_regime_which_is_what_makes_the_regime_a_measurement():
    forward, normal_vol = 0.05, 0.08                     # F/sigma_N = 0.62
    variance = normal_vol * normal_vol
    assert cev_absorption_probability(forward, variance, 0.0) > 0.4
    got = cev_call(forward, forward, variance, 0.0)
    want = normal_positive_part(0.0, normal_vol)
    assert abs(got / want - 1.0) > 0.10, (got, want)


# ============================================================================================
# 6. Requirement oracles on the caplet, and the measured skew.
# ============================================================================================

@pytest.mark.parametrize("forward,strike,variance,beta", CASES)
def test_put_call_PARITY(forward, strike, variance, beta):
    accrual, df_pay = 0.4932, 0.94
    unit = _period(forward, df_pay, accrual)
    cap = cev_caplet(unit, strike, variance, beta, CapFloorKind.CAP)
    floor = cev_caplet(unit, strike, variance, beta, CapFloorKind.FLOOR)
    want = accrual * df_pay * (unit.forward_rate() - strike)
    assert cap - floor == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("forward,strike", ((0.05, 0.03), (0.05, 0.07)))
def test_a_vanishing_variance_collapses_to_the_INTRINSIC_value(forward, strike):
    accrual, df_pay = 0.4932, 0.94
    unit = _period(forward, df_pay, accrual)
    got = cev_caplet(unit, strike, 1e-14, BETA, CapFloorKind.CAP)
    assert got == pytest.approx(accrual * df_pay * max(forward - strike, 0.0), abs=1e-11)


def test_the_price_RISES_with_the_volatility():
    unit = _period(0.05, 0.94, 0.4932)
    prices = [cev_caplet(unit, 0.05, v * v, BETA, CapFloorKind.CAP)
              for v in (0.05, 0.1, 0.2, 0.4)]
    assert prices == sorted(prices), prices
    assert prices[-1] > 2.0 * prices[0]


def test_the_CEV_skew_is_MEASURED_and_is_DOWNWARD_which_is_all_the_finance_forces():
    # §4's obligation-not-form clause: the sign is forced (local vol falls with the level for
    # beta < 1, so low strikes imply higher Black vol), the SHAPE is measured and reported.
    forward, years, df = 0.05, 1.0, math.exp(-0.05)
    unit = _period(forward, df)
    vols = []
    for strike in (0.03, 0.04, 0.05, 0.06, 0.07):
        px = cev_caplet(unit, strike, CEV_VOL**2 * years, BETA, CapFloorKind.CAP)
        vols.append(black_implied_vol(px / df, forward, strike, years, CapFloorKind.CAP))
    assert vols == sorted(vols, reverse=True), vols
    assert vols[0] - vols[-1] > 0.01, vols


def test_a_HIGHER_exponent_gives_a_FLATTER_skew_approaching_blacks_flat_smile():
    # The cross-model continuity the spec asks for, with the units caveat made explicit: the vol
    # is RE-SCALED per beta (`sigma * F^(1-beta)` held fixed), because a CEV vol is not comparable
    # across exponents and a fixed `sigma` would confound the skew with a vol change.
    forward, years, df = 0.05, 1.0, math.exp(-0.05)
    unit = _period(forward, df)
    spans = []
    for beta in (0.2, 0.5, 0.9):
        vol = 0.22 * forward ** (1.0 - beta)
        vols = [
            black_implied_vol(
                cev_caplet(unit, k, vol**2 * years, beta, CapFloorKind.CAP) / df,
                forward, k, years, CapFloorKind.CAP,
            )
            for k in (0.04, 0.05, 0.06)
        ]
        spans.append(vols[0] - vols[-1])
    assert spans == sorted(spans, reverse=True), spans
    assert spans[-1] < 0.3 * spans[0], spans


# ============================================================================================
# 7. The engine's composition.
# ============================================================================================

def test_the_STRIP_is_the_SUM_of_its_live_caplets():
    cap = _cap()
    model = CevModel(_snapshot(), CEV_VOL, BETA)
    expected = 0.0
    for start, pay, accrual in cap.schedule.periods():
        if pay <= VAL:
            continue
        unit = CapletPeriod(CURVE.df(start), CURVE.df(pay), accrual)
        years = MEASURE.year_fraction(start)
        expected += N * cev_caplet(unit, cap.strike, CEV_VOL**2 * years, BETA, CapFloorKind.CAP)
    assert _pv(cap, model) == pytest.approx(expected, rel=1e-13)


def test_the_STRADDLE_premise_the_historical_period_is_worth_a_DIFFERENT_amount_excluded():
    periods = list(_cap().schedule.periods())
    historical = [p for p in periods if p[1] <= VAL]
    assert len(historical) == 1 and len(periods) == 5, periods
    start, pay, accrual = historical[0]
    assert MEASURE.year_fraction(start) < 0.0 and accrual != pytest.approx(1.0)
    leaked = N * cev_caplet(
        CapletPeriod(CURVE.df(start), CURVE.df(pay), accrual),
        0.03, CEV_VOL**2 * abs(MEASURE.year_fraction(start)), BETA, CapFloorKind.CAP,
    )
    assert abs(leaked) > 1e-4 * N, leaked


def test_a_fully_SETTLED_strip_is_PV_ZERO_and_not_a_failure():
    after = DATES[-1] + timedelta(days=1)
    curve = DiscountCurve(TimeMeasure(after, DC), (0.0, 1.0, 4.0), (0.026, 0.030, 0.034))
    assert _pv(_cap(), CevModel(_snapshot(after, curve), CEV_VOL, BETA)) == 0.0
