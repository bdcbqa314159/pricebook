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
from pricebook.foundation.noncentral_chisquare import noncentral_chisquare_cdf
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
# Below the historical period's own forward of 2.6167%, so a settled caplet that fell through the
# invariant-6 filter would return a NON-ZERO intrinsic. At the 3% default strike it returns zero —
# the same answer PV-0 gives — and the dropped filter is invisible. See the straddle premise below.
IN_THE_MONEY_STRIKE = 0.02


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


@pytest.mark.parametrize("beta", (0.3, 0.4, 0.49))
def test_the_PRINTED_form_fails_the_zero_strike_requirement(beta):
    # The companion that proves the requirement discriminates (tranche 15): asserted here so the
    # ruling is in the suite, not only in the log.
    #
    # **Shown at beta < 0.5, because that is where the printed form is even DEFINED.** Its second
    # term asks for `2 - 1/(1-beta)` degrees of freedom, which is positive only below 0.5, zero at
    # exactly 0.5 and negative above — so the printed formula is undefined on the whole upper half
    # of the source's own range, including its stated market practice. Where it is defined it is
    # wrong, and it fails this requirement by returning roughly MINUS the strike.
    assert 2.0 - 1.0 / (1.0 - beta) > 0.0, beta
    assert _as_printed(0.05, 1e-12, 0.01, beta) < 0.0
    assert abs(_as_printed(0.05, 1e-12, 0.01, beta) - 0.05) > 0.049


def test_a_far_out_of_the_money_call_goes_to_ZERO_not_to_the_forward():
    for forward, _, variance, beta in CASES:
        assert cev_call(forward, 100.0, variance, beta) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("beta", (0.3, 0.4, 0.49))
def test_the_PRINTED_form_fails_the_deep_out_of_the_money_requirement(beta):
    # It returns the FORWARD where the answer is zero — the two requirements fail in opposite
    # directions, which is the signature of the swapped argument/noncentrality pairs rather than of
    # a scale error.
    assert _as_printed(0.05, 100.0, 0.01, beta) == pytest.approx(0.05, rel=1e-9)


def test_the_PRINTED_form_is_UNDEFINED_at_the_sources_own_market_practice_beta():
    # `2 - 1/(1-0.5)` is exactly zero degrees of freedom. MCINERNEY-ZASTAWNIAK p.148: "Market
    # practice is to put beta = 0.5."
    assert 2.0 - 1.0 / (1.0 - 0.5) == 0.0
    assert math.isnan(_as_printed(0.05, 0.05, 0.01, 0.5))


# ============================================================================================
# 3. The MONTE CARLO anchor — the only oracle here that does not share the formula.
# ============================================================================================

def _simulate_cev(forward: float, variance: float, beta: float,
                  paths: int = 120_000, steps: int = 1000):
    """Euler on `dF = F^beta dW` over the TOTAL VARIANCE, absorbed at zero (BROCKHAUS p.79).

    The time change is why the horizon is the variance and not the expiry: eq (6.9)'s driftless
    process has unit volatility, so `V` IS its clock.

    Returns the TERMINAL DISTRIBUTION, not a price, so one simulation serves every strike — the
    paths do not depend on `K`. Running it per strike (as the first cut did) computed the same
    distribution three times over and put the file at **13.4s on the Windows CI runner against the
    10.0s ceiling**, while reading 3.1s locally: §9's "for a numerically-heavy file, CI is the
    measurement". The reduction is **lossless** — identical scheme, identical seed, identical paths,
    and each strike's assertion still compares its own mean against its own closed form; sharing the
    draws only correlates the three tests' sampling errors, which no assertion depends on being
    independent."""
    rng = numpy.random.default_rng(20261121)
    root = math.sqrt(variance / steps)
    level = numpy.full(paths, forward)
    for _ in range(steps):
        live = level > 0.0
        level[live] += numpy.power(level[live], beta) * rng.standard_normal(
            int(live.sum())
        ) * root
        level[level < 0.0] = 0.0
    return level


@pytest.fixture(scope="module")
def cev_terminal():
    """The shared terminal distribution at `F = 5%`, `V = 0.04`, `beta = 0.5`."""
    return _simulate_cev(0.05, 0.04, 0.5)


def _mc_price(terminal, strike: float) -> tuple[float, float]:
    payoff = numpy.maximum(terminal - strike, 0.0)
    return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(terminal.size))


# `4*SE + 0.5%`: the two error sources banded SEPARATELY (§4). The 0.5% is the measured ceiling on
# the Euler-with-absorption discretisation bias — at 600k paths the deviations across 250/1000/4000
# steps all sat inside +/-0.28%, below the 0.18-0.32% sampling noise, so 0.5% is an upper bound and
# not a fitted number. RESOLUTION: at 120k paths the SE is ~0.54% of price, so a green here rules
# out a defect larger than about **2.7%** — which is why the pointwise and requirement oracles above
# carry the precision and this one carries the independence.
@pytest.mark.parametrize("strike", (0.03, 0.05, 0.07))
def test_the_closed_form_matches_an_INDEPENDENT_simulation_with_absorption(strike, cev_terminal):
    mean, se = _mc_price(cev_terminal, strike)
    exact = cev_call(0.05, strike, 0.04, 0.5)
    assert abs(mean - exact) < 4.0 * se + 0.005 * exact, (mean, se, exact)


def _variant_dof_first(forward, strike, variance, beta):
    """A plausible transcription slip: the first term's dof read as `2 + 2/(1-beta)`."""
    dof = 1.0 / (1.0 - beta)
    z, w = _z_ref(forward, variance, beta), _z_ref(strike, variance, beta)
    return forward * scipy_stats.ncx2.sf(2 * w, 2 + 2 * dof, 2 * z) - strike * (
        scipy_stats.ncx2.cdf(2 * z, dof, 2 * w)
    )


def _variant_dof_second(forward, strike, variance, beta):
    """Another: the second term's dof read as `1/(1-beta) + 2`."""
    dof = 1.0 / (1.0 - beta)
    z, w = _z_ref(forward, variance, beta), _z_ref(strike, variance, beta)
    return forward * scipy_stats.ncx2.sf(2 * w, 2 + dof, 2 * z) - strike * (
        scipy_stats.ncx2.cdf(2 * z, dof + 2, 2 * w)
    )


def _variant_unsquared(forward, strike, variance, beta):
    """And a third: `z` and `w` losing the square on `(1-beta)`."""
    dof = 1.0 / (1.0 - beta)
    scale = 2.0 * (1.0 - beta) * variance
    z, w = forward ** (2 * (1 - beta)) / scale, strike ** (2 * (1 - beta)) / scale
    return forward * scipy_stats.ncx2.sf(2 * w, 2 + dof, 2 * z) - strike * (
        scipy_stats.ncx2.cdf(2 * z, dof, 2 * w)
    )


def _z_ref(level, variance, beta):
    return level ** (2.0 * (1.0 - beta)) / (2.0 * (1.0 - beta) ** 2 * variance)


INTERIOR_VARIANTS = (
    ("the first term's dof doubled", _variant_dof_first),
    ("the second term's dof shifted by 2", _variant_dof_second),
    ("z and w unsquared in (1-beta)", _variant_unsquared),
)


@pytest.mark.parametrize("label,variant", INTERIOR_VARIANTS)
def test_the_requirement_oracles_are_BLIND_to_these_variants_which_is_why_the_MC_is_needed(
    label, variant
):
    # **The band's RESOLUTION, stated per DEFECT and not as a property of the band** (F-5.3), and
    # the concrete demonstration of why a reconstructed formula needs the simulation at all.
    #
    # Each variant below is a plausible transcription slip that **satisfies all three requirement
    # oracles** — `K -> 0` gives the forward, `K -> inf` gives zero — because those pin the
    # STRUCTURE and are silent on the interior MAGNITUDE (the tranche-43 gap). So the requirements
    # cannot distinguish them from the corrected form, and the Monte-Carlo anchor is what does.
    forward, strike, variance, beta = 0.05, 0.05, 0.04, 0.5
    assert variant(forward, 1e-12, variance, beta) == pytest.approx(forward, rel=1e-9), label
    assert variant(forward, 100.0, variance, beta) == pytest.approx(0.0, abs=1e-12), label

    exact = cev_call(forward, strike, variance, beta)
    deviation = abs(variant(forward, strike, variance, beta) / exact - 1.0)
    # Measured: 33.9%, 47.2% and 37.2%. The committed MC band below is 4*SE + 0.5% = ~2.7% at
    # 120k paths, so each sits 12-17x outside it; the 600k run that originally ruled the form
    # resolves ~1.5%. A green on that band is therefore evidence against any interior-magnitude
    # defect above ~2.7%, which is what this test states and measures.
    assert deviation > 0.30, (label, deviation)
    assert deviation > 10.0 * (4.0 * 0.0054 + 0.005), (label, deviation)


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


def test_the_second_terms_ARGUMENT_and_DOF_are_pinned_by_the_absorption_probability():
    # Confirmation that the corrected form is the KNOWN one rather than merely a form that passes
    # our checks — the half of it that is available from the source's own derivation.
    #
    # Brockhaus derives the forward's law before printing the option formula: the squared-Bessel
    # process is non-central chi-squared with the FORWARD's level entering as the noncentrality, so
    # the first term — the probability of finishing above the strike — must be a survival evaluated
    # at the STRIKE's `2w` with the FORWARD's `2z` as noncentrality, which is what this implements.
    # The second term is pinned here: at `w -> 0` its chi-squared degenerates to the central one,
    # and that central value is exactly `1 - P(absorbed)` — a quantity with its own independent
    # validation (the source's prose claims, and the simulation's own absorbed fraction). A wrong
    # argument or dof in the second term moves this identity.
    forward, variance, beta = 0.05, 0.04, 0.5
    dof = 1.0 / (1.0 - beta)
    central = noncentral_chisquare_cdf(2.0 * _z_ref(forward, variance, beta), dof, 0.0)
    assert central == 1.0 - cev_absorption_probability(forward, variance, beta)


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
    # §4's tranche-66 clause: a reduction that is exact in the reals but numerically unreachable
    # earns a convergence RATE over a stated regime, never a band.
    #
    # **And the regime is measured for THIS IMPLEMENTATION, not for the formula** — tranche 67's
    # tightening, which bites harder here than it did there. `scipy` evaluates these chi-squareds to
    # `1 - beta = 1e-4`; our Poisson-mixture primitive cannot, because the noncentrality grows as
    # `1/(1-beta)^2` and the mixture costs O(sqrt(noncentrality)) terms: 0.003s at `1 - beta = 1e-1`,
    # 0.34s at 1e-2, 3.6s at 3e-3, refused at 1e-3. Inheriting scipy's regime would have asserted a
    # convergence we cannot compute. The rate is measured over the regime we can afford.
    forward, strike, variance = 0.05, 0.05, 0.04
    target = black_price(forward, strike, variance, CapFloorKind.CAP)
    gaps = (1e-1, 3e-2, 1e-2)
    errors = [
        abs(cev_call(forward, strike, variance, 1.0 - gap) / target - 1.0) for gap in gaps
    ]
    assert errors == sorted(errors, reverse=True), errors
    # FIRST ORDER: the error falls by the same factor the gap does (3.33x, then 3x).
    for (coarse, fine), ratio in zip(zip(errors, errors[1:]), (10 / 3, 3.0)):
        assert 0.8 * ratio < coarse / fine < 1.3 * ratio, errors
    assert errors[-1] < 0.04, errors


def test_OUTSIDE_that_regime_the_price_is_REFUSED_and_not_quietly_the_intrinsic():
    # The regime is stated by being enforced, which is the stronger form of "stated". And the
    # direction of the refusal is the point: the first draft of the engine guarded on `z` being
    # large and returned the INTRINSIC there, on the reasoning that a large `z` means a negligible
    # variance. That premise is false — `z` also diverges as `beta -> 1` at a FIXED variance — so an
    # at-the-money caplet worth 0.00398 came back as 0.0, a plausible wrong number of exactly the
    # kind §3 forbids. Caught by the rate oracle above, which was written first.
    with pytest.raises(ArithmeticError, match="noncentrality"):
        cev_call(0.05, 0.05, 0.04, 1.0 - 1e-3)


def test_the_engine_turns_that_refusal_into_a_PRICING_FAILURE_not_an_exception():
    # §3's failure-is-a-value contract: the helper raises, the registered pricer returns a value.
    from pricebook.foundation import PricingFailure

    out = price(_cap(), CevModel(_snapshot(), CEV_VOL, 1.0 - 1e-4))
    assert isinstance(out, PricingFailure), out
    assert "noncentrality" in out.reason


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
    # A difference of equals at the at-the-money case (T55/F-8.6) — see the smile-caplet twin.
    assert cap - floor == pytest.approx(want, rel=1e-9, abs=1e-18)


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


def test_the_STRADDLE_premise_measures_the_value_the_dropped_GUARD_would_actually_LEAK():
    # §4's tenth degenerate shape, and this is its FOURTH bite — caught by the sweep after a premise
    # test was already written for it, which is the sharpest version of the lesson yet. Two halves
    # the charter names explicitly, and the first draft of this test got the second one wrong:
    #
    #   (a) the schedule straddles — necessary, and it did;
    #   (b) the excluded flow is worth a DIFFERENT amount excluded than the dropped guard would make
    #       it worth. The first draft asserted (b) using `abs(year_fraction(start))`, i.e. a
    #       positive variance and a diffused price. That is not what the engine computes: with the
    #       filter dropped the year fraction is NEGATIVE, the dispersion collapses to zero and
    #       `cev_call` returns the INTRINSIC. So the premise measured a quantity the mutation never
    #       produces, and the fixture's historical forward (2.6167%) was BELOW its 3% strike, making
    #       that intrinsic exactly 0 — the same answer PV-0 gives. The guard stayed invisible though
    #       the fixture straddled, which is the charter's own "option bypass additionally needs the
    #       expired option IN THE MONEY" sub-trap, verbatim.
    periods = list(_cap().schedule.periods())
    historical = [p for p in periods if p[1] <= VAL]
    assert len(historical) == 1 and len(periods) == 5, periods
    start, pay, accrual = historical[0]
    years = MEASURE.year_fraction(start)
    assert years < 0.0 and accrual != pytest.approx(1.0)                        # (a)

    unit = CapletPeriod(CURVE.df(start), CURVE.df(pay), accrual)
    forward = unit.forward_rate()
    # The value the engine WOULD produce for this period with the filter dropped, computed the way
    # the engine computes it — negative variance and all.
    leaked = N * cev_caplet(unit, IN_THE_MONEY_STRIKE, CEV_VOL**2 * years, BETA, CapFloorKind.CAP)
    assert forward > IN_THE_MONEY_STRIKE, (forward, IN_THE_MONEY_STRIKE)        # (b), the ITM half
    assert abs(leaked) > 1e-3 * N, leaked


def test_an_IN_THE_MONEY_historical_period_is_still_excluded_from_the_mark():
    # The test that actually kills the dropped filter, which the strike-3% fixture could not: here
    # the already-settled period is in the money, so letting it fall through adds a real number.
    cap = _cap(IN_THE_MONEY_STRIKE)
    model = CevModel(_snapshot(), CEV_VOL, BETA)
    live = 0.0
    for start, pay, accrual in cap.schedule.periods():
        if pay <= VAL:
            continue
        unit = CapletPeriod(CURVE.df(start), CURVE.df(pay), accrual)
        live += N * cev_caplet(
            unit, cap.strike, CEV_VOL**2 * MEASURE.year_fraction(start), BETA, CapFloorKind.CAP
        )
    assert _pv(cap, model) == pytest.approx(live, rel=1e-13)


def test_a_historical_FLOORLET_is_excluded_too_which_the_cap_fixture_cannot_show():
    # The second killer for the invariant-6 filter, and it falls out of the same asymmetry that
    # made the first one necessary. The fall-through value is an INTRINSIC, and a cap's and a
    # floor's intrinsics vanish on opposite sides of the forward: at the 3% default strike the
    # historical period's 2.6167% forward makes the CAPLET's intrinsic zero (invisible) and the
    # FLOORLET's 0.383% (visible). So the floor leg kills the dropped filter at the very strike the
    # cap leg cannot — a different arithmetic path (`cev_caplet`'s parity branch), not a second
    # reading of the same one.
    cap = _cap(0.03, CapFloorKind.FLOOR)
    model = CevModel(_snapshot(), CEV_VOL, BETA)
    start, pay, accrual = next(p for p in cap.schedule.periods() if p[1] <= VAL)
    settled = CapletPeriod(CURVE.df(start), CURVE.df(pay), accrual)
    leak = cev_caplet(
        settled, 0.03, CEV_VOL**2 * MEASURE.year_fraction(start), BETA, CapFloorKind.FLOOR
    )
    assert abs(leak) > 1e-3, leak                      # the premise: this one is NOT zero

    live = 0.0
    for s_, p_, a_ in cap.schedule.periods():
        if p_ <= VAL:
            continue
        live += N * cev_caplet(
            CapletPeriod(CURVE.df(s_), CURVE.df(p_), a_), 0.03,
            CEV_VOL**2 * MEASURE.year_fraction(s_), BETA, CapFloorKind.FLOOR,
        )
    assert _pv(cap, model) == pytest.approx(live, rel=1e-13)


def test_the_variance_horizon_is_the_FIXING_date_pinned_a_SECOND_way():
    # The sweep flagged the payment-vs-fixing horizon as a single-killer oracle (§9) — it died only
    # to the strip-sum test, so deleting that one test would silently reopen it. This is a second
    # killer on a different code path: ONE period, no loop, and the horizon written out explicitly
    # rather than reconstructed alongside the strip walk.
    #
    # It is a second killer for the HORIZON, not a second independent price oracle — it shares
    # `cev_caplet` with the engine by design, because what is in question is which date the engine
    # reads, not what the formula computes. The pointwise and Monte-Carlo oracles above carry the
    # formula. (The first attempt here tried to avoid sharing anything, by asserting that two caps
    # with a common fixing date have values in the ratio of their `accrual * df_pay`. That premise
    # is false: a longer period has a different FORWARD too, so the ratio carries the forward's
    # effect as well — measured 2.441 against the weights' 1.965.)
    fixing, pay = DATES[2], DATES[3]
    schedule = CouponSchedule((fixing, pay), DC)
    _, _, accrual = next(iter(schedule.periods()))
    model = CevModel(_snapshot(), CEV_VOL, BETA)
    unit = CapletPeriod(CURVE.df(fixing), CURVE.df(pay), accrual)

    # The premise that gives this power: the two candidate horizons are far enough apart that
    # reading the wrong one moves the price.
    fixing_years, pay_years = MEASURE.year_fraction(fixing), MEASURE.year_fraction(pay)
    assert math.sqrt(pay_years / fixing_years) > 1.3, (fixing_years, pay_years)

    want = N * cev_caplet(unit, 0.03, CEV_VOL**2 * fixing_years, BETA, CapFloorKind.CAP)
    got = _pv(NominalCap(schedule, 0.03, Money(N, EUR), CapFloorKind.CAP), model)
    assert got == pytest.approx(want, rel=1e-13)
    # and the wrong horizon really would differ, so a green here is a measurement
    wrong = N * cev_caplet(unit, 0.03, CEV_VOL**2 * pay_years, BETA, CapFloorKind.CAP)
    assert abs(wrong / want - 1.0) > 0.05, (want, wrong)


def test_a_fully_SETTLED_strip_is_PV_ZERO_and_not_a_failure():
    after = DATES[-1] + timedelta(days=1)
    curve = DiscountCurve(TimeMeasure(after, DC), (0.0, 1.0, 4.0), (0.026, 0.030, 0.034))
    assert _pv(_cap(), CevModel(_snapshot(after, curve), CEV_VOL, BETA)) == 0.0
