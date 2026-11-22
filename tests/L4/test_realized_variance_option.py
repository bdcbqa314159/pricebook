"""Options on realized variance under Heston, and the cross-METHOD anchor T74 lacked. KWOK-ZHENG.

Two pieces, and the first one's value is mostly that it anchors the previous tranche.

**The cross-method anchor (oracle 1).** T74 built the fair variance by **static replication** — a
`1/K²` strip of vanillas — and all its oracles were transcription, reduction and requirement: no
*independent route to the same number*. Heston gives one. Price the model's European options, imply
their Black vols, feed that skew through **T74's `fair_variance` strip**, and compare to the Heston
fair variance in closed form. Two genuinely different derivations — a replication of a model-generated
skew versus a moment of the variance process — with different code on each side. That is the §4
cross-product oracle raised to the level of *method*, and no within-T74 check could have been it.

**The moments, and where each comes from (§4).** The first is **transcribed**: KWOK-ZHENG §4.1 prints
`E[∫ V_t dt] = θT/N + (V_0 − θ)(e^{−κt_{k−1}} − e^{−κt_k})/κ` per monitoring interval, which at the
full horizon is `θT + (V_0 − θ)(1 − e^{−κT})/κ`. The second is **derived**, because the source's
printed second moment is per-interval and for the jump-extended model, and a `-layout` reflow of it
is not something to transcribe. T19's precedent applies — *"the substitute is to derive it and assert
it as a theorem to machine precision, which is stronger than transcription, being immune to a source
misprint"* — so `Var[∫V dt]` is derived from `Cov(V_s,V_t) = e^{−κ(t−s)}Var(V_s)` and asserted here
**two independent ways**: against a double quadrature of that covariance integral, and against
tranche 48's `integrated_variance_transform` (the CIR Laplace transform) by finite differences.

**What this tranche does NOT price, and why that is the spec's own reading.** The spec's target says
"a call/put on *discrete* realized variance", while its out-of-scope defers "the discrete-monitoring
correction" to T78 — and §5.2 (Drimus et al.) is explicitly *"the price of its continuous counterpart
plus an adjustment term due to discrete sampling"*. Those cannot both hold: the discrete price **is**
the continuous one plus the correction T78 owns. So this prices the **continuous** realized-variance
option by the normal approximation, which is the coherent half, and T78 adds the adjustment. Raised
as an `Ask:` rather than settled.
"""

import math
from datetime import date, timedelta

import numpy
import pytest

from pricebook.engine import price
from pricebook.engine.realized_variance_option import (
    heston_fair_variance,
    integrated_variance_moments,
    realized_variance_option_value,
)
from pricebook.engine.heston_fourier import FourierGrid, heston_call_prices
from pricebook.engine.variance_replication import generalized_fair_variance
from pricebook.engine.volatility_swap import integrated_variance_transform
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import (
    DividendSchedule,
    EquityAsset,
    EquityUnderlying,
    dividend_forward,
)
from pricebook.market.equity_skew import ImpliedVolSkew
from pricebook.models.black import black_implied_vol
from pricebook.models.variance_replication_model import StrikeGrid, VarianceStrip
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.realized_variance_option import RealizedVarianceOption

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
RATE = 0.03
CURVE = DiscountCurve.flat(VAL, RATE, DC)
#: A NON-UNIT horizon, so a dropped `/T` is not arithmetically invisible (T31's recorded trap).
EXPIRY = VAL + timedelta(days=621)
YEARS = 621.0 / 365.0
#: Genuine vol-of-vol and a real correlation: `sigma_v = 0` is the reduction, and the whole normal
#: approximation is about the dispersion `sigma_v` creates.
PARAMS = HestonParams(kappa=1.8, theta=0.06, sigma_v=0.55, rho=-0.7, v0=0.09)
N = Money(1_000_000.0, EUR)
ACME = EquityAsset("ACME")


def _quadrature_variance(params: HestonParams, years: float, nodes: int = 2000) -> float:
    """`Var[∫V dt] = 2∫₀^T ∫₀^t e^{−κ(t−s)} Var(V_s) ds dt`, integrated numerically.

    Independent of the closed form in the way §4 demands: it integrates the **covariance structure**
    the source prints, not the result of having integrated it."""
    kappa, theta, sigma_v, v0 = params.kappa, params.theta, params.sigma_v, params.v0

    def variance_of_v(s: float) -> float:
        decay = math.exp(-kappa * s)
        return (sigma_v * sigma_v / kappa) * (
            v0 * (decay - decay * decay) + 0.5 * theta * (1.0 - decay) ** 2
        )

    step = years / nodes
    outer = 0.0
    for i in range(nodes + 1):
        t = i * step
        inner_nodes = max(i, 1)
        inner_step = t / inner_nodes
        inner = 0.0
        for j in range(inner_nodes + 1):
            s = j * inner_step
            weight = 1.0 if j in (0, inner_nodes) else 2.0
            inner += weight * math.exp(-kappa * (t - s)) * variance_of_v(s)
        inner *= inner_step / 2.0
        outer += (1.0 if i in (0, nodes) else 2.0) * inner
    return 2.0 * outer * step / 2.0


# ============================================================================================
# 1. The two moments: one transcribed, one derived and asserted as a theorem.
# ============================================================================================

CASES = (
    (HestonParams(1.8, 0.06, 0.55, -0.7, 0.09), 1.7),
    (HestonParams(0.9, 0.04, 0.30, -0.5, 0.02), 3.0),
    (HestonParams(3.5, 0.09, 0.80, 0.2, 0.09), 0.5),
)


@pytest.mark.parametrize("params,years", CASES)
def test_the_FIRST_moment_matches_the_printed_form(params: HestonParams, years: float):
    # KWOK-ZHENG §4.1 at the full horizon: `theta*T + (V0 - theta)(1 - e^{-kappa*T})/kappa`.
    want = params.theta * years + (params.v0 - params.theta) * (
        1.0 - math.exp(-params.kappa * years)
    ) / params.kappa
    mean, _ = integrated_variance_moments(params, years)
    assert mean == pytest.approx(want, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("params,years", CASES)
def test_the_DERIVED_variance_matches_an_independent_QUADRATURE_of_the_covariance(
    params: HestonParams, years: float
):
    _, variance = integrated_variance_moments(params, years)
    assert variance == pytest.approx(_quadrature_variance(params, years), rel=1e-6, abs=0.0)


@pytest.mark.parametrize("params,years", CASES)
def test_both_moments_match_TRANCHE_48s_TRANSFORM_which_shares_no_line_with_them(
    params: HestonParams, years: float
):
    """The second independent check, and the one that makes these theorems rather than algebra.

    `L(ψ) = E[e^{−ψ∫V dt}]` is tranche 48's CIR Laplace transform, built and oracled against the
    source's p.144 for the volatility swap. Its derivatives at zero are the moments:
    `M₁ = −L'(0)`, `M₂ = L''(0)`. The finite differences are **one-sided** (the transform refuses a
    negative `ψ`) and therefore first order, so the band is the FD's own error and is asserted as
    such — a reference is code too, and this one's accuracy is `O(h)`, measured at ~1e-4 relative.
    """
    mean, variance = integrated_variance_moments(params, years)
    step = 1.0e-6
    fd_mean = -(integrated_variance_transform(params, step, years) - 1.0) / step
    assert fd_mean == pytest.approx(mean, rel=2e-5, abs=0.0)

    coarse = 1.0e-4
    second = (
        integrated_variance_transform(params, 2.0 * coarse, years)
        - 2.0 * integrated_variance_transform(params, coarse, years)
        + 1.0
    ) / (coarse * coarse)
    assert second - mean * mean == pytest.approx(variance, rel=2e-3, abs=0.0)


@pytest.mark.parametrize("params,years", CASES)
def test_the_moments_are_ORDERED_and_positive_for_all_parameters(
    params: HestonParams, years: float
):
    mean, variance = integrated_variance_moments(params, years)
    assert mean > 0.0 and variance > 0.0


def test_a_ZERO_vol_of_vol_leaves_the_mean_and_KILLS_the_variance():
    # The reduction: with no vol-of-vol the variance path is deterministic, so integrated variance
    # has no dispersion at all — exactly, not approximately.
    deterministic = HestonParams(1.8, 0.06, 0.0, -0.7, 0.09)
    mean, variance = integrated_variance_moments(deterministic, YEARS)
    stochastic_mean, stochastic_variance = integrated_variance_moments(PARAMS, YEARS)
    assert variance == 0.0
    assert mean == stochastic_mean                      # the mean does not depend on sigma_v
    assert stochastic_variance > 0.0                    # the companion: sigma_v is non-degenerate


# ============================================================================================
# 2. The fair variance, and the cross-METHOD anchor for tranche 74.
# ============================================================================================

def test_the_HESTON_fair_variance_is_the_ANNUALISED_first_moment():
    want, _ = integrated_variance_moments(PARAMS, YEARS)
    assert heston_fair_variance(PARAMS, YEARS) == pytest.approx(want / YEARS, rel=1e-15, abs=0.0)


def test_the_fair_variance_reduces_to_v0_as_the_horizon_VANISHES():
    # Over no time at all the only variance is the spot one. A limit, approached: the mean-reversion
    # has not had time to act.
    for short in (1e-4, 1e-6, 1e-8):
        assert heston_fair_variance(PARAMS, short) == pytest.approx(PARAMS.v0, rel=1e-3, abs=0.0)


def test_the_fair_variance_tends_to_THETA_over_a_long_horizon():
    assert heston_fair_variance(PARAMS, 200.0) == pytest.approx(PARAMS.theta, rel=1e-2, abs=0.0)
    # and the premise that this is not vacuous: v0 and theta differ materially
    assert abs(PARAMS.v0 / PARAMS.theta - 1.0) > 0.4


# ============================================================================================
# 3. The option on realized variance — the normal approximation.
# ============================================================================================

def test_the_option_is_the_NORMAL_positive_part_of_realized_variance():
    # The approximation is the normal match; the two moments are exact. Transcribed as the Bachelier
    # positive part of `RV - K` with the annualised moments — `normal_positive_part` reused (T71).
    from pricebook.foundation.distributions import normal_positive_part

    strike = 0.05
    mean, variance = integrated_variance_moments(PARAMS, YEARS)
    want = normal_positive_part(mean / YEARS - strike, math.sqrt(variance) / YEARS)
    got = realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP)
    assert got == pytest.approx(want, rel=1e-15, abs=0.0)


def test_put_call_PARITY_on_realized_variance():
    for strike in (0.02, 0.06, 0.12):
        call = realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP)
        put = realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.FLOOR)
        assert call - put == pytest.approx(
            heston_fair_variance(PARAMS, YEARS) - strike, rel=1e-13, abs=1e-18
        )


def test_the_option_is_MONOTONE_in_the_variance_strike():
    values = [
        realized_variance_option_value(PARAMS, YEARS, k, CapFloorKind.CAP)
        for k in (0.02, 0.04, 0.06, 0.09, 0.14)
    ]
    assert values == sorted(values, reverse=True), values
    assert values[0] > 2.0 * values[-1]


def test_a_ZERO_vol_of_vol_collapses_the_option_to_its_INTRINSIC():
    deterministic = HestonParams(1.8, 0.06, 0.0, -0.7, 0.09)
    fair = heston_fair_variance(deterministic, YEARS)
    for strike in (0.5 * fair, 2.0 * fair):
        assert realized_variance_option_value(
            deterministic, YEARS, strike, CapFloorKind.CAP
        ) == pytest.approx(max(fair - strike, 0.0), abs=1e-18)
    # the companion: with a real vol-of-vol the option is worth MORE than its intrinsic
    strike = 2.0 * fair
    assert realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP) > 1e-6


# ============================================================================================
# 4. The engine.
# ============================================================================================

def _model(params: HestonParams = PARAMS) -> HestonModel:
    return HestonModel(
        MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT_KEY: CURVE})), params
    )


def test_the_registered_pricer_DISCOUNTS_the_normal_approximation():
    strike = 0.05
    option = RealizedVarianceOption(ACME, EXPIRY, strike, N, CapFloorKind.CAP)
    out = price(option, _model())
    assert isinstance(out, PricingResult), out
    want = (
        N.amount
        * CURVE.df(EXPIRY)
        * realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP)
    )
    assert out.pv.amount == pytest.approx(want, rel=1e-14, abs=0.0)


def test_an_EXPIRED_option_is_PV_zero_invariant_6_and_the_fixture_is_IN_THE_MONEY():
    # The straddle's option-bypass form: an expired OTM option's intrinsic is 0, the same answer
    # PV-0 gives, so the guard would be invisible (§4's named sub-trap, F-8.3's shape).
    strike = 0.5 * heston_fair_variance(PARAMS, YEARS)
    option = RealizedVarianceOption(ACME, EXPIRY, strike, N, CapFloorKind.CAP)
    after = EXPIRY + timedelta(days=1)
    model = HestonModel(
        MarketSnapshot(valuation_date=after, curves=CurveSet({DISCOUNT_KEY: DiscountCurve.flat(after, RATE, DC)})),
        PARAMS,
    )
    out = price(option, model)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == 0.0
    # the premise: the fall-through would be a real number, not a coincidental zero
    assert heston_fair_variance(PARAMS, YEARS) - strike > 0.01


def test_a_MISSING_discount_curve_is_a_PricingFailure():
    bare = HestonModel(MarketSnapshot(valuation_date=VAL, curves=CurveSet({})), PARAMS)
    out = price(RealizedVarianceOption(ACME, EXPIRY, 0.05, N, CapFloorKind.CAP), bare)
    assert isinstance(out, PricingFailure), out


# ============================================================================================
# 5. ORACLE 1 — the cross-METHOD anchor, and the independent route tranche 74 never had.
# ============================================================================================

class _HestonSkew(ImpliedVolSkew):
    """Heston's own implied-volatility skew, presented through tranche 31's skew interface.

    **A subclass rather than a duck-typed stand-in, so this stays type-safe** — `fair_variance` is
    typed against `ImpliedVolSkew`, which is a *log-linear closed form* and cannot represent
    Heston's smile. Overriding `volatility` is what lets T74's strip be reused **literally** instead
    of re-implemented in the test: the strip then forms Black prices at Heston's implied vols, which
    by the definition of implied vol **are** Heston's prices. So the chain is the spec's — Heston
    options → implied vols → T74's strip — with no fitted intermediary to blur the comparison. (A
    log-linear *fit* to Heston's smile was the alternative, and it would have made the oracle a
    measure of the fit's quality rather than of the two methods' agreement.)

    **The vols are precomputed in ONE vectorised inversion, and that is a timing fix, not a
    shortcut.** Called per strike, this file ran **11.43s** against the `slow` gate's 10.0s ceiling
    — ~1800 separate Fourier inversions, one per strike per grid. `heston_call_prices` prices a
    whole strike vector in a single pass (the same `φ(u)` evaluated once rather than per strike —
    tranche 53's optimisation), so the numbers are identical and the cost collapses. Lossless: same
    quadrature, same grid, same strikes, same prices.
    """

    def __init__(self, params: HestonParams, underlying, expiry: date, discount, strikes) -> None:
        super().__init__(atm_volatility=0.2, slope=0.0)
        forward = dividend_forward(underlying, expiry, discount)
        horizon = discount.time_measure.year_fraction(expiry)
        wanted = tuple(sorted(set(strikes)))
        calls = heston_call_prices(
            params, underlying, wanted, expiry,
            (discount, FourierGrid(upper=200.0, count=4001)),
        )
        vols = {}
        for strike, call in zip(wanted, calls):
            undiscounted = call / discount.df(expiry)
            if strike < forward:                       # OTM on each side, by parity
                undiscounted -= forward - strike
                kind = CapFloorKind.FLOOR
            else:
                kind = CapFloorKind.CAP
            vols[strike] = black_implied_vol(
                max(undiscounted, 1e-300), forward, strike, horizon, kind
            )
        object.__setattr__(self, "_vols", vols)

    def volatility(self, strike: float, forward: float, years: float = 0.0) -> float:
        return self._vols[strike]


def test_the_HESTON_fair_variance_equals_TRANCHE_74s_STATIC_REPLICATION():
    """**The cross-method anchor, and the point of putting piece 1 in this tranche.**

    Tranche 74 (and 31) produce the fair variance by *replicating* a skew with a `1/K²` strip of
    vanillas; this tranche produces it as a *moment of the variance process* in closed form. Those
    are different mathematics with different code, so their agreement is the one thing neither
    tranche could establish alone — T74's oracles were all transcription, reduction and requirement.

    **The grid is measured here, not inherited** (the T74 lens): Heston's integrand is not the
    log-linear skew T31 measured its grid against, so the width and count are swept below and the
    band is set from where the replication has converged, not from T31's or T74's choice.
    """
    underlying = EquityUnderlying(100.0, 0.0, DividendSchedule())
    forward = dividend_forward(underlying, EXPIRY, CURVE)
    grid = StrikeGrid(0.15, 4.0, 601)
    skew = _HestonSkew(PARAMS, underlying, EXPIRY, CURVE, grid.strikes(forward))

    replicated = generalized_fair_variance(
        underlying, EXPIRY, CURVE, skew, VarianceStrip.vanilla(grid)
    )
    closed_form = heston_fair_variance(PARAMS, YEARS)
    # Measured: the replication converges to the closed form from below (truncation omits tail
    # value), reaching ~0.5% on this grid. The band is that measurement, not a hope.
    assert replicated == pytest.approx(closed_form, rel=0.01, abs=0.0)
    assert replicated < closed_form, (replicated, closed_form)


def test_the_replication_CONVERGES_to_the_closed_form_as_the_grid_widens():
    """The band above is only meaningful with the convergence beside it (§4/T25): a single band on a
    truncated quadrature could be agreement or could be two errors cancelling. Widening the grid
    must move the replication monotonically **toward** the closed form, from below."""
    underlying = EquityUnderlying(100.0, 0.0, DividendSchedule())
    forward = dividend_forward(underlying, EXPIRY, CURVE)
    closed_form = heston_fair_variance(PARAMS, YEARS)
    grids = [StrikeGrid(lo, up, 401) for lo, up in ((0.40, 1.8), (0.25, 2.6), (0.15, 4.0))]
    every_strike = [k for grid in grids for k in grid.strikes(forward)]
    skew = _HestonSkew(PARAMS, underlying, EXPIRY, CURVE, every_strike)
    errors = [
        abs(
            generalized_fair_variance(
                underlying, EXPIRY, CURVE, skew, VarianceStrip.vanilla(grid)
            )
            / closed_form
            - 1.0
        )
        for grid in grids
    ]
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 0.25 * errors[0]


# ============================================================================================
# 6. ORACLE 3 — the MC companion MEASURES the normal approximation's error.
# ============================================================================================

def _simulate_realized_variance(
    params: HestonParams, years: float, paths: int = 200_000, steps: int = 400
) -> "numpy.ndarray":
    """Annualised realized variance `(1/T)∫₀^T V dt` by full-truncation Euler on the CIR variance.

    **The variance process alone, which is the independent thing** — under Heston the continuous
    realized variance *is* `∫V dt`, so the spot path contributes nothing and simulating it would add
    cost and noise without adding evidence. (`simulate_heston_spots` returns terminal spots only, so
    there was nothing to reuse here; extending it to carry the variance path would be scope creep
    into tranche 33's module for one caller.)

    Full truncation (`max(V,0)` in the diffusion) rather than reflection, matching the tree's own
    Heston simulator: it is the scheme whose bias is measured below, and mixing schemes would make
    the measured bias belong to neither.
    """
    rng = numpy.random.default_rng(20261122)
    step = years / steps
    root = math.sqrt(step)
    variance = numpy.full(paths, params.v0)
    total = numpy.zeros(paths)
    for _ in range(steps):
        total += variance * step
        floored = numpy.maximum(variance, 0.0)
        variance = (
            variance
            + params.kappa * (params.theta - variance) * step
            + params.sigma_v * numpy.sqrt(floored) * rng.standard_normal(paths) * root
        )
    return total / years


@pytest.fixture(scope="module")
def realized_variance():
    """One simulation, shared by every test that reads it.

    Run per test this file took 4.56s locally — and §9 is explicit that a local reading is not the
    measurement, since the CI runners are several times slower and the `slow` ceiling is 10.0s. The
    draws do not depend on the strike, so sharing them is **lossless**: identical scheme, seed and
    paths, and each strike is still compared against its own closed form. Sharing only correlates
    the tests' sampling errors, which no assertion depends on being independent. (T74's twin fix,
    applied before CI had to find it — F-8.9's fifth file was found by CI, not by foresight.)
    """
    return _simulate_realized_variance(PARAMS, YEARS)


def test_the_SIMULATION_reproduces_the_closed_form_MOMENTS_before_it_is_trusted(
    realized_variance,
):
    """The simulation is a reference, and a reference is code too (§4) — so its own accuracy is
    established against the closed-form moments *before* it is used to judge the approximation.

    Measured: the mean agrees to well inside its standard error, and the standard deviation to about
    1%, which is the Euler scheme's discretisation bias at 400 steps and is what the band below
    carries as its second error source."""
    realized = realized_variance
    mean, variance = integrated_variance_moments(PARAMS, YEARS)
    standard_error = float(realized.std(ddof=1)) / math.sqrt(realized.size)
    assert abs(float(realized.mean()) - mean / YEARS) < 4.0 * standard_error
    assert float(realized.std(ddof=1)) == pytest.approx(
        math.sqrt(variance) / YEARS, rel=0.03, abs=0.0
    )


@pytest.mark.parametrize("strike", [0.03, 0.05, 0.08])
def test_the_normal_approximations_ERROR_is_MEASURED_against_the_simulation(
    strike: float, realized_variance
):
    """§4's two-error-source rule: the band is `4·SE + the approximation's own measured error`,
    never one flat percentage standing in for both.

    **The approximation error is the deliverable, not the agreement.** The normal match is the only
    approximation (the moments are exact), so the gap between this simulation and the closed form is
    precisely the cost of assuming realized variance is normal. Measured across these strikes it
    runs to about **8%** of the option value at the money and more in the wings, which is a real
    limitation of §5.2's scheme and the reason §5.3's partially-exact-and-bounded approximation
    exists (recorded in the arc plan as the refinement if this proves too wide).

    **Resolution:** at 200k paths the SE is ~0.5% of the at-the-money value, so a green here rules
    out an implementation defect larger than roughly 2% *on top of* the approximation's own error —
    which is why the pointwise and parity oracles above carry the precision and this one carries the
    independence.
    """
    payoff = numpy.maximum(realized_variance - strike, 0.0)
    simulated = float(payoff.mean())
    standard_error = float(payoff.std(ddof=1)) / math.sqrt(payoff.size)
    approximated = realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP)

    band = 4.0 * standard_error + 0.20 * simulated
    assert abs(approximated - simulated) < band, (approximated, simulated, standard_error)
    # and the premise that this is a MEASUREMENT: the approximation error is resolvable above the SE
    assert abs(approximated - simulated) > standard_error, (
        "the normal approximation's error is below the sampling noise here, so this strike asserts "
        "only that both numbers are noisy — widen the path count or pick a strike where it bites"
    )


def test_a_PUT_on_realized_variance_prices_through_its_OWN_transcription():
    """The `omega` factor's second independent killer.

    Parity (above) was its only one, and parity is a *relation* between the two arms — delete it and
    nothing checks that the put arm is the put. This transcribes the put directly: the normal
    positive part of `K − RV` rather than of `RV − K`. Independent by construction, not by
    parameter, which is F-8.3's distinction: a parameter twin shares the other test's deletion fate.
    """
    from pricebook.foundation.distributions import normal_positive_part

    mean, variance = integrated_variance_moments(PARAMS, YEARS)
    for strike in (0.03, 0.09):
        want = normal_positive_part(strike - mean / YEARS, math.sqrt(variance) / YEARS)
        got = realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.FLOOR)
        assert got == pytest.approx(want, rel=1e-15, abs=0.0)
    # and the premise: at these strikes the two arms are materially different numbers
    low = realized_variance_option_value(PARAMS, YEARS, 0.03, CapFloorKind.CAP)
    assert low > 5.0 * realized_variance_option_value(PARAMS, YEARS, 0.03, CapFloorKind.FLOOR)


def test_an_EXPIRED_in_the_money_PUT_is_PV_zero_too_invariant_6():
    """The invariant-6 guard's second independent killer, by the opposite arm.

    Its sibling uses an expired in-the-money **call** (strike below the fair variance). A put is in
    the money on the other side, so this fixture is non-degenerate exactly where the call's would be
    zero — the construction T73 used to harden F-4.1…F-4.5, applied here in-tranche rather than
    waiting for a re-audit to find the thin oracle.
    """
    strike = 2.0 * heston_fair_variance(PARAMS, YEARS)
    option = RealizedVarianceOption(ACME, EXPIRY, strike, N, CapFloorKind.FLOOR)
    after = EXPIRY + timedelta(days=1)
    model = HestonModel(
        MarketSnapshot(
            valuation_date=after,
            curves=CurveSet({DISCOUNT_KEY: DiscountCurve.flat(after, RATE, DC)}),
        ),
        PARAMS,
    )
    out = price(option, model)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == 0.0
    # the premise: the fall-through would be a real number, not a coincidental zero
    assert strike - heston_fair_variance(PARAMS, YEARS) > 0.01
