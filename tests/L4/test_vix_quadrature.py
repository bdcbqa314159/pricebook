"""The VIX by-parts quadrature's own structure (L4). Tranche 76.

**Split from `test_vix.py` along a real subject seam, not to fit a clock.** That file asks *is VIX²
the variance-swap rate, and do the products price* — questions about the finance. This one asks *is
the integral computed correctly* — questions about the quadrature: whether its span is a margin or a
load-bearing number, whether its below-mass analytic branch is reached at all, and whether the
boundary term survives a price band. The two sets share a module docstring's worth of context and
nothing else, and the concentrated law these tests need costs ~1.4s a call (noncentrality ~360, ~100
mixture terms per node), which is why keeping them beside the finance put one file at 6.3s locally
against a 10.0s ceiling that CI measures three times slower. §9 names splitting along a real seam as
the lossless move before an exemption; this is that seam.

The mathematics is in `engine/vix.py`'s docstring; the identity and the products are in
`test_vix.py`.
"""

# slow-test-exempt: the concentrated-law integrals cost ~1.4s each and reaching the quadrature's
# below-mass analytic branch REQUIRES a high noncentrality, whose mixture is O(sqrt(nc)) per node.
# Five lossless passes took the VIX tests from 8.99s to 2.97s + 3.60s locally; Windows CI still read
# 11.5s and 11.1s (a 4x factor, not the 3.3x projected). Trading the node count or the strike counts
# would be the oracle-trimming §9 forbids, and the one remaining optimisation measured SLOWER. Ledgered
# as OPEN.md OI-8 with its removal trigger.


import math
from datetime import date, timedelta

import numpy
import pytest

from pricebook.engine import price
from pricebook.engine.realized_variance_option import heston_fair_variance
from pricebook.engine.vix import (
    _terminal_variance_law,
    quadrature_span,
    vix_affine_coefficients,
    vix_future_value,
    vix_option_value,
    vix_squared,
)
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.noncentral_chisquare import noncentral_chisquare_sf
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.vix import VixFuture, VixOption

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
RATE = 0.03
CURVE = DiscountCurve.flat(VAL, RATE, DC)
#: The index's own window: 30 calendar days (§1.5.2).
WINDOW = 30.0 / 365.0
#: A NON-UNIT maturity, so a dropped horizon is not arithmetically invisible.
MATURITY = VAL + timedelta(days=274)
MATURITY_YEARS = 274.0 / 365.0
#: **`v0 != theta` is load-bearing**, not decoration: with the spot variance already at the long-run
#: level, `[0,T]` and `[T,T+tau]` carry the *same* expected variance, so a mutation that drops the
#: forward offset and reads the spot window survives — §4's translation-invariance trap (T17), here
#: in the time dimension. Mean-reverting away from `theta` is what separates the two windows.
PARAMS = HestonParams(kappa=1.8, theta=0.06, sigma_v=0.55, rho=-0.7, v0=0.09)
N = Money(1_000.0, EUR)


def _simulate_terminal_variance(
    params: HestonParams, years: float, paths: int = 150_000, steps: int = 600
) -> "numpy.ndarray":
    """`V_T` by full-truncation Euler — the independent law the closed forms are checked against.

    **The path and step counts are measured, not chosen round.** Against the closed-form future:
    300k x 1200 costs 1.76s for a bias of −0.011% and `SE/mean = 0.0010`; **150k x 600 costs 0.45s**
    for a bias of **+0.073%** and `SE/mean = 0.0014`. The band below is `4*SE + 0.5%`, so the bias
    sits seven times inside its own term and the standard error stays under the **0.002** resolution
    premise this file asserts. Nothing was traded for the clock — a configuration whose bias broke
    the band or whose SE broke the premise would have been the trimming §9 forbids; this one breaks
    neither, and the numbers are here so the next reader can check that rather than trust it."""
    rng = numpy.random.default_rng(20261123)
    step = years / steps
    root = math.sqrt(step)
    variance = numpy.full(paths, params.v0)
    for _ in range(steps):
        floored = numpy.maximum(variance, 0.0)
        variance = (
            variance
            + params.kappa * (params.theta - variance) * step
            + params.sigma_v * numpy.sqrt(floored) * rng.standard_normal(paths) * root
        )
    return numpy.maximum(variance, 0.0)


@pytest.fixture(scope="module")
def terminal_variance():
    """One simulation, shared — the draws do not depend on the strike. Lossless, and the `slow`
    gate's lesson applied before CI has to find it (F-8.9, T74/T75)."""
    return _simulate_terminal_variance(PARAMS, MATURITY_YEARS)


# ============================================================================================
# 6. The quadrature's own structure — the branches and margins a price oracle cannot see.
# ============================================================================================

#: A vol-of-vol low enough that `V_T`'s mass sits **far from zero relative to its spread**, so the
#: quadrature's lower limit lands strictly above zero and its analytic below-mass piece is actually
#: executed. Under the main fixture (`sigma_v = 0.55`) the mass straddles zero within 12 spreads, so
#: `start == lower` always and that branch is **unreachable** — a mutation dropping its lower limit
#: survived the whole suite, §4's third survivor kind (not a hole, not an equivalent mutant: dead
#: code in the fixture's regime). `stdev/mean` here is ~0.05, inside the 1/12 the span needs.
def _model(params: HestonParams = None) -> HestonModel:
    return HestonModel(
        MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT_KEY: CURVE})),
        params if params is not None else PARAMS,
    )


CONCENTRATED = HestonParams(kappa=1.8, theta=0.06, sigma_v=0.025, rho=-0.7, v0=0.09)


@pytest.fixture(scope="module")
def concentrated_future():
    """`E₀[VIX_T]` under the concentrated law, computed once and shared.

    At a noncentrality of ~360 each survival evaluation costs ~100 mixture terms, so one integral is
    ~1.4s and three tests each computing their own put this file at **8.99s** — the `slow` gate's
    lesson a third time in three tranches. The value does not depend on which test reads it, so
    sharing is lossless."""
    return vix_future_value(CONCENTRATED, MATURITY_YEARS, WINDOW)


def test_the_below_mass_ANALYTIC_piece_is_reached_and_correct_at_a_concentrated_law(
    concentrated_future,
):
    """The unreachable-branch fix, with its premise asserted rather than hoped for.

    Where the distribution is concentrated the quadrature spans only the mass and integrates the
    stretch beneath it in closed form (`∫ a/2√(ax+b) dx = √(ax+b)`, since the survival is 1 there).
    This fixture reaches that branch; the main one never does. The check is the requirement that
    makes the split sound — the future must still sit strictly between its own lower bound `√b` and
    the Jensen ceiling — plus agreement with a direct simulation, which knows nothing of the split.
    """
    a, b = vix_affine_coefficients(CONCENTRATED, WINDOW)
    scale = CONCENTRATED.sigma_v ** 2 * (1.0 - math.exp(-CONCENTRATED.kappa * MATURITY_YEARS)) / (
        4.0 * CONCENTRATED.kappa
    )
    dof = 4.0 * CONCENTRATED.kappa * CONCENTRATED.theta / CONCENTRATED.sigma_v ** 2
    noncentrality = (
        4.0 * CONCENTRATED.kappa * math.exp(-CONCENTRATED.kappa * MATURITY_YEARS)
        * CONCENTRATED.v0
        / (CONCENTRATED.sigma_v ** 2 * (1.0 - math.exp(-CONCENTRATED.kappa * MATURITY_YEARS)))
    )
    mean = scale * (dof + noncentrality)
    spread = scale * math.sqrt(2.0 * (dof + 2.0 * noncentrality))
    assert mean - 12.0 * spread > 0.0, "the premise: the span must clear zero, or the branch is dead"

    future = concentrated_future
    assert math.sqrt(b) < future < math.sqrt(a * mean + b)

    simulated = numpy.sqrt(
        a * _simulate_terminal_variance(CONCENTRATED, MATURITY_YEARS, paths=60_000, steps=400) + b
    )
    standard_error = float(simulated.std(ddof=1)) / math.sqrt(simulated.size)
    assert abs(future - float(simulated.mean())) < 4.0 * standard_error + 0.005 * future


def test_the_FUTURE_exceeds_its_own_sqrt_b_FLOOR_which_no_price_band_notices():
    """The `√b` boundary term's second independent killer.

    `E[VIX_T] = √b + ∫(a/2√(ax+b))·S dx` and the integral is strictly positive, so the future has
    `√b` as a hard floor — a requirement, true for every parameter set, that the simulation band
    cannot see because `√b` is a small part of the value. Dropping the term left the price inside
    every band; it cannot survive the floor.
    """
    # Two cheap parameter sets rather than the concentrated law: the floor is a requirement true for
    # EVERY parameter set, so it needs no expensive fixture — and the concentrated law costs ~1.4s a
    # call (noncentrality ~360, ~100 mixture terms per node) on a file the `slow` gate watches.
    for params in (PARAMS, HestonParams(0.9, 0.04, 0.35, -0.4, 0.03)):
        _, b = vix_affine_coefficients(params, WINDOW)
        future = vix_future_value(params, MATURITY_YEARS, WINDOW)
        assert future > math.sqrt(b)
        # the premise: `sqrt(b)` is a MATERIAL share of the value, so the floor discriminates
        assert math.sqrt(b) > 0.1 * future


def test_the_quadrature_SPAN_is_a_MARGIN_and_is_coupled_to_the_node_count():
    """The span is a **margin**, not a formula — and measuring it turned up a coupling worth stating.

    A mutation narrowing the span (`dof + 2·nc` → `dof + nc` inside the spread) survived, and for a
    margin that is the right outcome: its job is to be *wide enough*, and a narrower one that still
    covers the mass is not a defect. But "wider is safer" is **false here**, which is the part worth
    recording: the node count is fixed, so widening the span **thins the nodes**, and past the mass
    the answer drifts *away*. Measured, with 2000 nodes throughout: 0.2479 at 0.25 spreads (8.1%
    wrong — the mass is cut off), 0.22827 at 3, then flat to about **1e-3** from 6 through 30, and
    drifting again by 9e-4 at 60 as the step coarsens. Holding the *step size* constant instead —
    scaling the nodes with the span — the value converges properly: 0.2293664 at 6 spreads/1000
    nodes, 0.2294107 at 12/2000, 0.2294116 at 24/4000, so the shipped pair is converged to **4e-6**.

    So two things are asserted, and the second is the one a single band would have hidden: the value
    is stable across the plateau at the tolerance the measurement supports, **and** a span too
    narrow to cover the mass moves it by orders more — so the stability claim is not vacuous.
    """
    import pricebook.engine.vix as engine

    original = engine._SPREAD_SPANS
    try:
        values = []
        for spans in (6.0, 9.0, 12.0, 18.0, 30.0):
            engine._SPREAD_SPANS = spans
            values.append(vix_future_value(PARAMS, MATURITY_YEARS, WINDOW))
        shipped = values[2]
        assert all(abs(v / shipped - 1.0) < 2e-3 for v in values), values

        engine._SPREAD_SPANS = 0.25
        cut_off = vix_future_value(PARAMS, MATURITY_YEARS, WINDOW)
    finally:
        engine._SPREAD_SPANS = original
    assert abs(cut_off / shipped - 1.0) > 0.05, (cut_off, shipped)


#: A HIGH-noncentrality fixture, every parameter load-bearing for the regime: `v0` far above `theta`
#: (the initial condition dominates), low mean reversion and a short horizon (it has not decayed).
#: `nc ~ 1100` against `dof ~ 0.6`. **The value is measured, not chosen for roundness** — see the
#: test below: the span centre's defect costs 1.1e-04 at `nc = 195` and 8.3e-04 at `nc = 438`, inside
#: any sane band, and only reaches **14.9% at nc ~ 1100**. The arc's other VIX fixtures reach `nc = 2…34`.
HIGH_NC = HestonParams(v0=0.40, kappa=0.2, theta=0.01, sigma_v=0.12, rho=-0.5)
HIGH_NC_YEARS = 0.1


def test_the_span_CENTRE_is_a_FORMULA_where_its_width_is_a_margin():
    """**Re-audit #9's F-9.5.** The span's *centre* is a formula; its *width* is a margin.

    The margin test above records that narrowing the **width** survived mutation and rules that
    correct — a margin's job is to be wide enough. The **centre** is a different kind of quantity,
    and the two were indistinguishable from outside because the only observable was a price in which
    a mis-centred grid stays invisible until the noncentrality is large.

    **Measured, by applying the real mutation across the regime** (`mean = scale·(dof+nc)` →
    `scale·dof`): 1.08e-04 at `nc = 195`, 8.31e-04 at `nc = 438`, **1.49e-01 at `nc = 1100`**,
    3.7e-01 at 3722, 5.5e-01 at 14328. So the defect is real and the fixtures simply never reached
    it. *(A first attempt put this fixture at `nc = 195` on a coverage argument — a span that fails
    to bracket the mass at 8 sigma — and the mutation **survived**, because at 5 sigma above the
    true mean the surviving tail is worth ~1e-6 of the integral. Coverage was the wrong criterion;
    the integral's own sensitivity is the right one.)*

    **The oracle is on the extracted span, not on a price**, which is §4's reachability rule: a
    buried step gets named surface so a test can reach it. Pinning the centre through the price
    would need an `nc ≈ 1100` integral **pair** (~9s on an already-exempt file); pinning it here is
    free, and it is *stronger* — it asserts the property the formula exists to deliver (the grid
    covers the mass) rather than a number the formula happens to produce.
    """
    scale, dof, noncentrality = _terminal_variance_law(HIGH_NC, HIGH_NC_YEARS)
    # the premise: this is two orders of noncentrality above the rest of the suite's fixtures
    assert noncentrality > 1000.0, noncentrality
    assert noncentrality / dof > 1000.0, (noncentrality, dof)

    start, stop = quadrature_span(HIGH_NC, HIGH_NC_YEARS, 0.0)

    # The requirement, derived from the LAW and not from the span formula: beyond `stop` the
    # survival must be negligible, or the integral is truncated. A centre that drops the
    # noncentrality puts `stop` BELOW the law's own mean, where the survival is ~1 -- not small.
    assert noncentral_chisquare_sf(stop / scale, dof, noncentrality) < 1e-9
    # and `stop` must sit above the mean by many spreads, which the mis-centred version cannot
    mean = scale * (dof + noncentrality)
    assert stop > mean, (stop, mean)


def test_the_high_nc_future_sits_just_UNDER_its_jensen_bound_a_SECOND_killer():
    """The independent second killer for F-9.5, because one was not enough — and §9 says so.

    The span assertion above kills the mis-centred grid, but the sweep flagged it as a
    **single-killer**: one test, whose later deletion would silently reopen the hole. That is
    re-audit #8's **F-8.3** exactly — five previously-shipped holes each closed with a single test,
    §9's second-killer preference never applied. So this is the second, and it is independent by
    construction: it is a **finance requirement on the price**, not a property of the span.

    At `nc ≈ 1100` the terminal variance is concentrated (relative spread **6.0%**), so
    `E[√(aV+b)] ≈ √(aE[V]+b)` — and by **Jensen**, with `√` concave, the true future must sit
    *below* that bound. Measured: the bound is 0.623821, the future 0.623538, **0.045% under it**,
    on the correct side. The mis-centred grid gives 0.530534 — **14.95%** off, failing by 15x.

    Two different things are asserted, and the first is exact: the **direction** (Jensen, which no
    band can fudge) and the **magnitude** (concentration, which is what makes the bound tight enough
    to discriminate). Cost: ~2.3s, on OI-8's already-ledgered exemption — the price of a second
    killer for a closed hole, which §9 prefers for exactly the reason F-8.3 records.
    """
    a, b = vix_affine_coefficients(HIGH_NC, WINDOW)
    scale, dof, noncentrality = _terminal_variance_law(HIGH_NC, HIGH_NC_YEARS)
    expected_variance = scale * (dof + noncentrality)
    bound = math.sqrt(a * expected_variance + b)

    future = vix_future_value(HIGH_NC, HIGH_NC_YEARS, WINDOW)

    # Jensen, exactly: `sqrt` is concave, so E[sqrt(.)] <= sqrt(E[.]). No tolerance needed.
    assert future <= bound, (future, bound)
    # and the law is concentrated enough that the bound is TIGHT, which is what discriminates
    relative_spread = math.sqrt(2.0 * (dof + 2.0 * noncentrality)) / (dof + noncentrality)
    assert relative_spread < 0.08, relative_spread
    assert future == pytest.approx(bound, rel=0.01, abs=0.0), (future, bound)


def test_the_concentrated_law_also_prices_through_the_ENGINE_not_only_the_helper(
    concentrated_future,
):
    """The below-mass analytic branch's second independent killer.

    Its sibling reaches that branch through `vix_future_value`; this reaches it through the
    registered pricer and an **option** rather than a future, so the branch is exercised on a
    different payoff and neither test's deletion leaves it uncovered. The concentrated law is what
    makes the branch reachable at all (see `CONCENTRATED`).
    """
    model = _model(CONCENTRATED)
    level = concentrated_future

    # An option struck just below the (nearly deterministic) level is almost pure intrinsic, and an
    # analytic piece with the wrong lower limit shifts the level it is measured against.
    strike = 0.9 * level
    option = price(VixOption(MATURITY, WINDOW, strike, N, CapFloorKind.CAP), model)
    assert isinstance(option, PricingResult), option
    intrinsic = N.amount * CURVE.df(MATURITY) * (level - strike)
    assert option.pv.amount == pytest.approx(intrinsic, rel=0.02, abs=0.0)
    assert option.pv.amount > intrinsic          # a little time value, since sigma_v > 0
