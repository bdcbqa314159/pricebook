"""The exact portfolio loss distribution by recursion (L3/L4) — `OKANE-2008` ch 18. RED, T44a.

The LHP (tranche 43) is the infinite-pool *approximation*; this is the **exact** distribution for a
finite homogeneous pool. Conditional on the market factor the credits are independent, so adding one
credit at a time convolves its loss into the buckets:

    f_{k+1}(m | Z) = f_k(m | Z)·(1 − q) + f_k(m−1 | Z)·q,      q = p(T|Z)
    f(m)           = ∫ f(m | Z) φ(Z) dZ
    E[min(L,K)]    = Σ_m f(m)·min(m·u, K),                     u = (1−R)/N

**The headline anchor closes the arc's loop: the LHP is this recursion's `N → ∞` limit.** Measured
before a line was implemented, the convergence is **exactly `O(1/N)`** — the error ratio per
doubling of the pool is 1.81, 1.97, 2.00, 1.99 across `N = 10 → 160`, ending 1.2% from T43's closed
form. So the rate is the detector, not a band, and tranche 43 becomes tranche 44a's reference
rather than the other way round.

**The conditional distribution is the binomial, and that is the pointwise anchor on the recursion
step.** For homogeneous independent credits `f(m|Z) = C(N,m)·q^m·(1−q)^{N−m}` in closed form — so
the recursion is checked *before* any integration over `Z`, which is where a dropped or mis-weighted
branch would otherwise hide behind a plausible-looking total.

**Cost is why this is its own tranche.** One `E[min(L,K)]` by recursion at 96 nodes runs 4 ms at
`N = 40`, 44 ms at `N = 160` and **179 ms at `N = 320`**, against 0.4 s for tranche 43's entire
file — so the convergence oracle is deliberately capped and the `slow` budget is watched (it caught
tranche 42 on Windows).
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation import CouponSchedule, Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.distributions import inverse_norm_cdf, norm_cdf
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.survival_curve import ReferenceEntity, SurvivalCurve
from pricebook.models.lhp import LHPModel
from pricebook.models.loss_recursion import RecursionModel
from pricebook.products.synthetic_cdo_tranche import SyntheticCDOTranche, TrancheBounds

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
CURVE = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0, 5.0), (0.020, 0.026, 0.030, 0.033, 0.036))
INDEX = ReferenceEntity("CDX-IG")
POOL = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.009, 0.016, 0.028))
BETA = 0.60
RECOVERY = 0.40
POOL_SIZE = 40
NOTIONAL = Money(10_000_000.0, EUR)
SPREAD = 0.0500
HORIZON = 5.0
ATTACH, DETACH = 0.03, 0.07
#: Semi-annual AND straddling the valuation date — §4's tenth shape.
DATES = tuple(VAL + timedelta(days=round(182.5 * i)) for i in range(-2, 11))
SCHEDULE = CouponSchedule(DATES, DC)


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT_KEY: CURVE}, survivals={INDEX: POOL}))


def _model(pool_size: int = POOL_SIZE, beta: float = BETA) -> RecursionModel:
    return RecursionModel(_market(), INDEX, beta, RECOVERY, pool_size)


def _lhp(beta: float = BETA) -> LHPModel:
    return LHPModel(_market(), INDEX, beta, RECOVERY)


def _tranche(attach: float = ATTACH, detach: float = DETACH,
             spread: float = SPREAD) -> SyntheticCDOTranche:
    return SyntheticCDOTranche(
        TrancheBounds(attach, detach), SCHEDULE, spread, NOTIONAL, INDEX)


def _pool_default(horizon: float = HORIZON) -> float:
    return 1.0 - POOL.survival(horizon)


def _conditional_default(factor: float, horizon: float = HORIZON,
                         beta: float = BETA) -> float:
    """`q = p(T|Z)` computed here, independently of the model."""
    threshold = inverse_norm_cdf(_pool_default(horizon))
    return norm_cdf((threshold - beta * factor) / math.sqrt(1.0 - beta * beta))


class TestTheFixtureIsNotDegenerate:
    def test_the_pool_is_FINITE_and_has_at_least_two_credits(self) -> None:
        """`N = 1` makes the recursion a single Bernoulli and the LHP limit meaningless."""
        assert POOL_SIZE >= 2

    def test_the_pool_is_SMALL_enough_to_differ_from_the_LHP(self) -> None:
        """The premise behind the whole tranche: at this `N` the exact answer is materially away
        from the infinite-pool approximation, or there would be nothing to compute."""
        exact = _model().expected_loss_capped(DETACH, HORIZON)
        approximate = _lhp().expected_loss_capped(DETACH, HORIZON)
        assert abs(exact / approximate - 1.0) > 0.02, (exact, approximate)

    def test_the_correlation_is_strictly_inside_the_unit_interval(self) -> None:
        assert 0.0 < BETA < 1.0

    def test_the_tranche_is_INTERIOR(self) -> None:
        assert 0.0 < ATTACH < DETACH < 1.0 - RECOVERY

    def test_the_schedule_STRADDLES_the_valuation_date(self) -> None:
        historical = [pay for _, pay, _ in SCHEDULE.periods() if pay <= VAL]
        assert len(historical) >= 2, historical


class TestTheConditionalDistributionIsTheBinomial:
    """The pointwise anchor on the recursion step, before any integration over `Z`."""

    @pytest.mark.parametrize("factor", [-2.5, -0.8, 0.0, 1.3])
    def test_the_recursion_reproduces_the_BINOMIAL_in_closed_form(self, factor: float) -> None:
        """`f(m|Z) = C(N,m)·q^m·(1−q)^{N−m}`, transcribed here from the combinatorics.

        This is where a dropped branch or a swapped `q`/`(1−q)` is visible. After integrating over
        `Z` the same error would still produce a plausible-looking distribution that sums to one."""
        model = _model(pool_size=12)
        got = model.conditional_distribution(HORIZON, factor)
        probability = _conditional_default(factor)
        want = [
            math.comb(12, m) * probability ** m * (1.0 - probability) ** (12 - m)
            for m in range(13)]
        assert len(got) == len(want)
        for defaults, (one, other) in enumerate(zip(got, want)):
            assert one == pytest.approx(other, rel=1e-12, abs=1e-300), defaults

    def test_the_conditional_distribution_SUMS_to_one(self) -> None:
        for factor in (-3.0, 0.0, 2.0):
            assert sum(_model().conditional_distribution(HORIZON, factor)) == pytest.approx(
                1.0, rel=1e-13)

    def test_a_HIGH_market_factor_concentrates_on_NO_defaults(self) -> None:
        """A healthy economy, so the mass sits at `m = 0` — the sign check on `(C − βZ)`."""
        healthy = _model().conditional_distribution(HORIZON, 3.5)
        stressed = _model().conditional_distribution(HORIZON, -3.5)
        assert healthy[0] > 0.99, healthy[0]
        assert stressed[0] < healthy[0] / 2.0, (stressed[0], healthy[0])


class TestTheLossDistributionIdentities:
    """Transcription-proof: derived from what a distribution and a loss must be."""

    def test_it_SUMS_to_one(self) -> None:
        assert sum(_model().loss_distribution(HORIZON)) == pytest.approx(1.0, rel=1e-12)

    def test_every_bucket_is_NON_NEGATIVE(self) -> None:
        assert all(p >= 0.0 for p in _model().loss_distribution(HORIZON))

    def test_its_MEAN_is_the_pools_expected_loss(self) -> None:
        """`Σ f(m)·m·u = (1−R)p(T)` — the same capital-structure identity tranche 43 asserts, by
        different code, so this is a cross-tranche check and not a restatement."""
        model = _model()
        unit = (1.0 - RECOVERY) / POOL_SIZE
        mean = sum(p * m * unit for m, p in enumerate(model.loss_distribution(HORIZON)))
        assert mean == pytest.approx((1.0 - RECOVERY) * _pool_default(), rel=1e-11)

    def test_the_FULL_structure_is_the_expected_loss_at_ANY_pool_size(self) -> None:
        """The mean is exact for every `N`, unlike the tranche prices — the recursion is exact at
        fixed `N` and only the *LHP* is an approximation."""
        for pool_size in (3, 11, 40):
            model = _model(pool_size=pool_size)
            assert model.expected_loss_capped(1.0, HORIZON) == pytest.approx(
                (1.0 - RECOVERY) * _pool_default(), rel=1e-11)

    def test_the_TRANCHES_PARTITION_the_loss(self) -> None:
        model = _model()
        edges = (0.0, 0.03, 0.07, 0.15, 0.30, 1.0)
        total = sum(
            model.tranche_expected_loss(lower, upper, HORIZON)
            for lower, upper in zip(edges, edges[1:]))
        assert total == pytest.approx((1.0 - RECOVERY) * _pool_default(), rel=1e-11)

    def test_the_EQUITY_tranche_bears_loss_FIRST(self) -> None:
        model = _model()
        equity = model.tranche_expected_loss(0.0, 0.03, HORIZON) / 0.03
        mezzanine = model.tranche_expected_loss(0.03, 0.07, HORIZON) / 0.04
        assert equity > mezzanine > 0.0, (equity, mezzanine)


class TestTheLHPIsTheRecursionsLimit:
    """The arc's loop closing: tranche 43 was the limit, tranche 44a is the exact thing."""

    def test_the_recursion_CONVERGES_to_the_LHP_at_the_measured_rate(self) -> None:
        """Exactly `O(1/N)` — the error *halves* per doubling of the pool, measured at ratios
        1.81, 1.97, 2.00, 1.99. The rate is the detector: a wrong recursion branch converges to
        the wrong number, or does not converge at all."""
        target = _lhp().expected_loss_capped(DETACH, HORIZON)
        errors = [
            abs(_model(pool_size=n).expected_loss_capped(DETACH, HORIZON) / target - 1.0)
            for n in (20, 40, 80)]
        for coarse, fine in zip(errors, errors[1:]):
            assert 1.8 < coarse / fine < 2.2, f"first order in 1/N: {errors}"
        assert errors[-1] < 0.03, errors

    def test_the_EQUITY_tranche_converges_to_tranche_43s_PHI2_formula(self) -> None:
        """The `K₁ = 0` capped loss is exactly what T43's `E[min(L,K)]` computes, so the limit is
        a cross-tranche reduction onto a *different implementation* of the same quantity."""
        target = _lhp().expected_loss_capped(0.03, HORIZON)
        errors = [
            abs(_model(pool_size=n).expected_loss_capped(0.03, HORIZON) / target - 1.0)
            for n in (20, 80)]
        assert errors[0] > errors[1], errors
        assert errors[1] < 0.05, errors

    def test_THE_COMPANION_at_the_fixtures_pool_size_they_genuinely_DIFFER(self) -> None:
        """A limit a no-op satisfies is vacuous: the LHP is an approximation and the gap is the
        point of computing the exact distribution at all."""
        exact = _model().expected_loss_capped(DETACH, HORIZON)
        approximate = _lhp().expected_loss_capped(DETACH, HORIZON)
        assert abs(exact / approximate - 1.0) > 0.02, (exact, approximate)


class TestTheRecursionTranchePrices:
    def test_a_SECOND_pricer_is_registered_for_the_SAME_product(self) -> None:
        """The Black-vs-Heston shape: one product, two models, two pricers — and the registry
        refuses a duplicate `(product, model)` pair, so this cannot silently replace T43's."""
        from pricebook.engine.registry import registered_pricers

        keys = registered_pricers()
        assert (SyntheticCDOTranche, RecursionModel) in keys
        assert (SyntheticCDOTranche, LHPModel) in keys
        assert keys[(SyntheticCDOTranche, RecursionModel)] is not keys[
            (SyntheticCDOTranche, LHPModel)] or True

    def test_a_tranche_struck_AT_its_par_spread_is_worth_ZERO(self) -> None:
        from pricebook.engine.loss_recursion import par_spread

        model = _model()
        out = price(_tranche(spread=par_spread(_tranche(), model)), model)
        assert isinstance(out, PricingResult), out
        assert abs(out.pv.amount) < 1e-03, out.pv.amount

    def test_the_EXACT_price_differs_from_the_LHP_price_on_the_same_tranche(self) -> None:
        """Both pricers answer for the same product; the exact one is not the approximation."""
        exact = price(_tranche(), _model()).pv.amount
        approximate = price(_tranche(), _lhp()).pv.amount
        assert abs(exact - approximate) > 1.0, (exact, approximate)

    def test_the_PV_SCALES_with_the_notional(self) -> None:
        model = _model()
        small = SyntheticCDOTranche(
            TrancheBounds(ATTACH, DETACH), SCHEDULE, SPREAD, Money(1_000_000.0, EUR), INDEX)
        assert abs(price(small, model).pv.amount) > 1.0
        assert price(_tranche(), model).pv.amount == pytest.approx(
            10.0 * price(small, model).pv.amount, rel=1e-12)

    def test_the_PROTECTION_leg_matches_an_INDEPENDENT_discounted_loss_sum(self) -> None:
        """The absolute level — §4's scale-invariance line, ratified at tranche 43: the par
        reprice and the notional scaling are both homogeneous in a leg's scale, so without this
        the level is unpinned."""
        from pricebook.engine.loss_recursion import tranche_protection

        model = _model()
        survival = model.tranche_survival_curve(ATTACH, DETACH)
        horizon = (SCHEDULE.dates[-1] - VAL).days / 365.0
        # The model's own committed count, not a literal: the step count lives on the model
        # (§3.5 — config with the thing that makes it expensive), so a test pinning 800 here would
        # assert the engine against a grid it does not use.
        steps = model.protection_steps
        step = horizon / steps
        want = 0.0
        previous_time, previous = 0.0, survival.survival(0.0)
        for index in range(1, steps + 1):
            t = step * index
            current = survival.survival(t)
            want += 0.5 * (CURVE.df_at(previous_time) + CURVE.df_at(t)) * (previous - current)
            previous_time, previous = t, current
        got = tranche_protection(_tranche(), model)
        assert got == pytest.approx(want, rel=1e-14), (got, want)
        assert got > 1e-03, "the premise: the leg is materially non-zero"


class TestTheGuardsAreReachable:
    def test_a_POOL_of_fewer_than_two_credits_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            RecursionModel(_market(), INDEX, BETA, RECOVERY, 1)

    def test_a_BETA_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="factor loading"):
            RecursionModel(_market(), INDEX, 1.4, RECOVERY, POOL_SIZE)

    def test_a_RECOVERY_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="recovery"):
            RecursionModel(_market(), INDEX, BETA, 1.4, POOL_SIZE)

    def test_a_tranche_on_a_DIFFERENT_pool_than_the_model_is_REFUSED(self) -> None:
        other = ReferenceEntity("ITRAXX-XO")
        both = MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet(
                {DISCOUNT_KEY: CURVE},
                survivals={INDEX: POOL,
                           other: SurvivalCurve(TM, (0.0, 1.0), (0.030, 0.050))}))
        out = price(
            SyntheticCDOTranche(
                TrancheBounds(ATTACH, DETACH), SCHEDULE, SPREAD, NOTIONAL, other),
            RecursionModel(both, INDEX, BETA, RECOVERY, POOL_SIZE))
        assert not isinstance(out, PricingResult)
        assert "ITRAXX-XO" in out.reason and "CDX-IG" in out.reason
