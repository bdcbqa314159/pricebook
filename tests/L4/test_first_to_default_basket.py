"""The first-to-default basket (L4) — `OKANE-2008` ch 15.1-15.2. Pre-registered RED, tranche 42.

A first-to-default basket is **a CDS on the basket survival curve**, which is what makes the method
exact and the reuse large:

    Q_B(t|Z) = Π_i Q_i(t|Z)        (conditional independence — tranche 41's copula)
    Q_B(t)   = ∫ Q_B(t|Z) φ(Z) dZ  (Gauss-Hermite over the market factor)
    then the premium and protection legs of a CDS on Q_B.

**Three measurements shaped this suite before a line of it was implemented.**

* **The `ρ = 1` limit cannot be reached by quadrature.** At `|β| = 1` each conditional survival is
  an indicator in `Z`, so their product is a step — and Gauss-Hermite cannot integrate a step
  (tranche 41's unit-β finding, now one level up). Measured: `Q_B` came out **6.1% below** `min_i
  Q_i` at `t = 5`, and at `t = 3` it came out **above** it, which is an impossible probability —
  the Fréchet upper bound violated. So the all-unit-β case is taken analytically: the product is
  `1{Z > max_i C_i}`, giving `Q_B = 1 − Φ(max_i C_i) = min_i Q_i` **exactly** (verified for `β = ±1`
  alike, 0 ULP). A *mixed* basket — some names at unit loading, some below — is **refused**, because
  the quadrature is silently wrong there and nothing needs it.
* **`credit_legs.protection_leg` is not valid on `Q_B`.** Its correctness rests on the hazard being
  constant between pillars, which is true of a bootstrapped `SurvivalCurve` and false of `Q_B`.
  Measured: fed `Q_B`, it **plateaus at −1.56e-03** — identical at 50 and 200 points per segment,
  the §4 signature of a model error rather than a discretisation one. So the protection leg here is
  the `−dQ_B` form the spec itself writes, which needs no hazard and converges cleanly
  (−7.3e-06 → −4.5e-07 → −2.7e-08 at 50/200/800 steps).
* **The premium leg *does* reuse cleanly** — `risky_pv01` reads only `survival(t)`.
"""

import math
from collections.abc import Callable
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation import CouponSchedule, Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.distributions import norm_cdf
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.credit_legs import protection_leg_from_survival, risky_pv01
from pricebook.market.survival_curve import ReferenceEntity, SurvivalCurve
from pricebook.models.basket import BasketModel, BasketSurvival
from pricebook.models.gaussian_copula import CreditName, GaussianCopula
from pricebook.products.first_to_default_basket import FirstToDefaultBasket

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
RECOVERY = 0.40
NOTIONAL = Money(10_000_000.0, EUR)
SPREAD = 0.0400
HORIZON = 5.0

CURVE = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0, 5.0), (0.020, 0.026, 0.030, 0.033, 0.036))

#: DISTINCT loadings and DISTINCT, term-structured marginals. Equal names would make the product
#: over names symmetric, so a dropped or repeated factor would be invisible (§4).
ENTITIES = (ReferenceEntity("ACME"), ReferenceEntity("BRAVO"), ReferenceEntity("CIRRUS"))
BETAS = (0.30, 0.60, 0.45)
HAZARDS = (
    (0.0, 1.0, 3.0), (0.010, 0.025, 0.045),
    (0.0, 1.0, 3.0), (0.020, 0.030, 0.050),
    (0.0, 2.0), (0.015, 0.038),
)
#: Semi-annual, so accruals are ~0.5 and unequal — the τ=1 history baked in.
DATES = tuple(VAL + timedelta(days=round(182.5 * i)) for i in range(11))
SCHEDULE = CouponSchedule(DATES, DC)


def _curves() -> tuple[SurvivalCurve, ...]:
    return (
        SurvivalCurve(TM, HAZARDS[0], HAZARDS[1]),
        SurvivalCurve(TM, HAZARDS[2], HAZARDS[3]),
        SurvivalCurve(TM, HAZARDS[4], HAZARDS[5]),
    )


def _copula(betas: tuple[float, ...] = BETAS, count: int | None = None) -> GaussianCopula:
    curves = _curves()
    names = tuple(
        CreditName(entity, beta, curve)
        for entity, beta, curve in zip(ENTITIES, betas, curves))
    return GaussianCopula(
        MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT_KEY: CURVE})),
        names if count is None else names[:count])


def _model(betas: tuple[float, ...] = BETAS) -> BasketModel:
    return BasketModel(_copula(betas), RECOVERY)


def _basket(spread: float = SPREAD) -> FirstToDefaultBasket:
    return FirstToDefaultBasket(ENTITIES, SCHEDULE, spread, NOTIONAL)


def _independent_product(t: float) -> float:
    """`Π_i Q_i(t)` computed here, straight off the hazards."""
    return math.prod(curve.survival(t) for curve in _curves())


def _widest(t: float) -> float:
    """`min_i Q_i(t)` — the Fréchet upper bound on the basket's survival."""
    return min(curve.survival(t) for curve in _curves())


class TestTheFixtureIsNotDegenerate:
    def test_there_are_at_least_TWO_names(self) -> None:
        assert len(ENTITIES) >= 2

    def test_the_BETAS_are_DISTINCT(self) -> None:
        assert len(set(BETAS)) == len(BETAS)

    def test_the_MARGINALS_are_DISTINCT_so_the_product_is_not_symmetric(self) -> None:
        values = [curve.survival(HORIZON) for curve in _curves()]
        assert len(set(round(v, 12) for v in values)) == len(values), values

    def test_every_correlation_is_strictly_inside_the_unit_interval(self) -> None:
        copula = _copula()
        for i in range(len(BETAS)):
            for j in range(len(BETAS)):
                if i != j:
                    assert 0.0 < abs(copula.asset_correlation(i, j)) < 1.0

    def test_the_accruals_are_NON_UNIT_and_UNEQUAL(self) -> None:
        accruals = [a for _, _, a in SCHEDULE.periods()]
        assert all(abs(a - 1.0) > 0.4 for a in accruals)
        assert len(set(round(a, 12) for a in accruals)) > 1


class TestTheBasketSurvivalCurve:
    """The one new quantity; everything else is reuse."""

    def test_it_starts_at_ONE_and_DECREASES(self) -> None:
        basket = BasketSurvival(_copula())
        values = [basket.survival(t / 4.0) for t in range(0, 21)]
        assert values[0] == 1.0
        for earlier, later in zip(values, values[1:]):
            assert later <= earlier, values

    @pytest.mark.parametrize("horizon", [0.5, 1.0, 3.0, 5.0])
    def test_THE_SANDWICH_product_below_and_widest_name_above(self, horizon: float) -> None:
        """`Π_i Q_i ≤ Q_B ≤ min_i Q_i`, for all parameters.

        The lower bound is independence, the upper is "the basket defaults no later than its first
        name" — the Fréchet bounds, so this is both a requirement oracle and the monotonicity in
        `ρ`. It survives a transcription error, and it is what caught the unit-β quadrature
        failure: the step integrand put `Q_B` **above** `min_i Q_i` at `t = 3`, an impossible
        probability that no value-against-value check would have flagged as impossible."""
        got = BasketSurvival(_copula()).survival(horizon)
        assert _independent_product(horizon) <= got <= _widest(horizon) + 1e-15, (
            _independent_product(horizon), got, _widest(horizon))

    def test_the_SINGLE_NAME_basket_IS_that_names_curve(self) -> None:
        """The reduction is exact **in ℝ** and a few ULP in floating point, which is the most that
        is available — and an earlier draft of this test wrongly asserted bit-equality.

        One name makes the product trivial, and `E_Z[1 − p₁(t|Z)] = Q₁(t)` is an exact identity.
        But the two sides are *different computations*: a 64-node Gauss-Hermite sum over `Φ`
        against a single `exp(−∫λ)`. They happened to agree to the last bit on one platform, so
        the first version asserted `==` — and **CI failed on both Linux and Windows at 1 ULP**,
        which is precisely what §9's "tolerance-based oracles are a hard rule" exists to prevent.
        Four ULP is asserted: tight enough that a real defect in the product-over-names cannot
        hide (the independence bound is 4.5% away), loose enough to be a fact rather than luck."""
        solo = BasketSurvival(_copula(count=1))
        curve = _curves()[0]
        for horizon in (1.0, 3.0, 5.0):
            got, want = solo.survival(horizon), curve.survival(horizon)
            assert abs(got - want) <= 4.0 * math.ulp(want), (horizon, got, want)

    @pytest.mark.parametrize("horizon", [1.0, 3.0, 5.0])
    def test_INDEPENDENCE_rho_zero_gives_the_PRODUCT(self, horizon: float) -> None:
        got = BasketSurvival(_copula(betas=(0.0, 0.0, 0.0))).survival(horizon)
        assert got == pytest.approx(_independent_product(horizon), rel=1e-13)

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    @pytest.mark.parametrize("horizon", [1.0, 3.0, 5.0])
    def test_COMONOTONE_unit_beta_gives_the_WIDEST_NAME_exactly(
        self, horizon: float, sign: float) -> None:
        """At `|β| = 1` every name is driven by `Z` alone, so the basket survives exactly when the
        most-default-prone name does: `Q_B = 1 − Φ(max_i C_i) = min_i Q_i`.

        Taken analytically, because the quadrature cannot integrate the step — and both signs give
        the same answer, which is the tell that it is the *magnitude* of the loading that matters
        here, not its direction."""
        got = BasketSurvival(_copula(betas=(sign,) * len(BETAS))).survival(horizon)
        want = _widest(horizon)
        # Exact in ℝ, a few ULP in floating point: `1 − Φ(max C)` against `min exp(−∫λ)` is a
        # round trip through `Φ`/`Φ⁻¹`, so bit-equality is not available here either (same lesson
        # as the single-name reduction above).
        assert abs(got - want) <= 4.0 * math.ulp(want), (horizon, sign, got, want)

    def test_THE_COMPANION_a_generic_correlation_sits_STRICTLY_between(self) -> None:
        """A limit a no-op satisfies is vacuous (§4): the generic case must be materially inside
        both bounds, or the two reductions above are testing nothing about the integral."""
        got = BasketSurvival(_copula()).survival(HORIZON)
        lower, upper = _independent_product(HORIZON), _widest(HORIZON)
        assert lower < got < upper
        assert got / lower - 1.0 > 0.02, (got, lower)
        assert upper / got - 1.0 > 0.10, (got, upper)

    def test_the_market_factor_integral_CONVERGES_in_the_node_count(self) -> None:
        """A rate, not a tolerance — the product of smooth conditional survivals is smooth in `Z`
        for `|β| < 1`, so convergence is available and is asserted."""
        reference = BasketSurvival(_copula(), nodes=256).survival(HORIZON)
        errors = [
            abs(BasketSurvival(_copula(), nodes=n).survival(HORIZON) / reference - 1.0)
            for n in (4, 8, 16)]
        assert errors[0] > errors[-1], errors
        assert errors[-1] < 1e-07, errors


class TestTheProtectionLegOnABasketCurve:
    """Why `credit_legs.protection_leg` is not reused, measured rather than asserted."""

    def test_the_minus_dQ_form_CONVERGES(self) -> None:
        basket = BasketSurvival(_copula())
        reference = protection_leg_from_survival(CURVE, basket, RECOVERY, HORIZON, 4000)
        errors = [
            abs(protection_leg_from_survival(CURVE, basket, RECOVERY, HORIZON, n) / reference - 1.0)
            for n in (50, 200, 800)]
        for coarse, fine in zip(errors, errors[1:]):
            assert coarse > fine, errors
        assert errors[0] < 1e-04 and errors[-1] < 1e-07, errors

    def test_the_WRONG_PAIRING_cannot_happen_SILENTLY(self) -> None:
        """C13's extension asks for this guard, so here it is, in both tiers.

        **Statically:** `protection_leg` is annotated with the concrete `SurvivalCurve`, while the
        premium leg takes the `SurvivalFunction` protocol — so pyright rejects a basket passed to
        the pillar integrator, and pyright runs in CI.

        **At runtime:** `BasketSurvival` deliberately carries no `.times` and no `.hazard`, because
        it has neither — there are no pillars in an integral. So the wrong pairing raises rather
        than quietly computing the 1.56e-03-wrong number. That is the guard that matters: not a
        tolerance on a bad result, but the bad result being unreachable."""
        basket = BasketSurvival(_copula())
        assert not hasattr(basket, "times"), "a basket survival has no pillars"
        assert not hasattr(basket, "hazard"), "and no piecewise-constant hazard"
        from pricebook.market.credit_legs import protection_leg

        # Called through a deliberately loose alias, not a `type: ignore` — the charter forbids
        # suppressions (§7) and the debt gate counts them (§8, which caught the first draft of
        # this line). The looseness is the point: the static guard is pyright's and is exercised
        # in CI, while what a runtime test can show is that bypassing it still fails loudly.
        dynamic: Callable[..., float] = protection_leg
        with pytest.raises(AttributeError, match="times"):
            dynamic(CURVE, basket, RECOVERY, HORIZON, 50)

    def test_the_PILLAR_HAZARD_form_would_PLATEAU_which_is_why_it_is_not_used(self) -> None:
        """The §4 rate-detector: a model error plateaus where a discretisation error converges.

        `credit_legs.protection_leg` reads one hazard per pillar segment, exact when the hazard is
        piecewise constant and wrong for `Q_B`, whose hazard varies continuously. Fed `Q_B`'s
        implied hazard it sits **1.56e-03** from the truth and **does not improve** with
        refinement — asserted here so the choice of form is evidenced, not preferred."""
        basket = BasketSurvival(_copula())
        truth = protection_leg_from_survival(CURVE, basket, RECOVERY, HORIZON, 4000)
        pillars = sorted({0.0, HORIZON} | {
            t for curve in _curves() for t in curve.times if 0.0 < t < HORIZON})

        def pillar_hazard_form(per_segment: int) -> float:
            area = 0.0
            for lo, hi in zip(pillars, pillars[1:]):
                mid = 0.5 * (lo + hi)
                hazard = -(math.log(basket.survival(mid + 1e-06))
                           - math.log(basket.survival(mid - 1e-06))) / 2e-06
                step = (hi - lo) / per_segment
                previous = CURVE.df_at(lo) * basket.survival(lo)
                for index in range(1, per_segment + 1):
                    t = lo + step * index
                    current = CURVE.df_at(t) * basket.survival(t)
                    area += 0.5 * (previous + current) * step * hazard
                    previous = current
            return (1.0 - RECOVERY) * area

        coarse = abs(pillar_hazard_form(50) / truth - 1.0)
        fine = abs(pillar_hazard_form(200) / truth - 1.0)
        assert coarse > 1e-03, coarse
        assert fine == pytest.approx(coarse, rel=0.05), (coarse, fine)


class TestTheBasketPrices:
    def test_a_basket_struck_AT_its_par_spread_is_worth_ZERO(self) -> None:
        from pricebook.engine.first_to_default_basket import par_spread

        model = _model()
        par = par_spread(_basket(), model)
        out = price(_basket(par), model)
        assert isinstance(out, PricingResult), out
        assert abs(out.pv.amount) < 1e-03, out.pv.amount

    def test_the_value_is_LINEAR_and_decreasing_in_the_contractual_spread(self) -> None:
        model = _model()
        values = [price(_basket(s), model).pv.amount for s in (0.02, 0.04, 0.06)]
        assert values[0] > values[1] > values[2]
        first = values[1] - values[0]
        second = values[2] - values[1]
        assert second == pytest.approx(first, rel=1e-09)

    def test_THE_SPREAD_sits_between_the_WIDEST_name_and_the_SUM(self) -> None:
        """A finance requirement no `Q_B` transcription can fake: first-to-default protection is
        worth at least the widest single name (it covers that name too) and at most the sum (it
        pays once, not `N` times). Measured 210.96bp ≤ 491.28bp ≤ 554.27bp on this fixture."""
        from pricebook.engine.first_to_default_basket import par_spread

        basket_spread = par_spread(_basket(), _model())
        singles = []
        for curve in _curves():
            annuity = risky_pv01(SCHEDULE.periods(), CURVE, curve)
            protection = protection_leg_from_survival(CURVE, curve, RECOVERY, HORIZON, 800)
            singles.append(protection / annuity)
        assert max(singles) < basket_spread < sum(singles), (singles, basket_spread)
        assert basket_spread / max(singles) - 1.0 > 0.5, "and materially above the widest"

    def test_the_premium_leg_REUSES_the_single_name_annuity_unchanged(self) -> None:
        """`risky_pv01` reads only `survival(t)`, so the basket feeds it directly — the reuse the
        spec promised, and the half of it that actually holds."""
        from pricebook.engine.first_to_default_basket import basket_annuity

        basket = BasketSurvival(_copula())
        # `==` is right *here*, unlike the two reductions above: the engine calls this very
        # function with the same periods, the same curve and an equal-valued basket, so the
        # operations and their order are identical and bit-equality is genuinely available. The
        # distinction is the point — bit-equality is a claim about the computation, not the maths.
        assert basket_annuity(_basket(), _model()) == risky_pv01(
            SCHEDULE.periods(), CURVE, basket)


class TestTheMonteCarloCrossCheck:
    """The non-degenerate anchor: simulate the latent variables and count first defaults.

    The engine is **deterministic** — the market-factor integral prices — so this is an oracle and
    not a pricer (the tranche-41 pattern). Two things are pinned, and the second matters more than
    it looks:

    * the simulated basket survival matches `Q_B` within the sampling error, which checks the
      *integral*;
    * the **realised correlation of the simulated `A_i`** reproduces `β_iβ_j`. Because `ρ_ij`
      enters the **factorisation** of the `A_i` rather than a single scalar output, a basket-level
      number is a weak test of it — the tranche-17 lesson: pin a factorisation where it is
      directly observable, which is the realised covariance, not a price.
    """

    PATHS = 40_000
    SEED = 20_261_115

    def _draw(self) -> list[list[float]]:
        """`PATHS` draws of the latent variables, one row per path."""
        from pricebook.foundation.rng import UniformSource
        from pricebook.foundation.variates import standard_normals

        count = len(BETAS)
        source = UniformSource(self.SEED)
        normals = standard_normals(source.uniforms(self.PATHS * (count + 1)))
        rows = []
        for path in range(self.PATHS):
            block = normals[path * (count + 1):(path + 1) * (count + 1)]
            factor, idiosyncratic = block[0], block[1:]
            rows.append([
                beta * factor + math.sqrt(1.0 - beta * beta) * epsilon
                for beta, epsilon in zip(BETAS, idiosyncratic)])
        return rows

    def test_the_SIMULATED_basket_survival_matches_Q_B_within_sampling_error(self) -> None:
        copula = _copula()
        thresholds = [copula.threshold(i, HORIZON) for i in range(len(BETAS))]
        rows = self._draw()
        survived = sum(
            1 for row in rows if all(a >= c for a, c in zip(row, thresholds)))
        estimate = survived / self.PATHS
        error = math.sqrt(estimate * (1.0 - estimate) / self.PATHS)
        want = BasketSurvival(copula).survival(HORIZON)
        assert abs(estimate - want) < 4.0 * error, (estimate, want, error)
        # and the band must be narrow enough to mean something: 4 SE is ~1.5% of the value here,
        # which rules out the independence bound (4.5% away) and the widest-name bound (28% away).
        assert 4.0 * error / want < 0.02, 4.0 * error / want
        assert abs(_independent_product(HORIZON) - want) > 6.0 * error
        assert abs(_widest(HORIZON) - want) > 6.0 * error

    def test_the_REALISED_CORRELATION_reproduces_beta_i_times_beta_j(self) -> None:
        """Every pair individually, which a single basket price could never pin."""
        rows = self._draw()
        count = len(BETAS)
        means = [sum(row[i] for row in rows) / self.PATHS for i in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                covariance = sum(
                    (row[i] - means[i]) * (row[j] - means[j]) for row in rows) / self.PATHS
                want = BETAS[i] * BETAS[j]
                assert covariance == pytest.approx(want, abs=0.02), (i, j, covariance, want)

    def test_THE_PREMISE_the_simulated_variables_have_UNIT_variance(self) -> None:
        """`A_i = βZ + √(1−β²)ε` has unit variance only if the idiosyncratic weight is right — so
        this is where a dropped `√(1−β²)` shows up, independently of any correlation."""
        rows = self._draw()
        for i in range(len(BETAS)):
            values = [row[i] for row in rows]
            mean = sum(values) / self.PATHS
            variance = sum((v - mean) ** 2 for v in values) / self.PATHS
            assert variance == pytest.approx(1.0, abs=0.03), (i, variance)


class TestTheFixtureBlindSPOTSTheSweepFound:
    """Four mutants survived the first sweep, every one of them a fixture gap rather than a bug.

    The most instructive is the last: **the premium schedule began at the valuation date**, so no
    period was historical and the invariant-6 filter had nothing to filter — which is *precisely*
    the blind spot that hid tranche 40's 73% annuity overstatement, found in tranche 41b. Having
    fixed that one tranche earlier, this tranche's fixture reproduced its shape. The convention
    ("schedules start today") is systemic, not per-tranche, and the sweep is what catches it.
    """

    SEASONED = tuple(VAL + timedelta(days=round(182.5 * i)) for i in range(-4, 7))

    def test_the_PV_SCALES_with_the_notional(self) -> None:
        """Nothing pinned the notional: the par reprice is ~0 at any scale and the linearity check
        compares differences, so dropping the factor entirely survived."""
        model = _model()
        small = FirstToDefaultBasket(ENTITIES, SCHEDULE, SPREAD, Money(1_000_000.0, EUR))
        large = FirstToDefaultBasket(ENTITIES, SCHEDULE, SPREAD, Money(10_000_000.0, EUR))
        small_pv = price(small, model).pv.amount
        large_pv = price(large, model).pv.amount
        assert abs(small_pv) > 1.0, "the premise: the value is materially non-zero"
        assert large_pv == pytest.approx(10.0 * small_pv, rel=1e-12)

    def test_the_basket_uses_ONLY_the_names_the_CONTRACT_protects(self) -> None:
        """The copula may carry a wider universe than any one basket trades.

        The first fixture gave the basket *exactly* the copula's three names, so selecting the
        contract's subset and taking all of them were the same operation — a mutation ignoring the
        contract's entity list survived. A fourth name in the copula separates them."""
        extra = CreditName(
            ReferenceEntity("DELTA"), 0.55,
            SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.050, 0.070, 0.090)))
        wider = GaussianCopula(_copula().market, (*_copula().names, extra))
        assert len(wider.names) == len(ENTITIES) + 1, "the premise: the copula is wider"

        on_three = price(_basket(), BasketModel(wider, RECOVERY)).pv.amount
        on_all = price(
            FirstToDefaultBasket((*ENTITIES, extra.entity), SCHEDULE, SPREAD, NOTIONAL),
            BasketModel(wider, RECOVERY)).pv.amount
        assert on_three != on_all
        # and the three-name basket must match the three-name copula exactly: the extra name is
        # ignored, not merely down-weighted
        assert on_three == pytest.approx(price(_basket(), _model()).pv.amount, rel=1e-12)

    def test_a_SEASONED_basket_excludes_its_SETTLED_premiums(self) -> None:
        """Invariant 6, with a schedule that straddles the valuation date — the case the original
        fixture could not express."""
        schedule = CouponSchedule(self.SEASONED, DC)
        historical = [pay for _, pay, _ in schedule.periods() if pay <= VAL]
        assert len(historical) >= 3, "the premise: there are settled periods to drop"

        from pricebook.engine.first_to_default_basket import basket_annuity
        from pricebook.market.credit_legs import risky_pv01

        product = FirstToDefaultBasket(ENTITIES, schedule, SPREAD, NOTIONAL)
        live = [p for p in schedule.periods() if p[1] > VAL]
        want = risky_pv01(live, CURVE, BasketSurvival(_copula()))
        assert basket_annuity(product, _model()) == pytest.approx(want, rel=1e-14)
        # the whole-schedule annuity is materially larger, so this is not a vacuous equality
        whole = risky_pv01(schedule.periods(), CURVE, BasketSurvival(_copula()))
        assert whole / want - 1.0 > 0.5, (whole, want)

    def test_a_NON_POSITIVE_horizon_has_no_protection_left(self) -> None:
        """The behaviour, asserted — but the guard producing it is a **proved equivalent** mutant.

        Removing `if horizon <= 0: return 0.0` changes nothing, bit for bit, over 45
        horizon/step/curve combinations: every survival function is flat at 1 below zero, so each
        difference `Q(prev) − Q(t)` is exactly zero and the sum is `0.0` either way. So this test
        pins the *outcome* and cannot kill that mutation — which is the honest position, not a gap
        to paper over. The guard is kept for symmetry with `protection_leg`, where the same line
        IS load-bearing."""
        basket = BasketSurvival(_copula())
        assert protection_leg_from_survival(CURVE, basket, RECOVERY, 0.0, 800) == 0.0
        assert protection_leg_from_survival(CURVE, basket, RECOVERY, -1.0, 800) == 0.0


class TestTheGuardsAreReachable:
    def test_a_basket_of_ONE_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            FirstToDefaultBasket(ENTITIES[:1], SCHEDULE, SPREAD, NOTIONAL)

    def test_a_basket_naming_an_entity_the_COPULA_lacks_is_a_PricingFailure(self) -> None:
        other = ReferenceEntity("NOT-IN-THE-COPULA")
        out = price(
            FirstToDefaultBasket((ENTITIES[0], other), SCHEDULE, SPREAD, NOTIONAL), _model())
        assert not isinstance(out, PricingResult)
        assert "NOT-IN-THE-COPULA" in out.reason

    def test_DUPLICATE_entities_are_refused(self) -> None:
        """A name twice would be counted twice in the product — a wrong basket, not a wide one."""
        with pytest.raises(ValueError, match="duplicate"):
            FirstToDefaultBasket((ENTITIES[0], ENTITIES[0]), SCHEDULE, SPREAD, NOTIONAL)

    def test_a_MIXED_unit_and_sub_unit_loading_basket_is_REFUSED(self) -> None:
        """The case the quadrature gets silently wrong: a step from the unit-loading names times a
        smooth part. The all-unit case is analytic and the sub-unit case is smooth; the mixture is
        neither, nothing needs it, and a 6%-wrong probability is worse than a refusal (§3.4)."""
        with pytest.raises(ValueError, match="mixes unit and sub-unit"):
            BasketSurvival(_copula(betas=(1.0, 0.60, 0.45))).survival(HORIZON)

    def test_a_RECOVERY_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="recovery"):
            BasketModel(_copula(), 1.4)
