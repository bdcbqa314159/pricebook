"""The LHP synthetic-CDO tranche (L4) — `OKANE-2008` ch 16.2 / 16.7.1. RED, tranche 43.

Vasicek's infinite-homogeneous-pool limit of the one-factor Gaussian copula: the law of large
numbers collapses the conditional loss to a point mass, so the whole loss distribution is closed
form.

    L(T|Z) = (1−R)·Φ((C − βZ)/√(1−β²)),   C = Φ⁻¹(p(T))
    A(K)   = (C − √(1−β²)·Φ⁻¹(K/(1−R))) / β,     Pr(L ≥ K) = Φ(A(K))
    E[min(L,K)] = (1−R)·Φ₂(C, −A(K); **−β**) + K·Φ(A(K))                        (ch 16.7.1)

**Eq 16.1's bivariate normal is confirmed to BE our `Φ₂`, and the `ρ` sign is `−β`** — the check
Cowork asked for, and it was not cosmetic: against an independent computation `+β` is **62% to
240%** wrong while `−β` agrees to **1e-14**. The reference is a *kink-split* market-factor integral
(below), because the naive one cannot establish this.

**Three measurements shaped the suite before any implementation.**

* **The reference had to be split at the kink.** `E[min(L,K)] = ∫ min((1−R)p(T|Z), K) φ(Z)dZ` has a
  **kink** where the cap binds, and Gauss-Hermite handles a kink badly: it oscillates around the
  closed form (+6.5e-03, −8.6e-04, +6.6e-04 at 64/256/370 nodes) instead of converging. Splitting
  the integral at `z* = A(K)` — the cap binds below it, the smooth tail above — agrees to **1e-14**.
  Using the naive integral would have made the *reference* the weaker side and forced a 1e-3 band
  wide enough to hide a real defect (§4, "a reference is code too").
* **The closed form is undefined at exactly the two points its headline identity needs.** `A(K)`
  takes `Φ⁻¹(K/(1−R))`, so it needs `0 < K < 1−R` **strictly** — and `inverse_norm_cdf` refuses
  the endpoints rather than returning `±∞`. But the capital-structure partition runs from `K = 0`
  to `K = 1`. So the two branches are not defensive extras: `K ≤ 0 → 0`, and `K ≥ 1−R → (1−R)p`
  (the cap cannot bind, since `L ≤ 1−R` always). With them the partition identity is **exact to
  2e-16**; without them it raises.
* **The maximum loss is `1−R`, not 1.** Which is why `K ≥ 1−R` is the branch point rather than
  `K ≥ 1`.
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
from pricebook.products.synthetic_cdo_tranche import SyntheticCDOTranche, TrancheBounds

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
CURVE = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0, 5.0), (0.020, 0.026, 0.030, 0.033, 0.036))

#: An index is a reference entity for keying purposes — the LHP has no per-name structure at all
#: (it is the infinite homogeneous limit), so the pool's identity *is* its one default curve.
INDEX = ReferenceEntity("CDX-IG")
#: TERM-STRUCTURED, so `p(T)` is not one number reused at every horizon.
POOL = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.009, 0.016, 0.028))
BETA = 0.60
RECOVERY = 0.40
NOTIONAL = Money(10_000_000.0, EUR)
SPREAD = 0.0500
HORIZON = 5.0
#: An INTERIOR tranche: `K₁ = 0` or `K₂ ≥ 1−R` degenerates a Φ₂ argument into a closed branch.
ATTACH, DETACH = 0.03, 0.07
#: Semi-annual AND straddling the valuation date — §4's tenth degenerate shape, so the historical
#: filter has something to filter.
DATES = tuple(VAL + timedelta(days=round(182.5 * i)) for i in range(-2, 11))
SCHEDULE = CouponSchedule(DATES, DC)


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT_KEY: CURVE}, survivals={INDEX: POOL}))


def _model(beta: float = BETA, recovery: float = RECOVERY) -> LHPModel:
    return LHPModel(_market(), INDEX, beta, recovery)


def _tranche(attach: float = ATTACH, detach: float = DETACH,
             spread: float = SPREAD) -> SyntheticCDOTranche:
    return SyntheticCDOTranche(
        TrancheBounds(attach, detach), SCHEDULE, spread, NOTIONAL, INDEX)


def _pool_default(horizon: float = HORIZON) -> float:
    """`p(T)` computed here, straight off the hazards."""
    return 1.0 - POOL.survival(horizon)


def _reference_capped_loss(
    strike: float, horizon: float = HORIZON, beta: float = BETA,
    recovery: float = RECOVERY, nodes: int = 160) -> float:
    """`E[min(L,K)]` by a market-factor integral **split at the kink** — the independent side.

    The cap binds for `Z` below `z* = A(K)` and not above, so each piece is smooth and
    Gauss-Legendre converges: measured agreement with the closed form is 1e-14, against the naive
    unsplit Gauss-Hermite's oscillating 1e-3.
    """
    import numpy as np

    loss_given_default = 1.0 - recovery
    probability = 1.0 - POOL.survival(horizon)
    if strike <= 0.0:
        return 0.0
    if strike >= loss_given_default:
        return loss_given_default * probability

    threshold = inverse_norm_cdf(probability)
    spread_factor = math.sqrt(1.0 - beta * beta)
    kink = (threshold - spread_factor * inverse_norm_cdf(strike / loss_given_default)) / beta

    raw, weights = np.polynomial.legendre.leggauss(nodes)
    lower, upper = kink, kink + 12.0
    middle, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
    tail = half * sum(
        weight * loss_given_default
        * norm_cdf((threshold - beta * (middle + half * node)) / spread_factor)
        * math.exp(-0.5 * (middle + half * node) ** 2) / math.sqrt(2.0 * math.pi)
        for node, weight in zip(raw, weights))
    return strike * norm_cdf(kink) + tail


class TestTheFixtureIsNotDegenerate:
    def test_the_tranche_is_INTERIOR(self) -> None:
        """`K₁ = 0` and `K₂ ≥ 1−R` each collapse the closed form into a branch."""
        assert 0.0 < ATTACH < DETACH < 1.0 - RECOVERY

    def test_the_correlation_is_strictly_inside_the_unit_interval(self) -> None:
        assert 0.0 < BETA < 1.0

    def test_the_pool_default_curve_is_TERM_STRUCTURED(self) -> None:
        values = [1.0 - POOL.survival(t) for t in (1.0, 3.0, 5.0)]
        assert len(set(round(v, 12) for v in values)) == len(values)
        assert values == sorted(values)

    def test_the_schedule_STRADDLES_the_valuation_date(self) -> None:
        """§4's tenth shape, the one that cost tranches 40 and 42."""
        historical = [pay for _, pay, _ in SCHEDULE.periods() if pay <= VAL]
        assert len(historical) >= 2, historical

    def test_the_accruals_are_NON_UNIT_and_UNEQUAL(self) -> None:
        accruals = [a for _, _, a in SCHEDULE.periods()]
        assert all(abs(a - 1.0) > 0.4 for a in accruals)
        assert len(set(round(a, 12) for a in accruals)) > 1


class TestTheCapitalStructureIdentities:
    """Transcription-proof: derived from the finance, so they survive a misread formula."""

    def test_the_FULL_structure_is_the_pools_expected_loss(self) -> None:
        """`E[min(L,1)] = E[L] = (1−R)p` exactly — `L ≤ 1−R` always, so the cap cannot bind."""
        model = _model()
        want = (1.0 - RECOVERY) * _pool_default()
        assert model.expected_loss_capped(1.0, HORIZON) == pytest.approx(want, rel=1e-14)

    def test_the_MAXIMUM_loss_is_one_minus_R_not_one(self) -> None:
        """Which is why `K ≥ 1−R` is the branch point. Asserted so the boundary is not folklore."""
        model = _model()
        at_bound = model.expected_loss_capped(1.0 - RECOVERY, HORIZON)
        beyond = model.expected_loss_capped(0.95, HORIZON)
        assert at_bound == pytest.approx(beyond, rel=1e-14)

    @pytest.mark.parametrize("beta", [0.20, 0.60, 0.90])
    def test_the_TRANCHES_PARTITION_the_loss(self, beta: float) -> None:
        """`Σ EL([aₖ₋₁,aₖ]) = (1−R)p` over any partition of [0,1], for all parameters.

        Equity + mezzanine + senior losses sum to the pool's loss. This is the identity that
        **requires** the `K ≤ 0` and `K ≥ 1−R` branches: the partition's endpoints are exactly
        where `Φ⁻¹(K/(1−R))` is undefined, and `inverse_norm_cdf` refuses them rather than
        returning `±∞`. With the branches it is exact to 2e-16; without them it raises."""
        model = _model(beta=beta)
        edges = (0.0, 0.03, 0.07, 0.15, 0.30, 1.0)
        total = sum(
            model.tranche_expected_loss(lower, upper, HORIZON)
            for lower, upper in zip(edges, edges[1:]))
        assert total == pytest.approx((1.0 - RECOVERY) * _pool_default(), rel=1e-13)

    def test_the_expected_loss_is_MONOTONE_in_both_bounds(self) -> None:
        model = _model()
        rising_detachment = [
            model.tranche_expected_loss(ATTACH, d, HORIZON) for d in (0.05, 0.10, 0.20)]
        assert rising_detachment == sorted(rising_detachment)
        rising_attachment = [
            model.tranche_expected_loss(a, 0.30, HORIZON) for a in (0.01, 0.05, 0.12)]
        assert rising_attachment == sorted(rising_attachment, reverse=True)

    def test_the_tranche_SURVIVAL_stays_in_the_unit_interval_and_DECREASES(self) -> None:
        model = _model()
        values = [model.tranche_survival(ATTACH, DETACH, t) for t in (0.0, 1.0, 3.0, 5.0)]
        assert values[0] == 1.0
        for earlier, later in zip(values, values[1:]):
            assert 0.0 <= later <= earlier <= 1.0, values

    def test_the_EQUITY_tranche_bears_loss_FIRST(self) -> None:
        """A requirement no transcription can fake: the 0–3% loses more of itself than the 3–7%."""
        model = _model()
        equity = model.tranche_expected_loss(0.0, 0.03, HORIZON) / 0.03
        mezzanine = model.tranche_expected_loss(0.03, 0.07, HORIZON) / 0.04
        assert equity > mezzanine > 0.0, (equity, mezzanine)


class TestTheClosedFormAgainstTheSplitIntegral:
    """Two genuinely different computations of the capped expected loss."""

    @pytest.mark.parametrize("strike", [0.01, 0.03, 0.07, 0.20, 0.45])
    @pytest.mark.parametrize("beta", [0.30, 0.60])
    def test_eq_16_1_IS_our_PHI2_and_the_rho_sign_is_MINUS_beta(
        self, strike: float, beta: float) -> None:
        """The check Cowork asked for, and it discriminates hard: `+β` is 62%–240% wrong."""
        got = _model(beta=beta).expected_loss_capped(strike, HORIZON)
        want = _reference_capped_loss(strike, beta=beta)
        assert got == pytest.approx(want, rel=1e-11), (strike, beta, got, want)

    def test_THE_COMPANION_the_WRONG_rho_sign_is_MATERIALLY_different(self) -> None:
        """Proving the sign check has power rather than merely passing."""
        from pricebook.foundation.bivariate_normal import bivariate_normal_cdf

        strike, beta = 0.03, 0.60
        probability = _pool_default()
        threshold = inverse_norm_cdf(probability)
        loss_given_default = 1.0 - RECOVERY
        exceedance = (threshold - math.sqrt(1.0 - beta * beta)
                      * inverse_norm_cdf(strike / loss_given_default)) / beta
        wrong = (loss_given_default
                 * bivariate_normal_cdf(threshold, -exceedance, +beta)
                 + strike * norm_cdf(exceedance))
        want = _reference_capped_loss(strike, beta=beta)
        assert abs(wrong / want - 1.0) > 0.5, (wrong, want)

    def test_the_reference_is_the_SPLIT_integral_because_the_naive_one_OSCILLATES(self) -> None:
        """§4 — a reference is code too, and its accuracy must be argued where it is used.

        The unsplit Gauss-Hermite integrand has a kink where the cap binds, so refining the node
        count does not converge: it oscillates around the truth at the 1e-3 level. Asserted, so
        nobody replaces the split reference with the obvious one."""
        from pricebook.foundation.quadrature import standard_normal_expectation

        strike, beta = 0.03, 0.60
        probability = _pool_default()
        threshold = inverse_norm_cdf(probability)
        loss_given_default = 1.0 - RECOVERY
        scale = math.sqrt(1.0 - beta * beta)
        truth = _reference_capped_loss(strike, beta=beta)

        errors = [
            abs(standard_normal_expectation(
                lambda z: min(loss_given_default
                              * norm_cdf((threshold - beta * z) / scale), strike),
                nodes=n) / truth - 1.0)
            for n in (64, 256, 370)]
        assert min(errors) > 1e-05, f"the naive integral never gets close: {errors}"
        assert not (errors[0] > errors[1] > errors[2]), f"and it does not converge: {errors}"


class TestTheZeroCorrelationReduction:
    """`β → 0`: the law of large numbers removes the dispersion entirely."""

    @pytest.mark.parametrize("strike", [0.02, 0.10, 0.50])
    def test_the_loss_becomes_a_POINT_MASS_at_one_minus_R_times_p(self, strike: float) -> None:
        """So `E[min(L,K)] = min((1−R)p, K)` exactly — measured machine-exact by `β = 0.01`."""
        want = min((1.0 - RECOVERY) * _pool_default(), strike)
        got = _model(beta=1e-04).expected_loss_capped(strike, HORIZON)
        assert got == pytest.approx(want, rel=1e-12), (got, want)

    def test_THE_COMPANION_a_live_correlation_is_MATERIALLY_dispersed(self) -> None:
        """A reduction a no-op satisfies is vacuous: at the fixture's `β` the loss is genuinely
        spread, so the Φ₂ term is doing work."""
        strike = 0.02
        point_mass = min((1.0 - RECOVERY) * _pool_default(), strike)
        dispersed = _model().expected_loss_capped(strike, HORIZON)
        assert abs(dispersed / point_mass - 1.0) > 0.10, (dispersed, point_mass)


class TestTheTranchePrices:
    def test_a_tranche_struck_AT_its_par_spread_is_worth_ZERO(self) -> None:
        from pricebook.engine.synthetic_cdo_tranche import par_spread

        model = _model()
        out = price(_tranche(spread=par_spread(_tranche(), model)), model)
        assert isinstance(out, PricingResult), out
        assert abs(out.pv.amount) < 1e-03, out.pv.amount

    def test_the_value_is_LINEAR_and_decreasing_in_the_contractual_spread(self) -> None:
        model = _model()
        values = [price(_tranche(spread=s), model).pv.amount for s in (0.02, 0.05, 0.08)]
        assert values[0] > values[1] > values[2]
        assert (values[2] - values[1]) == pytest.approx(values[1] - values[0], rel=1e-09)

    def test_the_EQUITY_tranche_is_QUOTED_WIDER_than_the_mezzanine(self) -> None:
        """A finance requirement: first loss costs more to protect."""
        from pricebook.engine.synthetic_cdo_tranche import par_spread

        model = _model()
        equity = par_spread(_tranche(0.0, 0.03), model)
        mezzanine = par_spread(_tranche(0.03, 0.07), model)
        senior = par_spread(_tranche(0.15, 0.30), model)
        assert equity > mezzanine > senior > 0.0, (equity, mezzanine, senior)

    def test_the_PV_SCALES_with_the_notional(self) -> None:
        model = _model()
        small = SyntheticCDOTranche(
            TrancheBounds(ATTACH, DETACH), SCHEDULE, SPREAD, Money(1_000_000.0, EUR), INDEX)
        large = _tranche()
        assert abs(price(small, model).pv.amount) > 1.0
        assert price(large, model).pv.amount == pytest.approx(
            10.0 * price(small, model).pv.amount, rel=1e-12)

    def test_the_premium_leg_pays_on_the_AMORTISING_outstanding_notional(self) -> None:
        """The tranche is a CDS on its own survival, so the premium leg is `risky_pv01(Q)` — and
        the outstanding notional shrinking as losses eat the tranche *is* that `Q`."""
        from pricebook.engine.synthetic_cdo_tranche import tranche_annuity
        from pricebook.market.credit_legs import risky_pv01

        model = _model()
        live = [p for p in SCHEDULE.periods() if p[1] > VAL]
        survival = model.tranche_survival_curve(ATTACH, DETACH)
        assert tranche_annuity(_tranche(), model) == risky_pv01(live, CURVE, survival)


class TestTheGuardsAreReachable:
    def test_an_INVERTED_tranche_is_refused(self) -> None:
        with pytest.raises(ValueError, match="detachment"):
            TrancheBounds(0.07, 0.03)

    def test_a_ZERO_WIDTH_tranche_is_refused(self) -> None:
        with pytest.raises(ValueError, match="detachment"):
            TrancheBounds(0.05, 0.05)

    def test_bounds_OUTSIDE_the_unit_interval_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unit interval"):
            TrancheBounds(-0.01, 0.05)
        with pytest.raises(ValueError, match="unit interval"):
            TrancheBounds(0.05, 1.5)

    def test_a_BETA_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="correlation"):
            LHPModel(_market(), INDEX, 1.4, RECOVERY)

    def test_a_ZERO_BETA_is_refused_rather_than_dividing_by_it(self) -> None:
        """`A(K)` divides by `β`. The `β → 0` limit is the reduction above, approached not reached:
        exactly zero has a closed form (`min((1−R)p, K)`) but no `A(K)`, so it is refused rather
        than returning an infinity (§3.4)."""
        with pytest.raises(ValueError, match="correlation"):
            LHPModel(_market(), INDEX, 0.0, RECOVERY)

    def test_a_RECOVERY_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="recovery"):
            LHPModel(_market(), INDEX, BETA, 1.4)

    def test_an_UNKNOWN_pool_entity_is_a_PricingFailure(self) -> None:
        other = ReferenceEntity("NOT-AN-INDEX")
        out = price(
            SyntheticCDOTranche(
                TrancheBounds(ATTACH, DETACH), SCHEDULE, SPREAD, NOTIONAL, other),
            _model())
        assert not isinstance(out, PricingResult)
        assert "NOT-AN-INDEX" in out.reason
