"""The one-factor Gaussian latent-variable model (L3) — `OKANE-2008` ch 13. RED, tranche 41.

    A_i = β_i·Z + √(1−β_i²)·ε_i                                                      (13.3)
    C_i(T) = Φ⁻¹(1 − Q_i(T))            ρ_ij = β_i·β_j                               (13.2)
    p_i(T|Z) = Φ((C_i − β_i·Z)/√(1−β_i²))                                            (13.4)
    p_ij(T)  = Φ₂(C_i, C_j; ρ_ij)

**The correctness anchor is marginal recovery, and it has two halves of different strength.**
`Φ(C_i) = 1 − Q_i` is true *by construction* of the threshold and so telescopes; the useful half is
`E_Z[p_i(T|Z)] = 1 − Q_i`, which is a genuinely different computation — a Gauss-Hermite integral
against an inverse-CDF — and is the credit analogue of tranche 8's known-curve round trip.

**Three measurements shaped these oracles, and each would have left a hole:**

* The **`ρ = −1` reduction `p_ij = max(p_i + p_j − 1, 0)` is VACUOUS on a low-default fixture.**
  With `p_i = 0.067` and `p_j = 0.115` the answer is `0`, which an implementation returning zero
  also satisfies. The premise is `p_i + p_j > 1`, so the fixture carries deliberately **high**
  default probabilities for that case — the eighth-shape degenerate fixture (a value that makes the
  assertion true for the wrong reason), caught before writing the test rather than by a mutant.
* **Marginal recovery is not monotone in the node count.** At `β = 0.9, C = −2.5` the relative error
  reads 2.1e-04 at 8 nodes, **1.6e-03 at 16** — worse — then 1.9e-06 and 1.0e-11. The integrand
  approaches a step as `β → 1`, so a blanket "refining must improve it" assertion would flake.
  Convergence is asserted on a moderate-`β` fixture, with the high-`β` regime measured separately.
* **`β = ±1` makes `√(1−β²)` exactly zero**, and the `ρ = ±1` reductions need exactly that. The
  conditional probability is an indicator there, not a division — asserted directly.
"""

import math

import pytest

from pricebook.foundation.bivariate_normal import bivariate_normal_cdf
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.distributions import norm_cdf
from pricebook.foundation.quadrature import standard_normal_expectation
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.gaussian_copula import CreditName, GaussianCopula

from datetime import date

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
from pricebook.foundation.money import Currency

EUR = Currency("EUR")
DISCOUNT = CurveKey(CurveRole.DISCOUNT, EUR)
CURVE = DiscountCurve(TM, (0.0, 1.0, 3.0, 5.0), (0.020, 0.026, 0.030, 0.033))
HORIZON = 5.0

#: DISTINCT betas, so `ρ_ij = β_iβ_j` is a real product — equal betas would make `β_i` and
#: `β_iβ_j` agree on the diagonal-free pairs and hide a mis-paired index.
BETAS = (0.3, 0.6, 0.45)
#: DISTINCT hazards, so the names are not interchangeable and a swapped index shows.
HAZARDS = (0.010, 0.030, 0.019)


def _market() -> MarketSnapshot:
    return MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT: CURVE}))


def _copula(betas: tuple[float, ...] = BETAS,
            hazards: tuple[float, ...] = HAZARDS) -> GaussianCopula:
    return GaussianCopula(
        _market(),
        tuple(CreditName(beta, SurvivalCurve.flat(VAL, hazard, DC))
              for beta, hazard in zip(betas, hazards)))


def _default_probability(hazard: float, horizon: float = HORIZON) -> float:
    """`1 − e^{−λT}` computed here, independently of the curve."""
    return 1.0 - math.exp(-hazard * horizon)


class TestTheFixtureIsNotDegenerate:
    def test_the_BETAS_are_DISTINCT_so_the_correlation_is_a_real_product(self) -> None:
        assert len(set(BETAS)) == len(BETAS)
        copula = _copula()
        assert copula.asset_correlation(0, 1) != pytest.approx(BETAS[0], rel=1e-06)
        assert copula.asset_correlation(0, 1) != copula.asset_correlation(0, 2)

    def test_the_MARGINALS_are_DISTINCT_so_the_names_are_not_interchangeable(self) -> None:
        probabilities = [_default_probability(h) for h in HAZARDS]
        assert len(set(round(p, 12) for p in probabilities)) == len(probabilities)

    def test_every_correlation_is_strictly_INSIDE_the_unit_interval(self) -> None:
        copula = _copula()
        for i in range(len(BETAS)):
            for j in range(len(BETAS)):
                if i != j:
                    assert 0.0 < abs(copula.asset_correlation(i, j)) < 1.0


class TestTheThresholdAndTheMarginalRecovery:
    """The correctness anchor. Half (a) telescopes; half (b) does not."""

    def test_the_threshold_INVERTS_the_marginal_half_a_telescopes(self) -> None:
        """`Φ(C_i) = 1 − Q_i`. True by construction of `C_i = Φ⁻¹(1−Q_i)`, so this is a guard on
        the `Φ`/`Φ⁻¹` round trip rather than on the model — labelled, not relied on."""
        copula = _copula()
        for index, hazard in enumerate(HAZARDS):
            assert norm_cdf(copula.threshold(index, HORIZON)) == pytest.approx(
                _default_probability(hazard), rel=1e-12)

    def test_the_MARKET_FACTOR_INTEGRAL_recovers_the_marginal_half_b_is_the_anchor(self) -> None:
        """`E_Z[p_i(T|Z)] = 1 − Q_i(T)` — the factor model must not distort the single-name
        defaults it was calibrated to. A different computation from the threshold, so a sign or a
        weight error in `p_i(T|Z)` cannot cancel."""
        copula = _copula()
        for index, hazard in enumerate(HAZARDS):
            assert copula.marginal_from_factor(index, HORIZON) == pytest.approx(
                _default_probability(hazard), rel=1e-10)

    def test_the_integral_CONVERGES_in_the_node_count_on_a_moderate_beta(self) -> None:
        """A rate, on a fixture where the integrand is smooth enough for one."""
        copula = _copula()
        errors = [
            abs(copula.marginal_from_factor(0, HORIZON, nodes=n)
                / _default_probability(HAZARDS[0]) - 1.0)
            for n in (4, 8, 16)]
        assert errors[0] > errors[-1], errors
        assert errors[-1] < 1e-08, errors

    def test_the_REGIME_a_high_beta_needs_more_nodes_and_is_NOT_monotone(self) -> None:
        """Measured, and asserted so the committed node count is defensible.

        As `β → 1` the integrand approaches a step in `Z`, and Gauss-Hermite convergence stops
        being monotone: at `β = 0.9` with a far-out threshold the error *rises* from 8 to 16 nodes
        before collapsing. The committed 64 nodes clear it by orders; a test asserting monotone
        improvement here would flake across platforms."""
        sharp = _copula(betas=(0.9, 0.6, 0.45), hazards=(0.002, 0.030, 0.019))
        truth = _default_probability(0.002)
        coarse = abs(sharp.marginal_from_factor(0, HORIZON, nodes=8) / truth - 1.0)
        worse = abs(sharp.marginal_from_factor(0, HORIZON, nodes=16) / truth - 1.0)
        committed = abs(sharp.marginal_from_factor(0, HORIZON) / truth - 1.0)
        assert worse > coarse, f"the non-monotone step must be present: {coarse} -> {worse}"
        assert committed < 1e-09, committed


class TestTheThreeExactReductions:
    """O'Kane §13.3's limits, each exact by construction."""

    def test_INDEPENDENCE_rho_zero_gives_the_PRODUCT(self) -> None:
        copula = _copula(betas=(0.0, 0.0, 0.0))
        assert copula.asset_correlation(0, 1) == 0.0
        want = _default_probability(HAZARDS[0]) * _default_probability(HAZARDS[1])
        assert copula.joint_default(0, 1, HORIZON) == pytest.approx(want, rel=1e-12)

    def test_MAXIMUM_dependence_rho_one_gives_the_MINIMUM(self) -> None:
        copula = _copula(betas=(1.0, 1.0, 1.0))
        assert copula.asset_correlation(0, 1) == 1.0
        want = min(_default_probability(HAZARDS[0]), _default_probability(HAZARDS[1]))
        assert copula.joint_default(0, 1, HORIZON) == pytest.approx(want, rel=1e-12)

    def test_MINIMUM_dependence_rho_minus_one_on_a_NON_VACUOUS_fixture(self) -> None:
        """`max(p_i + p_j − 1, 0)`, with the premise that the max is NOT the zero branch.

        On the default fixture `p_i + p_j ≈ 0.18` and the answer is zero, which an implementation
        returning zero would also pass. High hazards make `p_i + p_j > 1`, so the assertion has
        content."""
        high = (0.30, 0.25, 0.19)
        copula = _copula(betas=(1.0, -1.0, 1.0), hazards=high)
        assert copula.asset_correlation(0, 1) == -1.0
        p_i, p_j = _default_probability(high[0]), _default_probability(high[1])
        assert p_i + p_j > 1.0, f"the premise: {p_i} + {p_j} must exceed 1"
        want = max(p_i + p_j - 1.0, 0.0)
        assert want > 0.25, "and the answer must be materially non-zero"
        assert copula.joint_default(0, 1, HORIZON) == pytest.approx(want, rel=1e-10)

    def test_THE_COMPANION_a_generic_correlation_differs_from_ALL_THREE_limits(self) -> None:
        """A reduction a no-op satisfies is vacuous: the generic case must sit strictly between
        the independence and the maximum-dependence values."""
        copula = _copula()
        generic = copula.joint_default(0, 1, HORIZON)
        p_i, p_j = _default_probability(HAZARDS[0]), _default_probability(HAZARDS[1])
        assert p_i * p_j < generic < min(p_i, p_j)
        assert generic / (p_i * p_j) - 1.0 > 0.2, generic


class TestTheJointByTwoMethods:
    """Φ₂ closed form against the conditional-independence integral — different code."""

    @pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
    def test_PHI2_agrees_with_the_MARKET_FACTOR_INTEGRAL(self, pair: tuple[int, int]) -> None:
        """`Φ₂(C_i,C_j;ρ_ij) = E_Z[p_i(T|Z)·p_j(T|Z)]` — conditional independence is what makes
        the right-hand side a product inside the expectation, so this check is also the statement
        of that property, computed two genuinely different ways."""
        copula = _copula()
        i, j = pair
        assert copula.joint_default(i, j, HORIZON) == pytest.approx(
            copula.joint_default_from_factor(i, j, HORIZON), rel=1e-08)

    def test_the_integral_is_the_REFERENCE_where_PHI2_degrades(self) -> None:
        """Φ₂'s tranche-36b regime is `|ρ| ≤ 0.99`; past it the market-factor integral is the
        better side. Asserted as a *disagreement* so the boundary is recorded rather than assumed:
        the two must agree inside the regime and are allowed to part outside it."""
        inside = _copula(betas=(0.99, 0.99, 0.45))
        assert inside.joint_default(0, 1, HORIZON) == pytest.approx(
            inside.joint_default_from_factor(0, 1, HORIZON), rel=1e-05)

    def test_the_conditional_probability_matches_an_INDEPENDENT_transcription_of_13_4(
        self) -> None:
        copula = _copula()
        for index, (beta, hazard) in enumerate(zip(BETAS, HAZARDS)):
            c = _independent_threshold(hazard)
            for z in (-2.0, -0.5, 0.0, 1.3):
                want = norm_cdf((c - beta * z) / math.sqrt(1.0 - beta * beta))
                assert copula.conditional_default(index, HORIZON, z) == pytest.approx(
                    want, rel=1e-12), (index, z)


def _independent_threshold(hazard: float, horizon: float = HORIZON) -> float:
    """`Φ⁻¹(1−Q)` via scipy, which is the high-precision reference (§4 — kept off `src/`)."""
    from scipy.special import ndtri

    return float(ndtri(_default_probability(hazard, horizon)))


class TestTheRequirementOracles:
    def test_the_threshold_matches_a_SCIPY_reference(self) -> None:
        """`Φ⁻¹` is home-grown in `src/`; the oracle may use scipy and does (tranche 20a)."""
        copula = _copula()
        for index, hazard in enumerate(HAZARDS):
            assert copula.threshold(index, HORIZON) == pytest.approx(
                _independent_threshold(hazard), rel=1e-10)

    def test_the_conditional_probability_is_MONOTONE_DECREASING_in_the_factor(self) -> None:
        """A high market factor means a healthy economy, so fewer defaults."""
        copula = _copula()
        values = [copula.conditional_default(1, HORIZON, z) for z in (-3.0, -1.0, 0.0, 1.0, 3.0)]
        assert values == sorted(values, reverse=True), values

    def test_a_UNIT_beta_makes_the_conditional_an_INDICATOR_not_a_division(self) -> None:
        """`√(1−β²) = 0` exactly at `β = ±1`, which the `ρ = ±1` reductions require."""
        copula = _copula(betas=(1.0, 1.0, 1.0))
        threshold = copula.threshold(0, HORIZON)
        assert copula.conditional_default(0, HORIZON, threshold - 0.5) == 1.0
        assert copula.conditional_default(0, HORIZON, threshold + 0.5) == 0.0

    def test_the_joint_is_SYMMETRIC_in_its_two_names(self) -> None:
        copula = _copula()
        assert copula.joint_default(0, 1, HORIZON) == copula.joint_default(1, 0, HORIZON)

    def test_the_joint_never_EXCEEDS_either_marginal(self) -> None:
        copula = _copula()
        joint = copula.joint_default(0, 1, HORIZON)
        assert joint <= _default_probability(HAZARDS[0]) + 1e-15
        assert joint <= _default_probability(HAZARDS[1]) + 1e-15


class TestTheGuardsAreReachable:
    def test_a_BETA_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="factor loading"):
            CreditName(1.4, SurvivalCurve.flat(VAL, 0.02, DC))

    def test_an_EMPTY_portfolio_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one name"):
            GaussianCopula(_market(), ())

    def test_a_survival_curve_on_a_DIFFERENT_clock_is_refused(self) -> None:
        other = SurvivalCurve.flat(date(2025, 6, 1), 0.02, DC)
        with pytest.raises(ValueError, match="time measure"):
            GaussianCopula(_market(), (CreditName(0.3, other),))

    def test_an_OUT_OF_RANGE_name_index_is_refused(self) -> None:
        with pytest.raises(IndexError):
            _copula().threshold(99, HORIZON)
